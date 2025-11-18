import torch
import torch.nn as nn
import argparse
import ast
import gc
from datetime import datetime
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from helper_datadict import *
from tqdm import tqdm
import copy
from time import sleep
from pytorch_metric_learning import losses, distances
from ModelArchitecture.Losses import PixelwiseContrastiveLoss, LocalGlobalPixelwiseContrastiveLoss
from scipy.ndimage import find_objects, gaussian_filter
from skimage.measure import label, regionprops
from utils import collate_pad, collate_resize, collate_padmax
from models.unet_model_targeted_dropout import UNet4_new
from unet_pruning_functions_zero_out import pruning_function
from pruning_tools.unet_zero_out_and_add_hook import zero_out_and_add_hook_equivalent


import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


IN = 0
OUT = 1


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def Lesion_Metrics(pred, gt, threshold=0.5):
    #print(len(pred), len(gt))
    pred_labeled = label(pred > threshold)
    gt_labeled = label(gt > 0)
    
    objects = find_objects(gt_labeled)
    tp = 0
    fp = 0
    fn = 0

    for obj in objects:
        if np.any(pred_labeled[obj] > 0):
            tp += 1
        else:
            fn += 1
    
    objects = find_objects(pred_labeled)

    for obj in objects:
        if np.any(gt_labeled[obj] > 0):
            continue
        else:
            fp += 1

    # Calculate recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Calculate precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return recall, precision, f1_score


CURRENT_DIRECTORY = os.getcwd()

RESULTS_DIR = CURRENT_DIRECTORY + '/results/'

# composed_transform = transforms.Compose([
#             RandomRotation3D([-10,10],p=0.8),
#             RandomNoise3D(p=0.8),
#             RandomBrightness3D(p=0.8),
#             #ToTensor3D(True)
#             ])

# Define a cosine similarity-based contrastive loss
class CosineContrastiveLoss(losses.ContrastiveLoss):
    def __init__(self, pos_margin=1, neg_margin=0):
        super().__init__(pos_margin=pos_margin, neg_margin=neg_margin, distance=distances.CosineSimilarity())


def apply_transformations_to_batch(batch, transform):
    transformed_batch = {'input': [], 'gt': []}
    for sample in batch:
        transformed_sample = transform(sample)
        transformed_batch['input'].append(torch.Tensor(transformed_sample['input']))
        transformed_batch['gt'].append(torch.Tensor(transformed_sample['gt']))
    transformed_batch['input'] = torch.stack(transformed_batch['input'])
    transformed_batch['gt'] = torch.stack(transformed_batch['gt'])
    return transformed_batch


def get_model_size_mb(model):
    total_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_size_mb = total_size_bytes / (1024 * 1024)
    return total_size_mb

def count_params_val_zero(model):
    #count the number of parameters in the model that have a value of 0
    zero_params = sum((p == 0).sum().item() for p in model.parameters())
    return zero_params

def count_params(model):
    return sum(p.numel() for p in model.parameters()) - count_params_val_zero(model)

def replace_layer_new(model, layer_name, new_layer):
    """
    Replace a layer in the model with the specified new layer.
    
    Args:
        model: The PyTorch model
        layer_name: The name of the layer to replace (e.g., 'encoder2conv2', 'upconv4', 'encoder2norm2')
        new_layer: The new layer to replace with
    
    Returns:
        The model with the replaced layer
    """
    # Iterate through all named modules in the model to find the exact layer
    found = False
    for name, module in model.named_modules():
        if name.endswith(layer_name):
            # Found the layer to replace
            # Now find the parent module
            parent_name = name[:-(len(layer_name)+1)] if '.' in name else ''  # Get parent module name
            if parent_name:
                # Navigate to the parent module
                parent_module = model
                for part in parent_name.split('.'):
                    if part.isdigit():
                        parent_module = parent_module[int(part)]
                    else:
                        parent_module = getattr(parent_module, part)
                
                # Get the attribute name within the parent
                attr_name = layer_name
                
                # Replace the layer
                setattr(parent_module, attr_name, new_layer)
            else:
                # The layer is at the top level
                setattr(model, layer_name, new_layer)
            
            found = True
            break
    
    if not found:
        raise ValueError(f"Layer '{layer_name}' not found in the model")
    
    return model

def get_reduced_model(model, zero_out_dict, device = 'cpu'):
    #move the old model to cpu first
    model.to('cpu')

    for layer_name, pruned_filters in zero_out_dict.items():
        # Find the layer in the model
        layer = None
        for name, module in model.named_modules():
            if name.endswith(layer_name):
                layer = module
                break
        
        if layer is None:
            print(f"Warning: Layer {layer_name} not found in model")
            continue

        layer_type = None
        if isinstance(layer, torch.nn.Conv3d):
            layer_type = "conv"
        elif isinstance(layer, torch.nn.BatchNorm3d):
            layer_type = "bn"
        elif isinstance(layer, torch.nn.ConvTranspose3d):
            layer_type= "convT"
        else:
            print(f"Warning: Layer type {type(layer)} not supported for {layer_name}")
            continue

        if layer_type == "conv":
            out_indices = set([entry[0] for entry in pruned_filters if entry[1] == OUT])
            in_indices = set([entry[0] for entry in pruned_filters if entry[1] == IN])
            new_out_channels = layer.out_channels - len(out_indices)
            new_in_channels = layer.in_channels - len(in_indices)
            new_conv = nn.Conv3d(new_in_channels, new_out_channels, layer.kernel_size, layer.stride, layer.padding, layer.dilation, layer.groups, layer.bias is not None)

            out_indices_to_retain = [i for i in range(layer.out_channels) if i not in out_indices]
            in_indices_to_retain = [i for i in range(layer.in_channels) if i not in in_indices]
            new_conv.weight.data = layer.weight.data[out_indices_to_retain, :, ...].clone()[:, in_indices_to_retain, ...]

            if layer.bias is not None:
                new_conv.bias.data = layer.bias.data[out_indices_to_retain].clone()
            layer = new_conv
        
        elif layer_type == "bn":
            indices = set([entry[0] for entry in pruned_filters if entry[1] == OUT])
            new_num_features = layer.num_features - len(indices)
            new_layer = nn.BatchNorm3d(new_num_features)
            indices_to_retain = [i for i in range(layer.num_features) if i not in indices]
            new_layer.weight.data = layer.weight.data[indices_to_retain].clone()
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data[indices_to_retain].clone()
            new_layer.running_mean = layer.running_mean[indices_to_retain].clone()
            new_layer.running_var = layer.running_var[indices_to_retain].clone()
            layer = new_layer
        
        elif layer_type == "convT":
            indices = set([entry[0] for entry in pruned_filters if entry[1] == OUT])
            new_num_features = layer.in_channels - len(indices)
            new_layer = nn.ConvTranspose3d(new_num_features, layer.out_channels, layer.kernel_size, layer.stride, layer.padding, layer.output_padding, layer.groups, layer.bias is not None, layer.dilation)

            indices_to_retain = [i for i in range(layer.in_channels) if i not in indices]
            new_layer.weight.data = layer.weight.data[indices_to_retain].clone()
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data.clone()
            layer = new_layer
        
        model = replace_layer_new(model, layer_name, layer)
    
    model.to(device)
    return model

    


def helper_save_model(round,model,round_loss,round_dice,model_type,best_dice=False):
    if(not best_dice):
        loss_type = '_state_dict'
    else:
        loss_type = '_state_dict_best_dice'
    torch.save({
    'round': round,
    'model_state_dict': model.state_dict(),
    'loss': round_loss,
    'dice':round_dice
    }, model_type+loss_type+str(round)+'.pth')

def save_bn_stats(model):
    """Save running mean and var of batch norm layers"""
    bn_stats = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm3d, nn.BatchNorm2d)):
            bn_stats[name] = {
                'running_mean': module.running_mean.clone(),
                'running_var': module.running_var.clone(),
                'weight': module.weight.clone() if module.weight is not None else None,
                'bias': module.bias.clone() if module.bias is not None else None
            }
    return bn_stats

def load_bn_stats(model, bn_stats):
    """Restore running mean and var of batch norm layers"""
    for name, module in model.named_modules():
        if name in bn_stats:
            # Non-gradient tensors can be copied directly
            module.running_mean.copy_(bn_stats[name]['running_mean'])
            module.running_var.copy_(bn_stats[name]['running_var'])
            
            # For parameters that require gradients, we need to handle differently
            if bn_stats[name]['weight'] is not None:
                with torch.no_grad():  # Temporarily disable gradient tracking
                    module.weight.copy_(bn_stats[name]['weight'])
            if bn_stats[name]['bias'] is not None:
                with torch.no_grad():  # Temporarily disable gradient tracking
                    module.bias.copy_(bn_stats[name]['bias'])


def helper_save_model(round,unet, zero_out_dict_lists, round_loss,round_dice,model_type,best_dice=False, local_model=False, before_prune=False, local_epoch_num=-1, local_dices_before=None, local_dices_after=None, best_before_prune=False, best_before_agg=False, best_dice_before_prune=0, best_dice_before_agg=0, best_dice_before_prune_round=0, best_dice_before_agg_round=0, client='client0', save_path_dict = dict(), bn_stats_unet=None):

    if not best_dice:
        loss_type = '_state_dict'
    else:
        loss_type = '_state_dict_best_dice'
    
    if local_model:
        loss_type = loss_type + '_local_model_client_' + client
    
        if before_prune:
            loss_type = loss_type + '_before_prune'
        else:
            loss_type = loss_type + '_after_prune'
    
    if best_before_prune:
        loss_type += '_best_before_prune'
    
    if best_before_agg:
        loss_type += '_best_before_agg'
    
    if loss_type in save_path_dict.keys():
        path = save_path_dict[loss_type]
        #remove the previous file if it exists
        if os.path.exists(path):
            os.remove(path)
    
    save_path_dict[loss_type] = model_type+loss_type+str(round)+'.pth'
    
    
    torch.save({
        'round': round,
        'unet_state_dict': unet.state_dict(),
        'lists_zero_out_dicts': zero_out_dict_lists,
        'loss': round_loss,
        'dice':round_dice,
        'local_dices_before': local_dices_before,
        'local_dices_after': local_dices_after,
        'best_dice_before_prune': best_dice_before_prune,
        'best_dice_before_agg':best_dice_before_agg,
        'best_dice_before_prune_round':best_dice_before_prune_round,
        'best_dice_before_agg_round':best_dice_before_agg_round,
        'local_epoch_num': local_epoch_num,
        'save_path_dict': save_path_dict,
        'bn_stats_unet': bn_stats_unet
    }, save_path_dict[loss_type])



def helper_train(unet, train_loader, optimizer, criterion, num_epochs, device, lmd=0, mu_lg=1, mu=1, include_lg_pw=False, include_lpw=False, global_unet=None, global_segmenter=None, method="FedAvg"):
    
    unet.train()
    losses = []
    dices = []

    pixel_contrast_loss = PixelwiseContrastiveLoss()
    lg_pw_loss = LocalGlobalPixelwiseContrastiveLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        total_dice = 0
        for _, data in enumerate(train_loader):
            image = data['input'].to(device)
            label = data['gt'].to(device)

            optimizer.zero_grad()
            #print("Image shape: ", image.shape)
            #print("Label shape: ", label.shape)
            output, bottleneck = unet(image)
            #print("Intermediate output shape: ", output.shape)
            #print("Output shape: ", output.shape)
            loss = criterion(output, label, wt=label)

            if include_lg_pw:
                g_output, g_bottleneck = global_unet(image)
                loss_lg_pw = lg_pw_loss(bottleneck, g_bottleneck, label)
                loss += mu_lg*loss_lg_pw
            
            if include_lpw:
                loss += mu*pixel_contrast_loss(bottleneck, label)
            
            if method == "FedProx":
                diff_norm_squared = 0
                for param1, param2 in zip(global_unet.parameters(), unet.parameters()):
                    diff_norm_squared += torch.norm(param1.data - param2.data) ** 2
                loss += lmd / 2 * diff_norm_squared

            #print(loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_dice += Dice_Score(output.cpu().detach().numpy(), label.cpu().detach().numpy())
            

        losses.append(total_loss/len(train_loader))
        dices.append(total_dice/len(train_loader))
    
    return losses, dices

def helper_validate(unet, val_loader, criterion, device, return_inference_time=False, validate_global=False):
    unet.eval()
    total_loss = 0
    total_dice = 0
    f1_score_total = 0
    inf_times = []
    with torch.no_grad():
        for _, data in enumerate(val_loader):
            image = data['input'].to(device)
            label = data['gt'].to(device)
            start = time.time()
            output, bottleneck = unet(image)
            end = time.time()
            inf_times.append(end-start)
            loss = criterion(output, label, wt=label)
            total_loss += loss.item()
            total_dice += Dice_Score(output.cpu().detach().numpy(), label.cpu().detach().numpy())
            if validate_global:
                #also calc the lesion level scores
                _, precision, f1 = Lesion_Metrics(output.cpu().detach().numpy(), label.cpu().detach().numpy())
                f1_score_total += f1

    return (total_loss/len(val_loader), total_dice/len(val_loader)) if not return_inference_time else (total_loss/len(val_loader), total_dice/len(val_loader), np.mean(np.array(inf_times)))


#this function aggregates the model weights using the weights provided.
def aggregate(client_models, weights, device="cpu"):
    global_model = copy.deepcopy(client_models[0])
    global_model.to(device)
    global_model.train()

    for global_param in global_model.parameters():
        global_param.data = torch.zeros_like(global_param.data)

    for client_model, weight in zip(client_models, weights):
        for global_param, client_param in zip(global_model.parameters(), client_model.parameters()):
            global_param.data += weight * client_param.data

    return global_model


def smart_aggregation(client_models, zero_dicts, weights, device="cpu"):
    global_model = copy.deepcopy(client_models[0])
    global_model.train()
    global_model.to(device)

    temp_model = copy.deepcopy(global_model)
    temp_model.train()
    temp_model.to(device)



    for global_param in global_model.parameters():
        global_param.data = torch.zeros_like(global_param.data)
    
    for temp_param in temp_model.parameters():
        temp_param.data = torch.ones_like(temp_param.data)
    
    module_list = list(temp_model._modules.items())

    for zero_dict, weight in zip(zero_dicts, weights):
        for key in zero_dict.keys():
            if len(key) == 2:
                tensor_ref = module_list[key[0]][key[1]]
            else:
                tensor_ref = module_list[key[0]][key[1]][key[2]]
        
            mask_weight = torch.zeros_like(tensor_ref.weight)
            mask_bias = torch.zeros_like(tensor_ref.bias) if tensor_ref.bias is not None else None

            for (filter_index, type) in zero_dict[key]:
                if type==1:
                    mask_weight[filter_index] = weight
                    mask_bias[filter_index] = weight
                elif type==2:
                    mask_weight[filter_index] = weight
                elif type==3:
                    mask_weight[:, filter_index] = weight
            
            tensor_ref.weight.data = tensor_ref.weight.data - mask_weight

            if tensor_ref.bias is not None:
                tensor_ref.bias.data = tensor_ref.bias.data - mask_bias

    for client_model, weight in zip(client_models, weights):
        for global_param, client_param in zip(global_model.parameters(), client_model.parameters()):
            global_param.data += weight * client_param.data
    
    for temp_param, global_param in zip(temp_model.parameters(), global_model.parameters()):
        #do element wise division of each tensor with a temp tensor
        global_param.data = global_param.data / temp_param.data
    
    return global_model

  
#implement federated learning using helper federated setup. do fedaverage and fed prox first

def save_adam_state(optimizer):
    """Save full Adam optimizer state"""
    state_dict = optimizer.state_dict()
    adam_state = {}
    for k, v in state_dict['state'].items():
        param_state = {}
        if 'exp_avg' in v:
            param_state['exp_avg'] = v['exp_avg'].clone()
        if 'exp_avg_sq' in v:
            param_state['exp_avg_sq'] = v['exp_avg_sq'].clone()
        if 'step' in v:
            param_state['step'] = v['step']
        if param_state:
            adam_state[k] = param_state
    return adam_state

def load_adam_state(optimizer, saved_state):
    """Load Adam state into optimizer"""
    state_dict = optimizer.state_dict()
    for k, v in saved_state.items():
        if k in state_dict['state']:
            state_dict['state'][k].update(v)

def freeze_bn_layers(model):
    """Freeze batch norm layers"""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def train(method="FedAvg", model=None, num_clients=4, datadict_train=None, datadict_val=None, datadict_test=None, num_rounds=200, num_clients_per_round=4, global_model=None, criterion=None, optimizer=None, scheduler=None, device=0, size=(128, 128, 128), no_crop=False, no_aug=False ,num_epochs=200, hyperparams=None, lmd = 0, mu=0,  checkpoint=None, batch_size=8, not_all_clients=False, num_workers=4, use_simulated=False, custom=False, dropout_contrastive = False, temperature=0.07, additional_save_path=None, patience=20, additional_losses=[''], starting_lr = 1e-3, lr_decay_rate = 0.99, min_rounds=0, include_lg_pw=False, include_local_pw = False, include_every=True, lrs = None, option=1, mu_lg=1, weightage=1, base_prob=0.05, before_prune_epochs=5, after_prune_epochs=5, pruning_percentage=1, pruning_mode='Taylor', smart_aggregate=False, start_round_pruning=0, introduce_importance=False, importance_factor=[0.1], T=20, glm_dir='glm_models/'):
    #global_model = helper_model(model_type=model, which_data="FeTS", hyper_parameters=hyperparams, device=device, size=size)
    
    #pruning_percentage = 1
    pruning_mode = 'Taylor'
    
    if criterion != "None":
        criterion = helper_criterion(criterion_type=criterion)
    else:
        criterion = None
    device = 'cuda:'+str(device)
    collate_fn = None

    if option == 2:
        collate_fn = collate_resize
    elif option == 3:
        collate_fn = collate_pad
    elif option == 4:
        collate_fn = collate_padmax

    #get the models:
    init_features = 16
    global_unet = UNet4_new(init_features=init_features, drop_prob = base_prob)
    train_dataloaders = {}
    train_dataloaders_prune = {}
    
    for client in datadict_train.keys():
        train_dataloaders[client] = DataLoader(datadict_train[client], batch_size=min(batch_size, len(datadict_train[client])), shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
        train_dataloaders_prune[client] = DataLoader(datadict_train[client], batch_size=1, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    val_dataloaders = {}

    for client in datadict_val.keys():
        val_dataloaders[client] = DataLoader(datadict_val[client], batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn, pin_memory=True)

    test_dataloaders = {}

    for client in datadict_test.keys():
        test_dataloaders[client] = DataLoader(datadict_test[client], batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn, pin_memory=True)
    
    lmd_str = str(lmd).replace('.', '_')
    mu_str = str(mu).replace('.', '_')
    add_path = ''

    if method == 'FedProx':
        add_path = 'lambda_' + lmd_str + '/'
    
    if dropout_contrastive:
        add_path = 'with_dropout_contrastive/' + add_path

    else:
        add_path = 'without_dropout_contrastive/' + add_path

    if temperature != 0:
        add_path = add_path + 'temperature_' + str(temperature) + '/'
    
    if batch_size != 8:
        add_path = add_path + 'batch_size_' + str(batch_size) + '/'

    if additional_save_path != None:
        add_path = additional_save_path + '/' + add_path

    data_repr = ''

    if not_all_clients:
        list_all_clients = list(datadict_train.keys())
        list_all_clients.sort()
        data_repr = 'clients_' + ''.join([client[6:] for client in list_all_clients]) + '/'

    if use_simulated and custom:
        data_repr = 'custom_simulated_' + data_repr
    elif use_simulated:
        data_repr = 'simulated_' + data_repr
    
    model_save_path = MODEL_DIR + 'Federated/' + method + '/' + add_path + data_repr + model + '/'
    os.makedirs(model_save_path, exist_ok=True)

    print("Model save path: ", model_save_path)

    result_save_path = RESULTS_DIR + 'Federated/' + method + '/' + add_path + data_repr + model + '/'
    os.makedirs(result_save_path, exist_ok=True)

    checkpoint_path = model_save_path + '_state_dict' + str(checkpoint) + '.pth'

    helper_transformation(no_aug)

    train_losses_all = []
    val_losses_all = []
    train_dices_all = []
    val_dices_all = []
    list_zero_out_dicts = [dict() for _ in range(num_clients)]
    local_dices_before = [[float('-inf')] for _ in range(num_clients)]
    local_dices_after = [[float('-inf')] for _ in range(num_clients)]
    train_times = []
    test_dices = []
    params_remaining = []
    inf_times = []
    model_sizes = []
    
    best_dice = 0
    best_dice_before_prune = 0
    best_dice_before_prune_round = 0
    best_dice_before_agg_round = 0
    best_dice_before_agg = 0
    start_round = 0
    save_path_dict = dict()

    bn_stats_unet = [None for _ in range(num_clients)]

    if method == "FixBN":
        client_adam_states = [None for _ in range(num_clients)]

    if checkpoint != None:
        checkpoint_dict = torch.load(checkpoint_path)
        global_unet.load_state_dict(checkpoint_dict['unet_state_dict'])
        list_zero_out_dicts = checkpoint_dict['lists_zero_out_dicts']
        local_dices_before = checkpoint_dict['local_dices_before']
        local_dices_after = checkpoint_dict['local_dices_after']
        train_times = np.load(model_save_path + '_train_times' + '.npy').tolist()
        test_dices = np.load(model_save_path + '_test_dices' + '.npy').tolist()
        params_remaining = np.load(model_save_path + '_params_remaining' + '.npy').tolist()
        inf_times = np.load(model_save_path + '_inf_times' + '.npy').tolist()
        model_sizes = np.load(model_save_path + '_model_sizes' + '.npy').tolist()


        best_dice = checkpoint_dict['dice']
        start_round = checkpoint_dict['round'] + 1
        best_dice_before_agg = checkpoint_dict.get('best_dice_before_agg', 0)
        best_dice_before_prune = checkpoint_dict.get('best_dice_before_prune', 0)
        best_dice_before_prune_round = checkpoint_dict.get('best_dice_before_prune_round', 0)
        best_dice_before_agg_round = checkpoint_dict.get('best_dice_before_agg_round', 0)
        save_path_dict = checkpoint_dict.get('save_path_dict', dict())
        bn_stats_unet = checkpoint_dict.get('bn_stats_unet', None)
        #print the entries in the save_path_dict
        print(save_path_dict)

    rounds_since_improvement = 0
    #learning_rate = lrs_scheduled[start_round]
    if lrs[0] == -1:
        lrs = [starting_lr for _ in range(len(datadict_train.keys()))]
    

    #####
    num_clients_per_round = len(datadict_train.keys()) #select all the clients in every round
    #summary = get_summary(global_model, datadict_train=datadict_train, device=device, buckets=buckets)

    for round in range(start_round, num_rounds):
        
        #select the clients. Here, we have only 5 clients. So, we select all of them.
        clients = list(datadict_train.keys())

        #every 5th round, sync the vram
        if round % 5 == 0:
            torch.cuda.empty_cache()

        num_train_samples_clients = np.array([len(datadict_train[i]) for i in clients])

        train_dataloaders_cur_round = [train_dataloaders[i] for i in clients]
        train_dataloaders_prune_cur_round = [train_dataloaders_prune[i] for i in clients]
        client_unets = [copy.deepcopy(global_unet) for _ in range(num_clients_per_round)]
        client_optimizers = [optim.Adam(list(client_unets[i].parameters()), lr = lrs[i], eps = 0.0001) for i in range(num_clients_per_round)]
        
        
        #freeze the global unet and segmenter
        for param in global_unet.parameters():
            param.requires_grad = False
        
        
        global_unet.to(device)


        train_losses_clients = []
        train_dices_clients = []

        train_time = 0
        inf_time = 0
        model_size = 0
        params_remaining_cur_round = 0
        test_dices_cur_round = []
    

        for i in range(num_clients_per_round):
            train_dataloader = train_dataloaders_cur_round[i]
            train_dataloader_prune = train_dataloaders_prune_cur_round[i]
            client_unet = client_unets[i]
            client_zero_out_dict = list_zero_out_dicts[i]
            client_optimizer = client_optimizers[i]
            test_dice_client = 0
            
            if method == "FedBN" and bn_stats_unet[i] is not None:
                load_bn_stats(client_unet, bn_stats_unet[i])
            
            if method == "FixBN":
                # Restore optimizer state before training
                if client_adam_states[i] is not None:
                    load_adam_state(client_optimizer, client_adam_states[i])

                if round >= T:
                    freeze_bn_layers(client_unet)
                    
            #client_scheduler = client_schedulers[i]

            print("Round: ", round+1, "Client: ", i+1)

            #move the model to the device
            client_unet.to(device)

            #add all the hooks and zero out the layers
            t1 = time.time()
            client_unet, hooks = zero_out_and_add_hook_equivalent(client_unet, client_zero_out_dict)
            train_losses, train_dices = helper_train(client_unet, train_dataloader, client_optimizer, criterion, before_prune_epochs, device, lmd, mu_lg, mu, include_lg_pw, include_local_pw, global_unet, method=method)
            t2 = time.time()
            train_time += t2 - t1

            #remove the hooks
            for hook in hooks:
                hook.remove()

            train_losses_clients.append(train_losses[-1])
            train_dices_clients.append(train_dices[-1])
            print("Client: ", i+1, "Train Loss: ", train_losses[-1], "Train Dice: ", train_dices[-1])

            #validate the local model on its validation data
            val_loss, val_dice = helper_validate(client_unet, val_dataloaders[clients[i]], criterion, device)
            print("Client: ", i+1, "Val Loss: ", val_loss, "Val Dice: ", val_dice)

            #test the local model on its test data
            test_loss, test_dice = helper_validate(client_unet, test_dataloaders[clients[i]], criterion, device)

            if test_dice > test_dice_client:
                test_dice_client = test_dice

            #Save the model
            if val_dice > max(local_dices_before[i]):
                helper_save_model(round, client_unet, client_zero_out_dict, val_loss, val_dice, model_save_path, local_model=True, before_prune=True, local_epoch_num=before_prune_epochs, best_dice=True, local_dices_before=local_dices_before, local_dices_after=local_dices_after, best_dice_before_agg=best_dice_before_agg, best_dice_before_prune=best_dice_before_prune, best_dice_before_prune_round=best_dice_before_prune_round, best_dice_before_agg_round=best_dice_before_agg_round, client=clients[i], save_path_dict=save_path_dict)

            local_dices_before[i].append(val_dice)

            #Now prune
            print("Pruning client: ", i+1)

            prune_percent = pruning_percentage if round >= start_round_pruning else 0
            # If more than one factor is given, select by client index otherwise default to the first element
            pr_imp_factor = importance_factor[i] if len(importance_factor) > i else importance_factor[0]
            t1 = time.time()
            client_unet, client_zero_out_dict = pruning_function(client_unet, client_zero_out_dict, train_dataloader_prune, prune_percent, pruning_mode, device, introduce_importance=introduce_importance, importance_factor=pr_imp_factor, glm_dir=glm_dir, num_clients=num_clients_per_round)
            print("client {}: zero out dict: {}".format(i+1, client_zero_out_dict))
            t2 = time.time()
            train_time += t2 - t1


            list_zero_out_dicts[i] = client_zero_out_dict
            client_unets[i] = client_unet
            
            if after_prune_epochs == 0:
                continue
            
            if method == "FixBN" and round >= T:
                # Freeze batch norm layers after pruning
                freeze_bn_layers(client_unet)
            #train again for after_prune_epochs
            client_optimizer = optim.Adam(list(client_unets[i].parameters()), lr = lrs[i], eps = 0.0001)
            client_unet.to(device)

            print("Retraining:")

            t1 = time.time()
            client_unet, hooks = zero_out_and_add_hook_equivalent(client_unet, client_zero_out_dict)
            train_losses, train_dices = helper_train(client_unet, train_dataloader, client_optimizer, criterion, after_prune_epochs, device, lmd, mu_lg, mu, include_lg_pw, include_local_pw, global_unet, method=method)
            t2 = time.time()
            train_time += t2 - t1

            
            print("Client: ", i+1, "Train Loss: ", train_losses[-1], "Train Dice: ", train_dices[-1])

            #remove the hooks
            for hook in hooks:
                hook.remove()

            #validate the local model on its validation data
            val_loss, val_dice = helper_validate(client_unet, val_dataloaders[clients[i]], criterion, device)
            print("Client: ", i+1, "Val Loss: ", val_loss, "Val Dice: ", val_dice)

            

            #test the local model on its test data

            #make a copy of the models
            client_unet_copy = copy.deepcopy(client_unet)
            client_unet_copy.to(device)

            client_unet_copy = get_reduced_model(client_unet_copy, client_zero_out_dict, device=device)
            
            params_remaining_cur_round += count_params(client_unet_copy)
            model_size += get_model_size_mb(client_unet_copy)

            test_loss, test_dice, inference_time = helper_validate(client_unet_copy, test_dataloaders[clients[i]], criterion, device, return_inference_time=True)

            inf_time += inference_time


            if test_dice > test_dice_client:
                test_dice_client = test_dice

            test_dices_cur_round.append(test_dice)            

            #Save the model
            

            if val_dice > max(local_dices_after[i]):
                helper_save_model(round, client_unet, client_zero_out_dict, val_loss, val_dice, model_save_path, local_model=True, before_prune=False, local_epoch_num=after_prune_epochs, best_dice=True, local_dices_after=local_dices_after, local_dices_before=local_dices_before, best_dice_before_agg=best_dice_before_agg, best_dice_before_prune=best_dice_before_prune, best_dice_before_prune_round=best_dice_before_prune_round, best_dice_before_agg_round=best_dice_before_agg_round, client=clients[i], save_path_dict=save_path_dict)
            
            local_dices_after[i].append(val_dice)

            

            #save the bn stats
            if method == "FedBN":
                bn_stats_unet[i] = save_bn_stats(client_unet)
            
            if method == "FixBN":
                client_adam_states[i] = save_adam_state(client_optimizers[i])
            
            gc.collect()
            torch.cuda.empty_cache()
        
        #learning_rate =  0.001 # lrs[round]
        
        avg_train_time = train_time/len(clients) #training time for 1 round of communication, averaged across all the clients
        avg_params_remaining = params_remaining_cur_round/len(clients)
        avg_inf_time = inf_time/len(clients)
        avg_model_size = model_size/len(clients)
        train_times.append(avg_train_time)
        test_dices.append(np.mean(np.array(test_dices_cur_round)))
        params_remaining.append(avg_params_remaining)
        inf_times.append(avg_inf_time)
        model_sizes.append(avg_model_size)
        
        print("Average params remaining: ", avg_params_remaining)
        #perform aggregation
        weights = num_train_samples_clients/np.sum(num_train_samples_clients)
        weights = torch.tensor(weights).to(device)
        global_unet = aggregate(client_unets, weights, device) if not smart_aggregate else smart_aggregation(client_unets, list_zero_out_dicts, weights, device)

        #global_model = aggregate(client_models, weights, device)

        #perform weighted average of the train dice and train loss in train_losses_clients and train_dices_clients
        train_loss = np.sum(np.array(train_losses_clients) * weights.cpu().numpy())
        train_dice = np.sum(np.array(train_dices_clients) * weights.cpu().numpy())

        #run validation on the global model. All the clients participate in the validation process.
        global_unet.eval()

        #(maybe try weighted average for validation as well). Here, we do a simple average.
        val_loss = 0
        val_dice = 0

        for client in datadict_val.keys():
            val_dataloader = test_dataloaders[client]
            val_loss_client, val_dice_client = helper_validate(global_unet, val_dataloader, criterion, device, validate_global=True)
            val_loss += val_loss_client
            val_dice += val_dice_client
        
        val_loss /= len(datadict_val.keys())
        val_dice /= len(datadict_val.keys())

        train_losses_all.append(train_loss)
        val_losses_all.append(val_loss)
        train_dices_all.append(train_dice)
        val_dices_all.append(val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            helper_save_model(round, global_unet, list_zero_out_dicts, val_loss, val_dice, model_save_path, best_dice=True, local_dices_before=local_dices_before, local_dices_after=local_dices_after, best_dice_before_agg=best_dice_before_agg, best_dice_before_prune=best_dice_before_prune, best_dice_before_prune_round=best_dice_before_prune_round, best_dice_before_agg_round=best_dice_before_agg_round, save_path_dict=save_path_dict)
            rounds_since_improvement = 0
        else:
            rounds_since_improvement +=1
        
        before_prune_val_dice = np.mean(np.array([local_dices_before[i][-1] for i in range(num_clients_per_round)]))
        before_agg_val_dice = np.mean(np.array([local_dices_after[i][-1] for i in range(num_clients_per_round)]))

        if before_prune_val_dice > best_dice_before_prune:
            best_dice_before_prune = before_prune_val_dice
            best_dice_before_prune_round = round
        
        if before_agg_val_dice > best_dice_before_agg:
            best_dice_before_agg = before_agg_val_dice
            best_dice_before_agg_round = round
            helper_save_model(round, global_unet, list_zero_out_dicts, 1, best_dice_before_agg, model_save_path, local_dices_before=local_dices_before, local_dices_after=local_dices_after, best_dice_before_agg=best_dice_before_agg, best_dice_before_prune=best_dice_before_prune, best_dice_before_prune_round=best_dice_before_prune_round, best_dice_before_agg_round=best_dice_before_agg_round, best_before_agg=True, save_path_dict=save_path_dict)
        if round % 5 == 0:
            helper_save_model(round, global_unet, list_zero_out_dicts, val_loss, val_dice, model_save_path, local_dices_before=local_dices_before, local_dices_after=local_dices_after, best_dice_before_agg=best_dice_before_agg, best_dice_before_prune=best_dice_before_prune, best_dice_before_prune_round=best_dice_before_prune_round, best_dice_before_agg_round=best_dice_before_agg_round, save_path_dict=save_path_dict, bn_stats_unet=bn_stats_unet)
            #save the avg_train_time
            np.save(model_save_path + '_train_times' + '.npy', train_times)
            np.save(model_save_path + '_test_dices' + '.npy', test_dices)
            np.save(model_save_path + '_params_remaining' + '.npy', params_remaining)
            np.save(model_save_path + '_inf_times' + '.npy', inf_times)
            np.save(model_save_path + '_model_sizes' + '.npy', model_sizes)


        #print("Round: ", round+1, "Train Loss: ", train_loss, "Train Dice: ", train_dice, "Val Loss: ", val_loss, "Val Dice: ", val_dice)
        print("Round: ", round+1, "Train Loss: ", train_loss, "Train Dice: ", train_dice, "Val Loss: ", val_loss, "Val Dice: ", val_dice)

        #np.save(result_save_path + '_loss' + '.npy', [train_losses_all, val_losses_all, train_dices, val_dices_all])

        if rounds_since_improvement >= patience:
            print("Early stopping at round: ", round+1)
            break

    helper_save_model(round, global_unet, list_zero_out_dicts, val_loss, val_dice, model_save_path, local_dices_before=local_dices_before, local_dices_after=local_dices_after, best_dice_before_agg=best_dice_before_agg, best_dice_before_prune=best_dice_before_prune, best_dice_before_prune_round=best_dice_before_prune_round, best_dice_before_agg_round=best_dice_before_agg_round, save_path_dict=save_path_dict, bn_stats_unet=bn_stats_unet)
    np.save(model_save_path + '_train_times' + '.npy', train_times)
    np.save(model_save_path + '_test_dices' + '.npy', test_dices)
    np.save(model_save_path + '_params_remaining' + '.npy', params_remaining)
    np.save(model_save_path + '_inf_times' + '.npy', inf_times)
    np.save(model_save_path + '_model_sizes' + '.npy', model_sizes)

    print("Best val dice of global model: ", best_dice)
    
    #print the best local dices
    for i in range(num_clients):
        print("Client: ", i+1, "Best Dice before pruning: ", max(local_dices_before[i]), "Best Dice after pruning: ", max(local_dices_after[i]))
    
    #print the best_before_prune_dice
    print("Best Avg Dice across clients (before pruning) was ", best_dice_before_prune, " at round: ", best_dice_before_prune_round)

    print("Best Avg Dice across clients (before aggregation) was ", best_dice_before_agg, " at round: ", best_dice_before_agg_round)

    #plot train times, test dices, params remaining, inf times, model sizes
    plt.figure()
    plt.plot(range(round+1), train_times)
    plt.xlabel("Round")
    plt.ylabel("Train Time")
    plt.title("Train Time vs Rounds")
    plt.savefig(model_save_path + 'train_time.png')
    
    plt.figure()
    plt.plot(range(round+1), test_dices)
    plt.xlabel("Round")
    plt.ylabel("Test Dice")
    plt.title("Test Dice vs Rounds")
    plt.savefig(model_save_path + 'test_dice.png')

    plt.figure()
    plt.plot(range(round+1), params_remaining)
    plt.xlabel("Round")
    plt.ylabel("Params Remaining")
    plt.title("Params Remaining vs Rounds")
    plt.savefig(model_save_path + 'params_remaining.png')

    plt.figure()
    plt.plot(range(round+1), inf_times)
    plt.xlabel("Round")
    plt.ylabel("Inference Time")
    plt.title("Inference Time vs Rounds")
    plt.savefig(model_save_path + 'inf_time.png')

    plt.figure()
    plt.plot(range(round+1), model_sizes)
    plt.xlabel("Round")
    plt.ylabel("Model Size (MB)")
    plt.title("Model Size vs Rounds")
    plt.savefig(model_save_path + 'model_size.png')

    return global_model


if(__name__ =="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("-method",default='FedAvg',choices=['FedAvg','FedProx', 'FedBN', 'FixBN'],
                        help="federated framwork")
    parser.add_argument("-lambda", default=0, type=float, dest='lmd', help='Lambda for FedProx')
    parser.add_argument("-mu", default=1, type=float, help='Mu for FedContr')
    parser.add_argument("-mu_lg", default=1, type=float, help='Mu for Local Global Pixel-wise Contrastive Loss')
    parser.add_argument("-weightage", default=1, type=float, help='Weightage for the loss')
    parser.add_argument("-data", nargs='+', default=['client1', 'client2', 'client3', 'client4', 'client5'], choices=['client1', 'client2', 'client3', 'client4', 'client5', 'client6', 'client7', 'client8', 'client9'], help='Which clients to train on?')
    parser.add_argument("-dataset", default="fets", choices=['fets', 'wmh', 'combined', 'combined_40percent'], help='Which dataset to use?')
    parser.add_argument("-model",default='unet', choices=['unet','slimunetr','ducknet','saunet','nestedunet','halfunet','resunet','unetr','sacunet'],help='Which model to run ?')
    parser.add_argument("-loss",dest='criterion',default='focal + dice',choices=['dice','focal + dice', 'None', 'focal'],help='Which loss to choose?')
    parser.add_argument("-use_simulated_as_augmentation", default=False, action='store_true', help='Use simulated data as augmentation to the dataset?')
    parser.add_argument("-workers",default=4,type=int)
    parser.add_argument("-device",default=0,type=int,choices=[0,1])
    parser.add_argument("-batch",default=8,type=int)
    parser.add_argument("-date",default="{:%d_%m_%y}".format(datetime.now()))
    parser.add_argument("-sim_factor",type=float,dest='scale_factor',default=1.0,choices=[0.2,0.4,0.6,0.75,1.0,5.0])
    parser.add_argument("-real_factor",type=float,dest='factor',default=1.0,choices=[0.2,0.4,0.6,0.75,1.0,5.0])
    parser.add_argument("-checkpoint",type=int, default=None, help='Give checkpoint index to start model training from')
    parser.add_argument("-starting_lr", type=float, default=1e-3, help='Starting learning rate')
    parser.add_argument("-lrs", nargs='+', default=[-1], type=float, help='Learning rates for each client')
    parser.add_argument("-lr_decay_rate", type=float, default=0.99, help='Learning rate decay rate per round')
    parser.add_argument("-pretrained",type=int,dest='pretrained',help='Give self supervised pre trained models index to start fine tuning')
    parser.add_argument("-hyperparam",default="{'init_features':16}",dest='hyper_parameters',type=ast.literal_eval,help='Pass dictionary of hyperparameter if needs changing.')
    parser.add_argument("-no_aug",action='store_true')
    parser.add_argument("-exp_id",default='default',help='Name to uniquely identify the experiment')
    parser.add_argument("-simulation_path",dest='sim_path',help='Path to simulated files')
    parser.add_argument("-size",nargs='+',default=(128,128,128), type=int,help='To run it in orginal dimensions')
    parser.add_argument("-no_crop",default=False,action='store_true',help='To not have the model tight crop the images')
    parser.add_argument("-num_rounds",default=50,type=int)
    parser.add_argument("-num_epochs",default=5,type=int)
    parser.add_argument("-min_rounds",default=0,type=int, help="Minimum number of rounds before starting contrastive loss")
    parser.add_argument("-not_all_clients", dest='not_all_clients', default=False, action='store_true', help='Are these not all the clients?')
    parser.add_argument("-use_simulated", dest='use_simulated', default=False, action='store_true', help='Use simulated data?')
    parser.add_argument("-custom", default=False, action='store_true', help='Use custom simulated data?')
    parser.add_argument("-data_path",dest='system_data_path',default=63,type=int,choices=[63,64])
    parser.add_argument("-dropout_contrastive", default=False, action='store_true', help='Use dropout contrastive loss')
    parser.add_argument("-temperature", default=0.07, type=float, help='Temperature for contrastive loss')
    parser.add_argument("-patience", default=100, type=int, help='Patience (in rounds) for scheduler')
    parser.add_argument("-add_losses", nargs='+', default=[], choices=['local_con_aug', 'local_con_sim', 'global_con_real', 'global_con_sim', 'global_con_aug'])
    parser.add_argument("-local_global_pw", default=False, action='store_true', help='Use local global pixel-wise contrastive loss')
    parser.add_argument("-local_pw", default=False, action='store_true', help='Use local pixel-wise contrastive loss')
    parser.add_argument("-include_every", default=False, action='store_true', help='Include the local global pixel-wise contrastive loss in every epoch')
    parser.add_argument("-before_prune_epochs", default=5, type=int, help='Number of epochs before pruning')
    parser.add_argument("-after_prune_epochs", default=5, type=int, help='Number of epochs after pruning')
    parser.add_argument("-prune_percentage", default=1, type=float, help='Percentage of weights to prune')
    parser.add_argument("-pruning_mode", default='Taylor', choices=['Taylor', 'L1', 'L1_std'], help='Pruning mode')
    parser.add_argument("-system", default=63, type=int, choices=[63, 64, 131, 67, 66])
    parser.add_argument("-option", default = 1, type=int, choices=[1, 2, 3, 4])
    parser.add_argument("-smart_aggregate", default=False, action='store_true', help='Use smart aggregation')
    parser.add_argument("-start_round_pruning", default=0, type=int, help='Round to start pruning')
    parser.add_argument("-introduce_importance", default=False, action='store_true', help='Introduce importance in the pruning process')
    parser.add_argument("-importance_factor", nargs='+', default=[0.1], type=float, help='List of importance factors, one per client')
    parser.add_argument("-fixbn_rounds", type=int, default=20,
                   help="Number of rounds before freezing BN in FixBN")
    #add base_prob
    parser.add_argument("-base_prob", default=0.05, type=float, help='Feature drop probability')
    parser.add_argument("-combine_at_single_client", default=False, action='store_true', help='Combine the data at a single client')
    parser.add_argument("-system_data_path", default=False, type=str, help='path of the data')
    parser.add_argument("-model_dir", default='./models/', type=str, help='path to save the models')
    args = parser.parse_args()
    
    print("-----------------------------Arguments for the current execution-----------------------------------")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # system_data_path = 'put the path of the data here'
    system_data_path = args.system_data_path
    MODEL_DIR = args.model_dir
    glm_dir = MODEL_DIR + 'GlmModels/'
    if not os.path.exists(glm_dir):
        os.makedirs(glm_dir)
    #set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    datadict_train, datadict_val, datadict_test = helper_federated_setup(train_clients=args.data, val_clients=args.data, test_clients=args.data, size=args.size, no_crop=args.no_crop, use_simulated=args.use_simulated, custom=args.custom, system_data_path=system_data_path, use_simulated_as_augmentation=args.use_simulated_as_augmentation, combine_at_single_client=args.combine_at_single_client)
    
    #measure the time taken to train
    start = time.time()

    train(method=args.method, model=args.model, num_clients=len(args.data), datadict_train=datadict_train, datadict_val=datadict_val, datadict_test = datadict_test, num_rounds=args.num_rounds, num_clients_per_round=len(args.data), global_model=None, criterion=args.criterion, optimizer=None, scheduler=None, device=args.device, size=args.size, no_crop=args.no_crop, no_aug=args.no_aug, num_epochs=args.num_epochs, hyperparams=args.hyper_parameters, lmd=args.lmd, mu=args.mu, checkpoint=args.checkpoint, not_all_clients=args.not_all_clients, batch_size=args.batch, num_workers=args.workers, use_simulated=args.use_simulated, custom=args.custom, dropout_contrastive=args.dropout_contrastive, temperature=args.temperature, additional_save_path=args.exp_id, patience=args.patience, additional_losses=args.add_losses, starting_lr = args.starting_lr, lr_decay_rate = args.lr_decay_rate, min_rounds=args.min_rounds, include_lg_pw = args.local_global_pw, include_local_pw = args.local_pw, include_every=args.include_every, lrs = args.lrs, option=args.option, mu_lg=args.mu_lg, weightage=args.weightage, before_prune_epochs=args.before_prune_epochs, after_prune_epochs=args.after_prune_epochs, pruning_percentage=args.prune_percentage, pruning_mode=args.pruning_mode, smart_aggregate=args.smart_aggregate, start_round_pruning=args.start_round_pruning, introduce_importance=args.introduce_importance, importance_factor=args.importance_factor, T=args.fixbn_rounds, base_prob=args.base_prob, glm_dir=glm_dir)

    end = time.time()

    #print the time taken in hours
    print("Time taken in hours: ", (end-start)/3600)