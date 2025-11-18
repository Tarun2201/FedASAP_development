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
from pruning_functions import *
from models.unet_model_targeted_dropout1 import UNet1, Segmenter
from pruning_tools.prune_unet_zero_out import zero_out_and_add_hook, replace_layer_new
from pruning_functions_zero_out import pruning_function
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


CURRENT_DIRECTORY = os.getcwd()

MODEL_DIR = '/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6/Karan/projects/FedSamp/models/'
RESULTS_DIR = CURRENT_DIRECTORY + '/results/'

# composed_transform = transforms.Compose([
#             RandomRotation3D([-10,10],p=0.8),
#             RandomNoise3D(p=0.8),
#             RandomBrightness3D(p=0.8),
#             #ToTensor3D(True)
#             ])

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

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def get_reduced_model(model, zero_out_dict, device = 'cpu'):
    #move the old model to cpu first
    model.to('cpu')
    module_list = list(model._modules.items())
    #print(module_list)
    for key in zero_out_dict.keys():

        #get a reference to the layer
        if len(key) == 2:
            layer_ref = module_list[key[0]][key[1]]
        else:
            layer_ref = module_list[key[0]][key[1]][key[2]]
        
        list_of_tuples = zero_out_dict[key]

        #if the type of layer is Conv3D
        if isinstance(layer_ref, nn.Conv3d):
            type_1_indices = sorted([entry[0] for entry in list_of_tuples if entry[1] == 1])
            type_3_indices = sorted([entry[0] for entry in list_of_tuples if entry[1] == 3])

            number_type1 = len(type_1_indices)
            number_type3 = len(type_3_indices)

            new_number_out_channels = layer_ref.out_channels - number_type1
            new_number_in_channels = layer_ref.in_channels - number_type3
            
            #if there are any type 1 indices, we need to remove the out filters from the layer
            #lets first deal with the outchannels
            if len(type_1_indices) > 0:
                new_layer = nn.Conv3d(layer_ref.in_channels, new_number_out_channels, layer_ref.kernel_size, layer_ref.stride, layer_ref.padding, layer_ref.dilation, layer_ref.groups, layer_ref.bias is not None)

                index = 0
                start = 0
                
                for avoid in type_1_indices:
                    new_layer.weight.data[index:index+avoid-start] = layer_ref.weight.data[start:avoid]
                    if layer_ref.bias is not None:
                        new_layer.bias.data[index:index+avoid-start] = layer_ref.bias.data[start:avoid]
                    index += avoid-start
                    start = avoid+1

                new_layer.weight.data[index:] = layer_ref.weight.data[start:]
                if layer_ref.bias is not None:
                    new_layer.bias.data[index:] = layer_ref.bias.data[start:]

                layer_ref = new_layer

            #if there are any type 3 indices, we need to remove the in filters from the layer
            if len(type_3_indices) > 0:
                type_3_indices = set(type_3_indices)
                new_layer = nn.Conv3d(new_number_in_channels, new_number_out_channels, layer_ref.kernel_size, layer_ref.stride, layer_ref.padding, layer_ref.dilation, layer_ref.groups, layer_ref.bias is not None)
                new_layer.weight.data = layer_ref.weight.data[:, [i for i in range(layer_ref.in_channels) if i not in type_3_indices]]
                if layer_ref.bias is not None:
                    new_layer.bias.data = layer_ref.bias.data

                layer_ref = new_layer
            
        #if the type of layer is BatchNorm3D
        elif isinstance(layer_ref, nn.BatchNorm3d):
            type1_indices = [entry[0] for entry in list_of_tuples if entry[1] == 1]

            number_type1 = len(type1_indices)

            new_number_features = layer_ref.num_features - number_type1

            if len(type1_indices) > 0:
                new_layer = nn.BatchNorm3d(num_features = new_number_features)
                type1_indices = set(type1_indices)
                new_layer.weight.data = layer_ref.weight.data[[i for i in range(layer_ref.num_features) if i not in type1_indices]]

                if layer_ref.bias is not None:
                    new_layer.bias.data = layer_ref.bias.data[[i for i in range(layer_ref.num_features) if i not in type1_indices]]
                
                #set the running mean and variances
                new_layer.running_mean = layer_ref.running_mean[[i for i in range(layer_ref.num_features) if i not in type1_indices]]
                new_layer.running_var = layer_ref.running_var[[i for i in range(layer_ref.num_features) if i not in type1_indices]]
                
                layer_ref = new_layer
        
        elif isinstance(layer_ref, nn.ConvTranspose3d):
            type2_indices = [entry[0] for entry in list_of_tuples if entry[1] == 2]

            number_channels = layer_ref.in_channels - len(type2_indices)

            if len(type2_indices) > 0:
                new_layer = nn.ConvTranspose3d(in_channels=number_channels, out_channels=layer_ref.out_channels, kernel_size=layer_ref.kernel_size, stride=layer_ref.stride, padding=layer_ref.padding, output_padding=layer_ref.output_padding, groups=layer_ref.groups, bias=layer_ref.bias is not None, dilation=layer_ref.dilation)

                type2_indices = set(type2_indices)
                new_layer.weight.data = layer_ref.weight.data[[i for i in range(layer_ref.in_channels) if i not in type2_indices]]
                if layer_ref.bias is not None:
                    new_layer.bias.data = layer_ref.bias.data
                
                layer_ref = new_layer
        
        #replace the layer in the model
        model = replace_layer_new(model, key[0], key[2] if len(key) == 3 else None, layer_ref)
    
    #print(model._modules.items())
    #move the model back to the device 
    model.to(device)
    return model
    


#def helper_save_model(round,model,round_loss,round_dice,model_type,best_dice=False):
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


def helper_save_model(round,unet,segmenter, zero_out_dict_unet_lists, zero_out_dict_segmenter_lists, round_loss,round_dice,model_type,best_dice=False, local_model=False, before_prune=False, local_epoch_num=-1, local_dices_before=None, local_dices_after=None, best_before_prune=False, best_before_agg=False, best_dice_before_prune=0, best_dice_before_agg=0, best_dice_before_prune_round=0, best_dice_before_agg_round=0, client='client0', save_path_dict = dict(), bn_stats_unet=None, bn_stats_segmenter=None):

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
        'segmenter_state_dict': segmenter.state_dict(),
        'lists_zero_out_dicts_unets': zero_out_dict_unet_lists,
        'lists_zero_out_dicts_segmenters': zero_out_dict_segmenter_lists,
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
        'bn_stats_unet': bn_stats_unet,
        'bn_stats_segmenter': bn_stats_segmenter
    }, save_path_dict[loss_type])



def helper_train(unet, segmenter, train_loader, optimizer, criterion, num_epochs, device, lmd=0, mu_lg=1, mu=1, include_lg_pw=False, include_lpw=False, global_unet=None, global_segmenter=None, method="FedAvg", prev_models=[]):
    
    unet.train()
    segmenter.train()
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
            output, _ = segmenter(output)
            #print("Output shape: ", output.shape)
            loss = criterion(output, label, wt=label)

            if include_lg_pw:
                g_features = global_unet(image)
                g_output, g_bottleneck = global_segmenter(g_features)
                loss_lg_pw = lg_pw_loss(bottleneck, g_bottleneck, label)
                loss += mu_lg*loss_lg_pw
            
            if include_lpw:
                loss += mu*pixel_contrast_loss(bottleneck, label)
            
            if method == "FedProx":
                diff_norm_squared = 0
                for param1, param2 in zip(global_unet.parameters(), unet.parameters()):
                    diff_norm_squared += torch.norm(param1.data - param2.data) ** 2
                loss += lmd / 2 * diff_norm_squared
            
            if method == 'MOON':
                batch_size = image.size(0)
                criterion1 = nn.CrossEntropyLoss().to(device)
                cos = torch.nn.CosineSimilarity(dim=-1)
                _, bottleneck_gl = global_unet(image)
                posi = cos(bottleneck, bottleneck_gl)
                logits = posi.reshape(batch_size, -1, 1)

                for prev_unet, prev_segmenter in prev_models:
                    prev_unet.to(device)
                    _, bottleneck_prev = prev_unet(image)
                    nega = cos(bottleneck, bottleneck_prev)
                    logits = torch.cat((logits, nega.reshape(batch_size, -1, 1)), dim=1)
                
                temperature = 0.5

                logits /= temperature

                
                labels = torch.zeros(batch_size, dtype=torch.long).to(device)
                labels = labels.reshape(-1, 1)
                #print(logits.shape, labels.shape)
                loss += mu* criterion1(logits, labels)

            #print(loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_dice += Dice_Score(output.cpu().detach().numpy(), label.cpu().detach().numpy())
            

        losses.append(total_loss/len(train_loader))
        dices.append(total_dice/len(train_loader))
    
    return losses, dices

def helper_validate(unet, segmenter, val_loader, criterion, device, return_inference_time=False, validate_global=False):
    unet.eval()
    segmenter.eval()
    total_loss = 0
    total_dice = 0
    f1_score_total = 0
    inf_times = []
    with torch.no_grad():
        for _, data in enumerate(val_loader):
            image = data['input'].to(device)
            label = data['gt'].to(device)
            start = time.time()
            output, _ = unet(image)
            output, bottleneck = segmenter(output)
            end = time.time()
            inf_times.append(end-start)
            loss = criterion(output, label, wt=label)
            total_loss += loss.item()
            total_dice += Dice_Score(output.cpu().detach().numpy(), label.cpu().detach().numpy())
            if validate_global:
                #also calc the lesion level scores
                _, precision, f1 = Lesion_Metrics(output.cpu().detach().numpy(), label.cpu().detach().numpy())
                f1_score_total += f1

    if validate_global:
        print("validation for global model")
        print("F1 Score: ", f1_score_total/len(val_loader))
        print("Dice Score: ", total_dice/len(val_loader))
    return (total_loss/len(val_loader), total_dice/len(val_loader)) if not return_inference_time else (total_loss/len(val_loader), total_dice/len(val_loader), np.mean(np.array(inf_times)))



def helper_train1(model, global_model, criterion, optimizer, trainloader, device, method = "FedAvg", lmd=0, mu_lg=1, mu=1, weightage=1, dropout_contrastive = False, temperature = 0.07, additional_losses=[], round=1, min_rounds=20, include_lg_pw=False, include_lpw=False):
    model.train()
    train_loss = 0
    train_dice = 0
    
    train_size = len(trainloader)

    num_losses = len(additional_losses) + 1

    pixel_contrast_loss = PixelwiseContrastiveLoss()

    lg_pw_loss = LocalGlobalPixelwiseContrastiveLoss()


    with tqdm(range(train_size)) as pbar:
        for i, data in zip(pbar, trainloader):
            torch.cuda.empty_cache()
            loss = 0
            image = data['input'].to(device)
            output_lr, bottleneck_lr = model.forward(image)
            label = data['gt'].to(device)
            loss_lr = 0
            if criterion is not None:
                loss_lr = weightage*criterion(output_lr, label, wt=label)

            loss_la = 0
            loss_ls = 0

            if include_lg_pw:
                _, bottleneck_gr = global_model.forward(image)
                loss_lg_pw = lg_pw_loss(bottleneck_lr, bottleneck_gr, label)
                loss += mu_lg*loss_lg_pw


            if method == "FedProx":
                
                diff_norm_squared = 0
                for param1, param2 in zip(global_model.parameters(), model.parameters()):
                    diff_norm_squared += torch.norm(param1.data - param2.data) ** 2

                # Add the regularization term to the loss
                loss += lmd / 2 * diff_norm_squared
            

            if include_lpw:
                """
                _, bottleneck_glob = global_model.forward(image)
                bottleneck_glob = bottleneck_glob.detach()
                loss_fn = losses.NTXentLoss(temperature=temperature)
                loss_fn = losses.SelfSupervisedLoss(loss_fn, symmetric=True)
                loss += mu*loss_fn(bottleneck, bottleneck_glob)"""

                if round > min_rounds:
                    #create a copy of bottleneck_lr
                    #_, bottleneck_lr1 = global_model.forward(image)
                    #bottleneck_lr_global = bottleneck_lr1_global.detach()

                    loss += mu*pixel_contrast_loss(bottleneck_lr, label)
            """
            if dropout_contrastive:
                _, bottleneck2 = model.forward(image)
                bottleneck2 = bottleneck2.detach()
                loss_fn = losses.NTXentLoss(temperature=temperature)
                loss_fn = losses.SelfSupervisedLoss(loss_fn, symmetric=True)
                loss += loss_fn(bottleneck, bottleneck2)
            """
            image1 = image.cpu().detach().numpy()
            label1 = label.cpu().detach().numpy() 

            if 'local_con_aug' in additional_losses or 'global_con_aug' in additional_losses:
                batch = [{'input': image, 'gt': label} for image, label in zip(image1, label1)]

                # Perform composed transform on the batch
                transformed_batch = apply_transformations_to_batch(batch, composed_transform)

                # Extract augmented images and labels
                image_aug = transformed_batch['input']
                label_aug = transformed_batch['gt']

                # Move the augmented images and labels to the device
                image_aug = image_aug.to(device)
                label_aug = label_aug.to(device)

                
                # Forward pass with augmented data
                output_la, bottleneck_la = model.forward(image_aug)

                loss_la = criterion(output_la, label_aug, wt=label_aug)

                num_losses += 1 #for local (dice + focal) loss for augmented image
            
            if 'global_con_sim' in additional_losses or 'local_con_sim' in additional_losses:
                sim_samples = data['sim_samples']
                sim_samples = np.array(sim_samples)
                #print the shape of the image_sim
                #print(image_sim.shape) (3, 2 (one for image and one for gt), batch, 128, 128, 128, 1)

                #transpose the image_sim to be batch, 3, 2, 128, 128, 128, 1. Then select one pair of image and gt the three choices available for each image in the batch
                #sim_samples = np.transpose(sim_samples, (2, 0, 1, 3, 4, 5, 6))
                #print the shape of the image_sim
                #print(image_sim.shape) (batch, 3, 2, 128, 128, 128, 1)

                #randomly select one pair of image and gt the three choices available for each image in the batch
                image_sim = np.zeros((image1.shape[0], 1, 128, 128, 128))
                gt_sim = np.zeros((image1.shape[0], 1, 128, 128, 128))

                for i in range(image1.shape[0]):
                    choice = np.random.choice(3)
                    image_sim[i] = sim_samples[choice, 0, i, :, :, :, 0]
                    gt_sim[i] = sim_samples[choice, 1, i, :, :, :, 0]
                
                image_sim = torch.Tensor(image_sim).to(device)
                gt_sim = torch.Tensor(gt_sim).to(device)

                output_ls, bottleneck_ls = model.forward(image_sim)
                loss_ls = criterion(output_ls, gt_sim, wt=gt_sim)

                num_losses += 1 #for global (dice + focal) loss for simulated image
            
            w_loss = 1/num_losses #weight for each loss

            #lets add the local losses
            loss += w_loss * loss_lr
            loss += w_loss * loss_la
            loss += w_loss * loss_ls

            #print("Loss after all: ", loss.item())

            if 'local_con_aug' in additional_losses:
                #loss_lcona = losses.SelfSupervisedLoss(losses.ContrastiveLoss(pos_margin=1, neg_margin=0, distance=distances.CosineSimilarity()), symmetric=True)
                #use NTXentLoss for contrastive loss and then use the wrapper for self supervised loss
                loss_lcona = losses.SelfSupervisedLoss(losses.NTXentLoss(temperature=temperature), symmetric=True)
                loss_loc_con_aug= loss_lcona(bottleneck_la, bottleneck_lr)
                loss += w_loss * loss_loc_con_aug
            
            if 'local_con_sim' in additional_losses:
                #loss_lcons = losses.SelfSupervisedLoss(losses.ContrastiveLoss(pos_margin=1, neg_margin=0, distance=distances.CosineSimilarity()), symmetric=True)
                loss_lcons = losses.SelfSupervisedLoss(losses.NTXentLoss(temperature=temperature), symmetric=True)
                loss_loc_con_sim = loss_lcons(bottleneck_ls, bottleneck_lr)
                loss += w_loss * loss_loc_con_sim
            
            if 'global_con_real' in additional_losses:
                #create a local copy of global model
                output_gr, bottleneck_gr = global_model.forward(image)
                loss_gconr = losses.SelfSupervisedLoss(losses.NTXentLoss(temperature=temperature), symmetric=True)
                loss_glob_con_real = loss_gconr(bottleneck_gr, bottleneck_lr)
                loss += w_loss * loss_glob_con_real
            
            if 'global_con_sim' in additional_losses:
                output_gs, bottleneck_gs = global_model.forward(image_sim)
                loss_gcons = losses.SelfSupervisedLoss(losses.NTXentLoss(temperature=temperature), symmetric=True)
                loss_glob_con_sim = loss_gcons(bottleneck_gs, bottleneck_ls)
                loss += w_loss * loss_glob_con_sim
            
            if 'global_con_aug' in additional_losses:
                output_ga, bottleneck_ga = global_model.forward(image_aug)
                loss_gcona = losses.SelfSupervisedLoss(losses.NTXentLoss(temperature=temperature), symmetric=True)
                loss_glob_con_aug = loss_gcona(bottleneck_ga, bottleneck_la)
                loss += w_loss * loss_glob_con_aug


            
            #print("Loss (focal+dice) simulated image: ", loss_sim.item())

            #contrastive loss between the simulated image and the original image
            #loss_contrastive_sim = losses.NTXentLoss(temperature=temperature)
            #loss_contrastive_sim = losses.SelfSupervisedLoss(loss_contrastive_sim, symmetric=True)
            #loss_contrastive_sim = losses.SelfSupervisedLoss(losses.ContrastiveLoss(pos_margin=1, neg_margin=0, distance=distances.CosineSimilarity()), symmetric=True)
            #loss_con_sim = loss_contrastive_sim(bottleneck, bottleneck_sim)
            #print("Contrastive loss simulated image: ", loss_con_sim.item())
            #loss += loss_con_sim

        

            count = 1

            dice = Dice_Score(output_lr.cpu().detach().numpy(),label.cpu().detach().numpy())

            if loss_ls != 0:
                dice += Dice_Score(output_ls.cpu().detach().numpy(),gt_sim.cpu().detach().numpy())
                count += 1

            if loss_la != 0:
                dice += Dice_Score(output_la.cpu().detach().numpy(),label_aug.cpu().detach().numpy())
                count += 1
                
            dice /= count

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice.item()
    
    return train_loss/train_size, train_dice/train_size




# def helper_validate(model, dataloader, criterion, device):
#     model.eval()
#     val_loss = 0
#     val_dice = 0
#     val_size = len(dataloader)

#     with tqdm(range(val_size)) as pbar:
#         for i, data in zip(pbar, dataloader):
#             torch.cuda.empty_cache()
#             loss = 0

#             with torch.no_grad():
#                 image = data['input'].to(device)
#                 output, _ = model.forward(image)
#                 label = data['gt'].to(device)

#                 loss = criterion(output, label, wt=label) if criterion is not None else 0

#                 dice = Dice_Score(output.cpu().detach().numpy(),label.cpu().detach().numpy())

#                 if type(loss) == int:
#                     val_loss += loss
#                 else:
#                     val_loss += loss.item()
        
#                 #val_loss += loss.item()
#                 val_dice += dice.item()
    
#     return val_loss/val_size, val_dice/val_size

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


def train(method="FedAvg", model=None, num_clients=4, datadict_train=None, datadict_val=None, datadict_test=None, num_rounds=200, num_clients_per_round=4, global_model=None, criterion=None, optimizer=None, scheduler=None, device=0, size=(128, 128, 128), no_crop=False, no_aug=False ,num_epochs=200, hyperparams=None, lmd = 0, mu=0,  checkpoint=None, batch_size=8, not_all_clients=False, num_workers=4, use_simulated=False, custom=False, dropout_contrastive = False, temperature=0.07, additional_save_path=None, patience=20, additional_losses=[''], starting_lr = 1e-3, lr_decay_rate = 0.99, min_rounds=0, include_lg_pw=False, include_local_pw = False, include_every=True, lrs = None, option=1, mu_lg=1, weightage=1, base_prob=0.05, before_prune_epochs=5, after_prune_epochs=5, pruning_percentage=1, pruning_mode='Taylor', smart_aggregate=False, start_round_pruning=0, introduce_importance=False, importance_factor=[0.1], T=20):
    #global_model = helper_model(model_type=model, which_data="FeTS", hyper_parameters=hyperparams, device=device, size=size)
    
    probs = {}
    probs = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.1, 10: 0.1, 11: 0.1, 12: 0.1, 13: 0.1, 14: 0.1, 15: 0.1, 16: 0.1}

    for key in probs:
        probs[key] = probs[key] * base_prob

    #pruning_percentage = 1
    pruning_mode = 'Taylor'
    
    #getting a list containing starting lr for each round.
    # optimizer = optim.Adam(global_model.parameters(), lr=0.002, eps=0.0001)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=20, eta_min=0.0002)

    # lrs_scheduled = []
    # for _ in range(num_rounds):
    #     scheduler.step()
    #     lrs_scheduled.append(optimizer.param_groups[0]['lr'])

    # plt.plot(range(num_rounds), lrs)
    # plt.xlabel("Round")
    # plt.ylabel("Learning Rate")
    # plt.title("Learning Rate vs Rounds")
    # plt.savefig('demo.png')

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
    global_unet = UNet1(drop_probs=probs, init_features=init_features)
    global_segmenter = Segmenter(init_features=init_features)
    #get the dataloaders for the clients
    train_dataloaders = {}
    
    for client in datadict_train.keys():
        train_dataloaders[client] = DataLoader(datadict_train[client], batch_size=min(batch_size, len(datadict_train[client])), shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    val_dataloaders = {}

    for client in datadict_val.keys():
        val_dataloaders[client] = DataLoader(datadict_val[client], batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)

    test_dataloaders = {}

    for client in datadict_test.keys():
        test_dataloaders[client] = DataLoader(datadict_test[client], batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)
    
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
    list_zero_out_dicts_unets = [dict() for _ in range(num_clients)]
    list_zero_out_dicts_segmenters = [dict() for _ in range(num_clients)]
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
    bn_stats_segmenter = [None for _ in range(num_clients)]

    if method == "FixBN":
        client_adam_states = [None for _ in range(num_clients)]

    # buckets = [(0,2), (2, 4), (4, 10), (10, 30), (30, 60)]

    # clients_data = [np.zeros((2, len(buckets), num_rounds//5)) for _ in range(num_clients)] #2 <- one for FPs and one for FNs

    if checkpoint != None:
        checkpoint_dict = torch.load(checkpoint_path)
        global_unet.load_state_dict(checkpoint_dict['unet_state_dict'])
        list_zero_out_dicts_unets = checkpoint_dict['lists_zero_out_dicts_unets']
        global_segmenter.load_state_dict(checkpoint_dict['segmenter_state_dict'])
        list_zero_out_dicts_segmenters = checkpoint_dict['lists_zero_out_dicts_segmenters']
        local_dices_before = checkpoint_dict['local_dices_before']
        local_dices_after = checkpoint_dict['local_dices_after']
        train_times = np.load(model_save_path + '_train_times' + '.npy').tolist()
        test_dices = np.load(model_save_path + '_test_dices' + '.npy').tolist()
        params_remaining = np.load(model_save_path + '_params_remaining' + '.npy').tolist()
        inf_times = np.load(model_save_path + '_inf_times' + '.npy').tolist()
        model_sizes = np.load(model_save_path + '_model_sizes' + '.npy').tolist()
        #losses = np.load(result_save_path + '_loss' + '.npy')

        # clients_data = np.load('./clients_data.npy')

        # train_losses_all = losses[0, :].tolist()[:checkpoint_dict['round']+1]
        # val_losses_all = losses[1, :].tolist()[:checkpoint_dict['round']+1]
        # train_dices_all = losses[2, :].tolist()[:checkpoint_dict['round']+1]
        # val_dices_all = losses[3, :].tolist()[:checkpoint_dict['round']+1]

        best_dice = checkpoint_dict['dice']
        start_round = checkpoint_dict['round'] + 1
        best_dice_before_agg = checkpoint_dict.get('best_dice_before_agg', 0)
        best_dice_before_prune = checkpoint_dict.get('best_dice_before_prune', 0)
        best_dice_before_prune_round = checkpoint_dict.get('best_dice_before_prune_round', 0)
        best_dice_before_agg_round = checkpoint_dict.get('best_dice_before_agg_round', 0)
        save_path_dict = checkpoint_dict.get('save_path_dict', dict())
        bn_stats_unet = checkpoint_dict.get('bn_stats_unet', None)
        bn_stats_segmenter = checkpoint_dict.get('bn_stats_segmenter', None)
        #print the entries in the save_path_dict
        print(save_path_dict)

    rounds_since_improvement = 0
    #learning_rate = lrs_scheduled[start_round]
    if lrs[0] == -1:
        lrs = [starting_lr for _ in range(len(datadict_train.keys()))]
    
    prev_models_lists = [[] for _ in range(len(datadict_train.keys()))]

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
        client_unets = [copy.deepcopy(global_unet) for _ in range(num_clients_per_round)]
        client_segmenters = [copy.deepcopy(global_segmenter) for _ in range(num_clients_per_round)]
        client_optimizers = [optim.Adam(list(client_unets[i].parameters())+list(client_segmenters[i].parameters()), lr = lrs[i], eps = 0.0001) for i in range(num_clients_per_round)]
        
        #client_schedulers = [optim.lr_scheduler.ReduceLROnPlateau(client_optimizers[i], factor=0.5, patience=10, min_lr = 1e-4, mode='max') for i in range(num_clients_per_round)]
        
        #freeze the global unet and segmenter
        for param in global_unet.parameters():
            param.requires_grad = False
        
        for param in global_segmenter.parameters():
            param.requires_grad = False
        
        global_unet.to(device)
        global_segmenter.to(device)


        train_losses_clients = []
        train_dices_clients = []

        train_time = 0
        inf_time = 0
        model_size = 0
        params_remaining_cur_round = 0
        test_dices_cur_round = []

        for i in range(num_clients_per_round):
            train_dataloader = train_dataloaders_cur_round[i]
            client_unet = client_unets[i]
            client_segmenter = client_segmenters[i]
            client_zero_out_dict_unet = list_zero_out_dicts_unets[i]
            client_zero_out_dict_segmenter = list_zero_out_dicts_segmenters[i]
            client_optimizer = client_optimizers[i]
            test_dice_client = 0
            
            if method == "FedBN" and bn_stats_unet[i] is not None:
                load_bn_stats(client_unets[i], bn_stats_unet[i])
                load_bn_stats(client_segmenters[i], bn_stats_segmenter[i])
            
            if method == "FixBN":
                # Restore optimizer state before training
                if client_adam_states[i] is not None:
                    load_adam_state(client_optimizers[i], client_adam_states[i])

                if round >= T:
                    freeze_bn_layers(client_unet)
                    freeze_bn_layers(client_segmenter)
            
            
            
                    
            #client_scheduler = client_schedulers[i]

            print("Round: ", round+1, "Client: ", i+1)

            #move the model to the device
            client_unet.to(device)
            client_segmenter.to(device)

            #add all the hooks and zero out the layers
            t1 = time.time()
            client_unet, client_segmenter, unet_hooks, seg_hooks = zero_out_and_add_hook(client_unet, client_segmenter, client_zero_out_dict_unet, client_zero_out_dict_segmenter)
            
            train_losses, train_dices = helper_train(client_unet, client_segmenter, train_dataloader, client_optimizer, criterion, before_prune_epochs, device, lmd, mu_lg, mu, include_lg_pw, include_local_pw, global_unet, global_segmenter, method=method, prev_models=prev_models_lists[i])
            t2 = time.time()
            train_time += t2 - t1

            



            train_losses_clients.append(train_losses[-1])
            train_dices_clients.append(train_dices[-1])
            print("Client: ", i+1, "Train Loss: ", train_losses[-1], "Train Dice: ", train_dices[-1])

            #validate the local model on its validation data
            val_loss, val_dice = helper_validate(client_unet, client_segmenter, val_dataloaders[clients[i]], criterion, device)
            print("Client: ", i+1, "Val Loss: ", val_loss, "Val Dice: ", val_dice)

            #test the local model on its test data
            test_loss, test_dice = helper_validate(client_unet, client_segmenter, test_dataloaders[clients[i]], criterion, device)

            if test_dice > test_dice_client:
                test_dice_client = test_dice

            #Save the model
            if val_dice > max(local_dices_before[i]):
                helper_save_model(round, client_unet, client_segmenter, client_zero_out_dict_unet, client_zero_out_dict_segmenter, val_loss, val_dice, model_save_path, local_model=True, before_prune=True, local_epoch_num=before_prune_epochs, best_dice=True, local_dices_before=local_dices_before, local_dices_after=local_dices_after, best_dice_before_agg=best_dice_before_agg, best_dice_before_prune=best_dice_before_prune, best_dice_before_prune_round=best_dice_before_prune_round, best_dice_before_agg_round=best_dice_before_agg_round, client=clients[i], save_path_dict=save_path_dict)

            local_dices_before[i].append(val_dice)

            #Now prune
            print("Pruning client: ", i+1)

            prune_percent = pruning_percentage if round >= start_round_pruning else 0
            # If more than one factor is given, select by client index otherwise default to the first element
            pr_imp_factor = importance_factor[i] if len(importance_factor) > i else importance_factor[0]
            t1 = time.time()
            client_unet, client_segmenter, zero_out_dict_unet, zero_out_dict_segmenter = pruning_function(client_unet, client_segmenter, client_zero_out_dict_unet, client_zero_out_dict_segmenter, train_dataloader, prune_percent, pruning_mode, device, introduce_importance=introduce_importance, importance_factor=pr_imp_factor)
            t2 = time.time()
            train_time += t2 - t1
            # print("Pruning done for client: ", i+1)
            # print("Zero out dict unet: ", zero_out_dict_unet)

            list_zero_out_dicts_unets[i] = zero_out_dict_unet
            list_zero_out_dicts_segmenters[i] = zero_out_dict_segmenter
            client_segmenters[i] = client_segmenter
            client_unets[i] = client_unet
            #remove the hooks
            for hook in unet_hooks:
                hook.remove()
            for hook in seg_hooks:
                hook.remove()
            
            if after_prune_epochs == 0:
                continue
            
            if method == "FixBN" and round >= T:
                # Freeze batch norm layers after pruning
                freeze_bn_layers(client_unet)
                freeze_bn_layers(client_segmenter)
            #train again for after_prune_epochs
            client_optimizer = optim.Adam(list(client_unets[i].parameters())+list(client_segmenters[i].parameters()), lr = lrs[i], eps = 0.0001)
            client_unet.to(device)
            client_segmenter.to(device)

            print("Retraining:")

            t1 = time.time()
            client_unet, client_segmenter, unet_hooks, seg_hooks = zero_out_and_add_hook(client_unet, client_segmenter, zero_out_dict_unet, zero_out_dict_segmenter)
            train_losses, train_dices = helper_train(client_unet, client_segmenter, train_dataloader, client_optimizer, criterion, after_prune_epochs, device, lmd, mu_lg, mu, include_lg_pw, include_local_pw, global_unet, global_segmenter, method=method)
            t2 = time.time()
            train_time += t2 - t1
            # train_losses_clients.append(train_losses[-1])
            # train_dices_clients.append(train_dices[-1])
            
            print("Client: ", i+1, "Train Loss: ", train_losses[-1], "Train Dice: ", train_dices[-1])

            #validate the local model on its validation data
            val_loss, val_dice = helper_validate(client_unet, client_segmenter, val_dataloaders[clients[i]], criterion, device)
            print("Client: ", i+1, "Val Loss: ", val_loss, "Val Dice: ", val_dice)

            #remove the hooks
            for hook in unet_hooks:
                hook.remove()
            for hook in seg_hooks:
                hook.remove()

            #test the local model on its test data

            #make a copy of the models
            client_unet_copy = copy.deepcopy(client_unet)
            client_segmenter_copy = copy.deepcopy(client_segmenter)
            client_unet_copy.to(device)
            client_segmenter_copy.to(device)

            client_unet_copy = get_reduced_model(client_unet_copy, zero_out_dict_unet, device=device)
            client_segmenter_copy = get_reduced_model(client_segmenter_copy, zero_out_dict_segmenter, device=device)
            
            params_remaining_cur_round += count_params(client_unet_copy) + count_params(client_segmenter_copy)
            model_size += get_model_size_mb(client_unet_copy) + get_model_size_mb(client_segmenter_copy)
            

            test_loss, test_dice, inference_time = helper_validate(client_unet_copy, client_segmenter_copy, test_dataloaders[clients[i]], criterion, device, return_inference_time=True)

            inf_time += inference_time


            if test_dice > test_dice_client:
                test_dice_client = test_dice

            test_dices_cur_round.append(test_dice)            

            #Save the model
            

            if val_dice > max(local_dices_after[i]):
                helper_save_model(round, client_unet, client_segmenter, zero_out_dict_unet, zero_out_dict_segmenter, val_loss, val_dice, model_save_path, local_model=True, before_prune=False, local_epoch_num=after_prune_epochs, best_dice=True, local_dices_after=local_dices_after, local_dices_before=local_dices_before, best_dice_before_agg=best_dice_before_agg, best_dice_before_prune=best_dice_before_prune, best_dice_before_prune_round=best_dice_before_prune_round, best_dice_before_agg_round=best_dice_before_agg_round, client=clients[i], save_path_dict=save_path_dict)
            
            local_dices_after[i].append(val_dice)

            

            #save the bn stats
            if method == "FedBN":
                bn_stats_unet[i] = save_bn_stats(client_unet)
                bn_stats_segmenter[i] = save_bn_stats(client_segmenter)
            
            if method == "FixBN":
                client_adam_states[i] = save_adam_state(client_optimizers[i])
        
        if method == 'MOON':
            for j, l in enumerate(prev_models_lists):
                if len(l) != 0:
                    l.pop()
                m = copy.deepcopy(client_unets[j])
                n = copy.deepcopy(client_segmenters[j])
                m.to('cpu')
                n.to('cpu')
                l.append((m, n))
            
                prev_models_lists[j] = l
        
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
        

        #perform aggregation
        weights = num_train_samples_clients/np.sum(num_train_samples_clients)
        weights = torch.tensor(weights).to(device)
        global_unet = aggregate(client_unets, weights, device) if not smart_aggregate else smart_aggregation(client_unets, list_zero_out_dicts_unets, weights, device)
        global_segmenter = aggregate(client_segmenters, weights, device) if not smart_aggregate else smart_aggregation(client_segmenters, list_zero_out_dicts_segmenters, weights, device)


        #global_model = aggregate(client_models, weights, device)

        #perform weighted average of the train dice and train loss in train_losses_clients and train_dices_clients
        train_loss = np.sum(np.array(train_losses_clients) * weights.cpu().numpy())
        train_dice = np.sum(np.array(train_dices_clients) * weights.cpu().numpy())

        #run validation on the global model. All the clients participate in the validation process.
        global_unet.eval()
        global_segmenter.eval()

        #(maybe try weighted average for validation as well). Here, we do a simple average.
        val_loss = 0
        val_dice = 0

        for client in datadict_val.keys():
            val_dataloader = test_dataloaders[client]
            val_loss_client, val_dice_client = helper_validate(global_unet, global_segmenter, val_dataloader, criterion, device, validate_global=True)
            val_loss += val_loss_client
            val_dice += val_dice_client
        
        val_loss /= len(datadict_val.keys())
        val_dice /= len(datadict_val.keys())

        train_losses_all.append(train_loss)
        val_losses_all.append(val_loss)
        train_dices_all.append(train_dice)
        val_dices_all.append(val_dice)

        # val_ts_dice = 0
        # val_ts_loss = 0

        # #perform validation on the training data as well. use different variable names to avoid confusion.
        # for client in datadict_train.keys():
        #     train_dataloader = train_dataloaders[client]
        #     train_loss_client, train_dice_client = helper_validate(global_model, train_dataloader, criterion, device)
        #     val_ts_dice += train_dice_client
        #     val_ts_loss += train_loss_client

        # val_ts_dice /= len(datadict_train.keys())
        # val_ts_loss /= len(datadict_train.keys())


        if val_dice > best_dice:
            best_dice = val_dice
            helper_save_model(round, global_unet, global_segmenter, list_zero_out_dicts_unets, list_zero_out_dicts_segmenters, val_loss, val_dice, model_save_path, best_dice=True, local_dices_before=local_dices_before, local_dices_after=local_dices_after, best_dice_before_agg=best_dice_before_agg, best_dice_before_prune=best_dice_before_prune, best_dice_before_prune_round=best_dice_before_prune_round, best_dice_before_agg_round=best_dice_before_agg_round, save_path_dict=save_path_dict)
            #save the clients_data
            # np.save('./clients_data.npy', clients_data)
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
            helper_save_model(round, global_unet, global_segmenter, list_zero_out_dicts_unets, list_zero_out_dicts_segmenters, 1, best_dice_before_agg, model_save_path, local_dices_before=local_dices_before, local_dices_after=local_dices_after, best_dice_before_agg=best_dice_before_agg, best_dice_before_prune=best_dice_before_prune, best_dice_before_prune_round=best_dice_before_prune_round, best_dice_before_agg_round=best_dice_before_agg_round, best_before_agg=True, save_path_dict=save_path_dict)
        if round % 5 == 0:
            helper_save_model(round, global_unet, global_segmenter, list_zero_out_dicts_unets, list_zero_out_dicts_segmenters, val_loss, val_dice, model_save_path, local_dices_before=local_dices_before, local_dices_after=local_dices_after, best_dice_before_agg=best_dice_before_agg, best_dice_before_prune=best_dice_before_prune, best_dice_before_prune_round=best_dice_before_prune_round, best_dice_before_agg_round=best_dice_before_agg_round, save_path_dict=save_path_dict, bn_stats_unet=bn_stats_unet, bn_stats_segmenter=bn_stats_segmenter)
            #save the avg_train_time
            np.save(model_save_path + '_train_times' + '.npy', train_times)
            np.save(model_save_path + '_test_dices' + '.npy', test_dices)
            np.save(model_save_path + '_params_remaining' + '.npy', params_remaining)
            np.save(model_save_path + '_inf_times' + '.npy', inf_times)
            np.save(model_save_path + '_model_sizes' + '.npy', model_sizes)
            # client_data = get_summary(global_model, datadict_train=datadict_train, device=device, buckets=buckets)

            # client_data = [client_data[client] for client in clients]

            # for i, (buckets_fps, buckets_fns) in enumerate(client_data):
                
            #     for j in range(len(buckets)):
            #         clients_data[i][0, j, round//5] = buckets_fps[j]
            #         clients_data[i][1, j, round//5] = buckets_fns[j]
            
            # print("Client data: ", client_data)
            # #save the clients_data
            # np.save('./clients_data.npy', clients_data)


        #print("Round: ", round+1, "Train Loss: ", train_loss, "Train Dice: ", train_dice, "Val Loss: ", val_loss, "Val Dice: ", val_dice)
        print("Round: ", round+1, "Train Loss: ", train_loss, "Train Dice: ", train_dice, "Val Loss: ", val_loss, "Val Dice: ", val_dice)

        #np.save(result_save_path + '_loss' + '.npy', [train_losses_all, val_losses_all, train_dices, val_dices_all])

        if rounds_since_improvement >= patience:
            print("Early stopping at round: ", round+1)
            break

    helper_save_model(round, global_unet, global_segmenter, list_zero_out_dicts_unets, list_zero_out_dicts_segmenters, val_loss, val_dice, model_save_path, local_dices_before=local_dices_before, local_dices_after=local_dices_after, best_dice_before_agg=best_dice_before_agg, best_dice_before_prune=best_dice_before_prune, best_dice_before_prune_round=best_dice_before_prune_round, best_dice_before_agg_round=best_dice_before_agg_round, save_path_dict=save_path_dict, bn_stats_unet=bn_stats_unet, bn_stats_segmenter=bn_stats_segmenter)
    np.save(model_save_path + '_train_times' + '.npy', train_times)
    np.save(model_save_path + '_test_dices' + '.npy', test_dices)
    np.save(model_save_path + '_params_remaining' + '.npy', params_remaining)
    np.save(model_save_path + '_inf_times' + '.npy', inf_times)
    np.save(model_save_path + '_model_sizes' + '.npy', model_sizes)
    # #save the clients_data
    # np.save('./clients_data.npy', clients_data)
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
    parser.add_argument("-method",default='FedAvg',choices=['FedAvg','FedProx', 'FedBN', 'FixBN', 'MOON'],
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
    parser.add_argument("-combine_at_single_client", default=False, action='store_true', help='Combine the data at a single client')
    args = parser.parse_args()
    
    print("-----------------------------Arguments for the current execution-----------------------------------")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    system_data_path = ''
    MODEL_DIR = ''

    #set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    datadict_train, datadict_val, datadict_test = helper_federated_setup(train_clients=args.data, val_clients=args.data, test_clients=args.data, size=args.size, no_crop=args.no_crop, use_simulated=args.use_simulated, custom=args.custom, system_data_path=system_data_path, use_simulated_as_augmentation=args.use_simulated_as_augmentation, combine_at_single_client=args.combine_at_single_client)
    
    #measure the time taken to train
    start = time.time()

    train(method=args.method, model=args.model, num_clients=len(args.data), datadict_train=datadict_train, datadict_val=datadict_val, datadict_test = datadict_test, num_rounds=args.num_rounds, num_clients_per_round=len(args.data), global_model=None, criterion=args.criterion, optimizer=None, scheduler=None, device=args.device, size=args.size, no_crop=args.no_crop, no_aug=args.no_aug, num_epochs=args.num_epochs, hyperparams=args.hyper_parameters, lmd=args.lmd, mu=args.mu, checkpoint=args.checkpoint, not_all_clients=args.not_all_clients, batch_size=args.batch, num_workers=args.workers, use_simulated=args.use_simulated, custom=args.custom, dropout_contrastive=args.dropout_contrastive, temperature=args.temperature, additional_save_path=args.exp_id, patience=args.patience, additional_losses=args.add_losses, starting_lr = args.starting_lr, lr_decay_rate = args.lr_decay_rate, min_rounds=args.min_rounds, include_lg_pw = args.local_global_pw, include_local_pw = args.local_pw, include_every=args.include_every, lrs = args.lrs, option=args.option, mu_lg=args.mu_lg, weightage=args.weightage, before_prune_epochs=args.before_prune_epochs, after_prune_epochs=args.after_prune_epochs, pruning_percentage=args.prune_percentage, pruning_mode=args.pruning_mode, smart_aggregate=args.smart_aggregate, start_round_pruning=args.start_round_pruning, introduce_importance=args.introduce_importance, importance_factor=args.importance_factor, T=args.fixbn_rounds)

    end = time.time()

    #print the time taken in hours
    print("Time taken in hours: ", (end-start)/3600)