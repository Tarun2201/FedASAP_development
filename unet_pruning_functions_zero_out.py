import torch
from losses.dice_loss import dice_loss
import numpy as np
import torch.optim as optim
import sys
from pruning_tools.unet_pruning_controller import UNet5PruningController
# from pruning_tools.prune_manager_unet_fixed_filters_zero_out import PruningController

def pruning_function(unet, zero_out_dict, train_dataloader, pruning_percentage, pruning_mode, device=torch.device("cpu"), introduce_importance=False, importance_factor=0.1, glm_dir = None, num_clients=4):

    if pruning_percentage == 0:
        return unet, zero_out_dict
    criterion = dice_loss()
    criterion.to(device)
    # print("Zero out dict: ", zero_out_dict)

    fine_tuner = UNet5PruningController(unet, train_dataloader, criterion, prune_percentage=pruning_percentage, mode=pruning_mode, device=device, zero_out_dict=zero_out_dict, introduce_importance=introduce_importance, importance_factor=importance_factor, glm_dir=glm_dir, num_clients_round=num_clients)
    returned_ranks, zero_out_dict = fine_tuner.prune()
    # print("Zero out dict after pruning: ", zero_out_dict)
    # torch.save(unet, unet_save_pth+'_pruned_iter_'+str(iteration))
    # torch.save(segmenter, segmenter_save_pth+'_pruned_iter_'+str(iteration))

    return unet, zero_out_dict



def new_weights(returned_ranks):
    for key in returned_ranks:
        tot = len(returned_ranks[key])
        returned_ranks[key] = (torch.sum(returned_ranks[key]) / tot).numpy()
        returned_ranks[key] = 1 - returned_ranks[key]

    m = max(returned_ranks.values())

    for key in returned_ranks:
        returned_ranks[key] = returned_ranks[key] / m

    base_prob = 0.10
    for key in returned_ranks:
        returned_ranks[key] = returned_ranks[key] * base_prob
    return returned_ranks

def new_weights_rankval(d):
    new_d = {}
    for key in d:
        array = d[key].numpy()
        for i in array:
            if i not in new_d.keys():
                new_d[i] = key
            else:
                i += 1e-12
                new_d[i] = key
    o_keys = sorted(new_d.keys())

    i = len(o_keys)
    prune_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
                  16: 0}
    for val in o_keys:
        i -= 1
        index = new_d[val]
        prune_dict[index] += i

    for key in prune_dict:
        prune_dict[key] = prune_dict[key] / len(d[key].numpy())
    max_val = max(prune_dict.values())
    for key in prune_dict:
        prune_dict[key] = prune_dict[key] / max_val

    base_prob = 0.10
    for key in prune_dict:
        prune_dict[key] = prune_dict[key] * base_prob
    return prune_dict

def update_droplayer(model, new_drop):
    model.encoder1.enc1drop1 = torch.nn.Dropout3d(p=new_drop[0])
    model.encoder1.enc1drop2 = torch.nn.Dropout3d(p=new_drop[1])
    model.encoder2.enc2drop1 = torch.nn.Dropout3d(p=new_drop[2])
    model.encoder2.enc2drop2 = torch.nn.Dropout3d(p=new_drop[3])
    model.encoder3.enc3drop1 = torch.nn.Dropout3d(p=new_drop[4])
    model.encoder3.enc3drop2 = torch.nn.Dropout3d(p=new_drop[5])
    model.encoder4.enc4drop1 = torch.nn.Dropout3d(p=new_drop[6])
    model.encoder4.enc4drop2 = torch.nn.Dropout3d(p=new_drop[7])
    model.bottleneck.bottleneckdrop1 = torch.nn.Dropout3d(p=new_drop[8])
    model.bottleneck.bottleneckdrop2 = torch.nn.Dropout3d(p=new_drop[9])
    model.decoder4.dec4drop1 = torch.nn.Dropout3d(p=new_drop[10])
    model.decoder4.dec4drop2 = torch.nn.Dropout3d(p=new_drop[11])
    model.decoder3.dec3drop1 = torch.nn.Dropout3d(p=new_drop[12])
    model.decoder3.dec3drop2 = torch.nn.Dropout3d(p=new_drop[13])
    model.decoder2.dec2drop1 = torch.nn.Dropout3d(p=new_drop[14])
    model.decoder2.dec2drop2 = torch.nn.Dropout3d(p=new_drop[15])
    model.decoder1.dec1drop2 = torch.nn.Dropout3d(p=new_drop[16])
    return model