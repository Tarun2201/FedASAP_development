import torch
from torch.utils.data import DataLoader
from helper_datadict import *
from tqdm import tqdm
import torch.nn as nn
import argparse
import os
import ast
import numpy as np
from ModelArchitecture.metrics import *
import skimage.transform as skiform
from utils import collate_resize, collate_pad
from skimage import measure
from scipy.ndimage import find_objects
from skimage.measure import label
from models.unet_model_targeted_dropout import UNet5, Segmenter
import matplotlib.pyplot as plt
import nibabel as nib
import copy
from train_fed_with_parser_args_prune import get_reduced_model


CURRENT_DIRECTORY = os.getcwd()
MODEL_DIR = CURRENT_DIRECTORY + '/models/' 
RESULTS_DIR = CURRENT_DIRECTORY + '/results/'

def extract_exp_id(file_name, keyword="Federated"):
    portions = file_name.split('/')
    for i, portion in enumerate(portions):
        if portion == 'Federated':
            return portions[i+2]
    return "temp"

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

def count_params_having_value(model, value):
    
    #go through all the parameters of the model and count the number of parameters that have the value
    count = 0
    for param in model.parameters():
        print(torch.sum(param == value).item(), end = "<--")
        
        print(param.shape, param.type())
        count += torch.sum(param == value).item()
    return count

def count_num_params_deleted(model):
    return count_params_having_value(model, 0)

def count_non_zero_params(model):
    count = 0
    for param in model.parameters():
        count += torch.sum(param != 0).item()
    return count

def count_total_params_non_zero(model):
    return count_non_zero_params(model)

def count_total_params(model):
    count = 0
    for param in model.parameters():
        count += param.numel()
    return count

def helper_test(unet, segmenter, dataloader, device, option=1):
    unet.eval()
    segmenter.eval()
    test_dice = 0
    test_tpr = 0
    test_lesion_tpr = 0
    test_precision = 0
    test_f1_score = 0
    test_hd = 0
    test_hd95 = 0
    test_size = len(dataloader)

    predictions = []
    ground_truth = []

    with tqdm(range(test_size)) as pbar:
        for i, data in zip(pbar, dataloader):
            torch.cuda.empty_cache()

            with torch.no_grad():
                image = data['input'].to(device)
                features = unet(image)
                # #extras has intermediate outputs
                # for each in extras:
                #     #find the minimum and maximum values in each intermediate output and the average of the non-zero values
                #     print("Min: ", torch.min(each).item(), "Max: ", torch.max(each).item(), "Mean: ", torch.mean(each[each != 0]).item())
                #     #number of zeros in the intermediate output
                #     print("Number of zeros: ", torch.sum(each == 0).item())
                #     print("Total number of elements: ", each.numel())

                output, bottleneck = segmenter(features)
                label = data['gt'].to(device)

                if type(output) == list or type(output) == tuple:
                    output = output[-1]
                output = output.squeeze()
                label = label.squeeze()

                label = label.cpu().detach().numpy()
                if option == 1:
                    output = skiform.resize(output.cpu().detach().numpy(), label.shape, order = 1, preserve_range=True)
                else:
                    output = output.cpu().detach().numpy()

                dice = Dice_Score(output, label)
                tpr = TPR(output, label)
                lesion_tpr, precision, f1_score = Lesion_Metrics(output, label)
                hd = Hausdorff_Distance(output, label)

                #print("Sample path: ", data['path'], " dice: ", dice, " tpr: ", tpr)
                #hd95 = Hausdorff_Distance_95(output, label)
                hd95 = 0
                test_dice += dice
                test_tpr += tpr
                test_lesion_tpr += lesion_tpr
                test_precision += precision
                test_f1_score += f1_score
                test_hd += hd
                test_hd95 += hd95

                predictions.append(output)
                ground_truth.append(label)
    
    return predictions, ground_truth, test_dice/test_size, test_tpr/test_size, test_lesion_tpr/test_size, test_precision/test_size, test_f1_score/test_size, test_hd/test_size, test_hd95/test_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", nargs='+', default=['client1', 'client2', 'client3', 'client4'], choices=['client1', 'client2', 'client3', 'client4', 'client5', 'client6', 'client7', 'client8', 'client9', 'client10', 'client11'], help='Which clients to test on?')
    parser.add_argument("-model",default='unet', choices=['unet','slimunetr','ducknet','saunet','nestedunet','halfunet','resunet','unetr','sacunet'],help='Which model to run ?')
    parser.add_argument("-workers", default=4, type=int)
    parser.add_argument("-hyperparam",default="{'init_features':16}",dest='hyper_parameters',type=ast.literal_eval,help='Pass dictionary of hyperparameter if needs changing.')
    parser.add_argument("-device", default=0, type=int, choices=[0,1])
    parser.add_argument("-size", nargs='+', default=(128, 128, 128), type=int, help='Input size of the model')
    parser.add_argument("-dataset", default='fets', choices=['fets', 'wmh', 'wmh_fp', 'combined', 'combined_40percent'], help='Which dataset to test on?')
    parser.add_argument("-no_crop", default=False, action='store_true', help='To not have the model tight crop the images')
    parser.add_argument("-system", default=63, type=int, choices=[63, 64, 131, 67, 66], help='Which system to run the code on?')
    parser.add_argument("-option", default=1, type=int, choices=[1, 2, 3])
    parser.add_argument("-on_validation", default=False, action='store_true', help='To test on validation set')
    parser.add_argument("-folder_path", default=None, help='Path to the folder containing the models')
    parser.add_argument("-binary_before_after_string", default="1111", help='Binary string to decide which models to use for each client, 0<-before, 1<-after')
    parser.add_argument("-save_scans", default=False, action='store_true', help='To save the scans')
    args = parser.parse_args()

    print("-----------------------------Arguments for the current execution-----------------------------------")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    
    option = args.option
    folder_path = args.folder_path
    binary_before_after_string = args.binary_before_after_string
    

    system_data_path = './fets_clients/'

    if args.dataset.lower() == 'wmh':
        if args.system == 63:
            system_data_path = '/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6/Karan/WMH/wmh_clients/'
        
        elif args.system == 131:
            system_data_path = '/mnt/a4ef64ea-1b6b-4423-b1d2-4794d2e97289/Karan/WMH/wmh_clients/'
        
        elif args.system == 64:
            system_data_path = '/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/Karan/WMH/wmh_clients/'
        
        elif args.system == 67:
            system_data_path = '/mnt/b0305b0a-824d-48cb-a829-2a6766e6b45b/Karan/WMH/wmh_clients/'
        elif args.system == 66:
            system_data_path = '/mnt/b63ea8f0-19df-47e7-9305-168c698c54ce/Karan/WMH/wmh_clients/'

    elif args.dataset.lower() == 'wmh_fp':
        if args.system == 63:
            system_data_path = '/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6/Karan/WMH/wmh_clients_new/'
        
        elif args.system == 131:
            system_data_path = '/mnt/a4ef64ea-1b6b-4423-b1d2-4794d2e97289/Karan/WMH/wmh_clients_new/'
        
        elif args.system == 64:
            system_data_path = '/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/Karan/WMH/wmh_clients_new/'


    elif args.dataset.lower() == 'combined':
        if args.system == 63:
            system_data_path = '/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6/Karan/combined_clients/'
        
        elif args.system == 131:
            system_data_path = '/mnt/a4ef64ea-1b6b-4423-b1d2-4794d2e97289/Karan/combined_clients/'
        
        elif args.system == 64:
            system_data_path = '/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/Karan/combined_clients/'
        elif args.system == 67:
            system_data_path = '/mnt/b0305b0a-824d-48cb-a829-2a6766e6b45b/Karan/combined_clients/'
        elif args.sytem == 66:
            system_data_path = '/mnt/b63ea8f0-19df-47e7-9305-168c698c54ce/Karan/combined_clients/'
    
    elif args.dataset.lower() == 'combined_40percent':
        if args.system == 63:
            system_data_path = '/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6/Karan/combined_clients_40_percent/'
        
        elif args.system == 131:
            system_data_path = '/mnt/a4ef64ea-1b6b-4423-b1d2-4794d2e97289/Karan/combined_clients_40_percent/'
        
        elif args.system == 64:
            system_data_path = '/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/Karan/combined_clients_40_percent/'
        elif args.system == 67:
            system_data_path = '/mnt/b0305b0a-824d-48cb-a829-2a6766e6b45b/Karan/combined_clients_40_percent/'
        elif args.sytem == 66:
            system_data_path = '/mnt/b63ea8f0-19df-47e7-9305-168c698c54ce/Karan/combined_clients_40_percent/'

    if args.system == 64:
        MODEL_DIR = '/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/Karan/projects/FedSamp/models/'
    elif args.system == 63:
        MODEL_DIR = '/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6/Karan/projects/FedSamp/models/'
    elif args.system == 131:
        MODEL_DIR = '/mnt/a4ef64ea-1b6b-4423-b1d2-4794d2e97289/Karan/projects/FedSamp/models/'
    elif args.system == 67:
        MODEL_DIR = '/mnt/b0305b0a-824d-48cb-a829-2a6766e6b45b/Karan/projects/FedSamp/models/'
    elif args.system == 66:
        MODEL_DIR = '/mnt/b63ea8f0-19df-47e7-9305-168c698c54ce/Karan/projects/FedSamp/models/'

    return_orig = True if args.save_scans else False
    
    if not args.on_validation:
        _, _, datadict_test = helper_federated_setup(test_clients=args.data, size=args.size, no_crop=args.no_crop, system_data_path=system_data_path, return_orig=return_orig)
    else:
        _, datadict_test, _ = helper_federated_setup(val_clients=args.data, size=args.size, no_crop=args.no_crop, system_data_path=system_data_path, return_orig=return_orig)
    
    test_dataloaders = {}

    collate_fn = None

    if option == 2:
        collate_fn = collate_resize
    elif option == 3:
        collate_fn = collate_pad

    test_dataloaders1 = {}
    for client in datadict_test.keys():
        test_dataloaders[client] = DataLoader(datadict_test[client], batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)
        test_dataloaders1[client] = DataLoader(datadict_test[client], batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)
    device = 'cuda:'+str(args.device)

    

    results = []
    optimal_results = []
    
    for i, char in enumerate(binary_before_after_string):
        if char == '0':
            pre = '_state_dict_best_dice_local_model_client_'+args.data[i]+'_before_prune'
        else:
            pre = '_state_dict_best_dice_local_model_client_'+args.data[i]+'_after_prune'

        #get the file_paths that start with pre
        #print(os.listdir(folder_path))
        file_paths = [f for f in os.listdir(folder_path) if f.startswith(pre)]
        #print("File Paths: ", file_paths)

        final_file = file_paths[0]
        round_number = int(final_file[len(pre):-4])

        for j in range(1, len(file_paths)):
            no = int(file_paths[j][len(pre):-4])
            if no > round_number:
                round_number = no
                final_file = file_paths[j]

        print("Loading model from: ", final_file)
        checkpoint = torch.load(folder_path + final_file)
        zero_out_dict_unet = checkpoint['lists_zero_out_dicts_unets']
        zero_out_dict_segmenter = checkpoint['lists_zero_out_dicts_segmenters']
        #print(zero_out_dict_unet)

        unet = UNet5(init_features=16, drop_probs={i:0.005 for i in range(21)})
        segmenter = Segmenter(init_features=16)

        unet.load_state_dict(checkpoint['unet_state_dict'])
        segmenter.load_state_dict(checkpoint['segmenter_state_dict'])

        unet.to(device)
        segmenter.to(device)

        layer_names = []
        total_filter_counts = []
        removed_counts = []

        unet_modules = list(unet._modules.items())
        for key in sorted(zero_out_dict_unet.keys()):
            if len(key) != 3:
                continue
            parent = unet_modules[key[0]]         # (parent_name, parent_module)
            child_modules = list(parent[1]._modules.items())   # get child modules of parent_module
            child = child_modules[key[2]]
                   # (grandchild_name, grandchild_module)
            if isinstance(unet_modules[key[0]][key[1]][key[2]], torch.nn.Conv3d):

                #print("Parent: ", parent[0], "Child: ", child[0])
                total_filter_counts.append(child[1].out_channels)
                rem_count = 0
                for tup in zero_out_dict_unet[key]:
                    if tup[1] == 1:
                        rem_count += 1
                removed_counts.append(rem_count)
                layer_names.append("{}.{}".format(parent[0], child[0]))

        
        
        seg_modules = list(segmenter._modules.items())
        seg_remove_counts = []
        for key in sorted(zero_out_dict_segmenter.keys()):
            if len(key) != 3:
                continue
            if isinstance(seg_modules[key[0]][key[1]][key[2]], torch.nn.Conv3d):
                #total_filter_counts.append(seg_modules[key[0]][key[1]][key[2]].out_channels)
                rem_count = 0
                for tup in zero_out_dict_segmenter[key]:
                    if tup[1] == 1:
                        rem_count += 1
                seg_remove_counts.append(rem_count)
        
        
        
        # print("=== BEFORE REDUCTION ===")
        number_params_before_unet = count_total_params(unet)
        number_params_before_segmenter = count_total_params(segmenter)
        print("Total parameters in the unet before pruning:", number_params_before_unet)
        print("Total parameters in the segmenter before pruning:", number_params_before_segmenter)
        # print("Full model parameters:")
        # print_model_parameters(unet, "UNet")
        # print_model_parameters(segmenter, "Segmenter")


        # unet = get_reduced_model(unet, zero_out_dict_unet, device)
        # segmenter = get_reduced_model(segmenter, zero_out_dict_segmenter, device)

        number_params_after_unet = count_total_params_non_zero(unet)
        number_params_after_segmenter = count_total_params_non_zero(segmenter)
        print("Total parameters in the unet after pruning: ", number_params_after_unet)
        print("Total parameters in the segmenter after pruning: ", number_params_after_segmenter)

        

        
        # print_model_parameters(unet, "UNet")
        # print_model_parameters(segmenter, "Segmenter")
                
        
        total = np.array(total_filter_counts)
        removed = np.array(removed_counts)

        percentage_removed = (removed*100)/total

        params_removed_unet_percentage = 100 - (number_params_after_unet*100/number_params_before_unet)
        params_removed_segmenter_percentage = 100 - (number_params_after_segmenter*100/number_params_before_segmenter)
        params_removed_percentage = 100 - (number_params_after_unet + number_params_after_segmenter)*100/(number_params_before_unet + number_params_before_segmenter)
        
        plt.figure()
        plt.bar(np.arange(len(percentage_removed)), percentage_removed)
        plt.xlabel('Layer')
        plt.ylabel('Percentage of filters removed')
        plt.title('Percentage of filters removed in each layer')
        plt.grid(True)
        plt.savefig(folder_path + args.data[i] + '_percentage_removed_filters.png')

        percent_removed = np.sum(removed)*100/np.sum(total)

        predictions, ground_truth, test_dice, test_tpr, test_lesion_tpr, test_precision, test_f1_score, test_hd, test_hd95 = helper_test(unet, segmenter, test_dataloaders1[args.data[i]], device, option=option)
        results.append([args.data[i], round_number, test_dice, test_tpr, test_lesion_tpr, test_precision, test_f1_score, test_hd, percent_removed, 
               params_removed_unet_percentage, params_removed_segmenter_percentage, params_removed_percentage])
        #print("Client: ", args.data[i], "Dice: ", test_dice, "TPR: ", test_tpr, "Lesion TPR: ", test_lesion_tpr, "Precision: ", test_precision, "F1 Score: ", test_f1_score, "HD: ", test_hd, "HD95: ", test_hd95)
        
        if not args.save_scans:
            continue
        
        exp = extract_exp_id(folder_path)

        
        for index, (data, prediction, gt) in enumerate(zip(test_dataloaders[args.data[i]], predictions, ground_truth)):
            client = args.data[i]
            data_path = data['path']
            #print("Data Path: ", data_path)
            components = data_path[0].split('/')
            folder_path1 = '/'.join(components[:-1]) + '/'

            prediction = np.squeeze(prediction)
            gt = np.squeeze(gt)

            prediction = prediction > 0.5

            #set the type of the prediction and ground truth to np.int8
            prediction = prediction.astype(np.int8)
            

            #save the prediction and ground truth
            #print all the keys in the data dictionary
            #print(data.keys())
            #print(data['affine'])
            affine = data['affine']
            affine = np.array(affine)[0]
            #print(affine.shape)
            #save prediction in the same folder. name it prediction_exp.nii.gz
            prediction_nii = nib.Nifti1Image(prediction, affine)
            nib.save(prediction_nii, folder_path1 + 'prediction__'+exp+'.nii.gz') 
            #save ground truth in the same folder. name it ground_truth_padded.nii.gz if it does not exist
            gt_nii = nib.Nifti1Image(gt, affine)
            if not os.path.exists(folder_path1 + 'ground_truth_padded.nii.gz'):
                nib.save(gt_nii, folder_path1 + 'ground_truth_padded.nii.gz')
            #save the input image in the same folder. name it input.nii.gz
            input_nii = nib.Nifti1Image(data['input'][0, 0].numpy(), affine)
            if not os.path.exists(folder_path1 + 'input.nii.gz'):
                nib.save(input_nii, folder_path1 + 'input.nii.gz')
            


        #remove the model from the GPU
        # unet.cpu()
        # segmenter.cpu()
        # torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary(device=device, abbreviated=False))

    #calculate the average of the results
    #append a line to the resilts which containes "all clients", "-", avg dice, avg tpr, avg lesion tpr, avg precision, avg f1 score, avg hd, avg percent removed
    avg_metrics = ["all clients", "-"]  # client name and round placeholders
    avg_metrics_optimal = ["all clients", "-"]  # client name and round placeholders
    for col in range(2, 12):  # extend range to include new metrics
        avg = np.mean([float(row[col]) for row in results])
        avg_metrics.append(avg)

    results.append(avg_metrics)
    results1 = np.array(results)
    np.savetxt(folder_path + 'results.txt', results1, fmt='%s')
    print("Results saved to: ", folder_path + 'results.txt')
    #print("Client, Round, Dice, TPR, Lesion-level TPR, Lesion-level Precision, Lesion-level F1 Score, HD, HD95, Percent Removed")

    print("Client\tRound\tDice\tTPR\tLesion-level TPR\tLesion-level Precision\tLesion-level F1 Score\tHD\t%Removed\t%UnetParams\t%SegParams\t%TotalParams")
    for i in range(len(results)):
        for j in range(len(results[i])):
            num = results[i][j]
            #if it is a number, round it to 4 decimal places
            if not isinstance(num, str) and not isinstance(num, int):
                print("{:.3f}".format(num), end='\t')
            else:
                print(results[i][j], end='\t')
        print()
    