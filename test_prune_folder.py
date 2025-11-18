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
from models.unet_model_targeted_dropout import UNet4_new
import matplotlib.pyplot as plt
import nibabel as nib
import copy
import time
from medpy.metric.binary import hd95 as medpy_hd95
from train_fed_with_parser_args_prune_redone_opt import get_reduced_model


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
        count += torch.sum(param == value).item()
    return count

def count_num_params_deleted(model):
    return count_params_having_value(model, 0)

def count_total_params(model):
    count = 0
    for param in model.parameters():
        count += param.numel()
    return count
    
def helper_test(unet, dataloader, device, option=1):
    unet.eval()
    test_dice = 0
    test_tpr = 0
    test_lesion_tpr = 0
    test_precision = 0
    test_f1_score = 0
    test_hd = 0
    hd95_list = []
    test_size = len(dataloader)
    inference_times = []

    predictions = []
    ground_truth = []

    with tqdm(range(test_size)) as pbar:
        for i, data in zip(pbar, dataloader):
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            
            with torch.no_grad():
                image = data['input'].to(device)
                start_time = time.time()
                output, bottleneck = unet(image)
                inference_times.append(time.time() - start_time)
                label = data['gt'].to(device)

                if type(output) == list or type(output) == tuple:
                    output = output[-1]
                output = output.squeeze()
                label = label.squeeze()

                label = label.cpu().detach().numpy()
                if option == 1:
                    output = skiform.resize(output.cpu().detach().numpy(), label.shape, order=1, preserve_range=True)
                else:
                    output = output.cpu().detach().numpy()

                dice = Dice_Score(output, label)
                tpr = TPR(output, label)
                lesion_tpr, precision, f1_score = Lesion_Metrics(output, label)
                hd = 0 # Hausdorff_Distance(output, label)
                binary_output = (output > 0.5).astype(bool)
                binary_label = (label > 0.5).astype(bool)
                if np.sum(binary_output) == 0 or np.sum(binary_label) == 0:
                    current_hd95 = np.nan
                else:
                    current_hd95 = medpy_hd95(binary_output, binary_label)

            torch.cuda.synchronize(device)
            

            test_dice += dice
            test_tpr += tpr
            test_lesion_tpr += lesion_tpr
            test_precision += precision
            test_f1_score += f1_score
            test_hd += hd
            hd95_list.append(current_hd95)

            predictions.append(output)
            ground_truth.append(label)
    
    avg_inference_time = np.mean(inference_times)
    test_hd95 = np.nanmean(hd95_list)
    return (predictions, ground_truth, 
            test_dice/test_size, test_tpr/test_size, test_lesion_tpr/test_size, 
            test_precision/test_size, test_f1_score/test_size, 
            test_hd/test_size, test_hd95, avg_inference_time)

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
    parser.add_argument("-option", default=3, type=int, choices=[1, 2, 3])
    parser.add_argument("-on_validation", default=False, action='store_true', help='To test on validation set')
    parser.add_argument("-folder_path", default=None, help='Path to the folder containing the models')
    parser.add_argument("-binary_before_after_string", default="1111", help='Binary string to decide which models to use for each client, 0<-before, 1<-after')
    parser.add_argument("-save_scans", default=False, action='store_true', help='To save the scans')
    parser.add_argument("-centralized", default=False, action='store_true', help='To test on centralized model')
    parser.add_argument("-system_data_path", default=None, type=str, help='path of the data')
    parser.add_argument("-model_dir", default=None, type=str, help='path of the model directory')
    args = parser.parse_args()

    print("-----------------------------Arguments for the current execution-----------------------------------")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    
    option = args.option
    folder_path = args.folder_path
    binary_before_after_string = args.binary_before_after_string
    

    system_data_path = args.system_data_path
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

    for client in datadict_test.keys():
        test_dataloaders[client] = DataLoader(datadict_test[client], batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)
    device = 'cuda:'+str(args.device)

    

    results = []
    optimal_results = []
    
    for i, char in enumerate(binary_before_after_string):
        if not args.centralized:
            if char == '0':
                pre = '_state_dict_best_dice_local_model_client_'+args.data[i]+'_before_prune'
            else:
                pre = '_state_dict_best_dice_local_model_client_'+args.data[i]+'_after_prune'
        else:
            if char == '0':
                pre = '_state_dict_best_dice_local_model_client_'+'combined_client'+'_before_prune'
            else:
                pre = '_state_dict_best_dice_local_model_client_'+'combined_client'+'_after_prune'


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
        zero_out_dict = checkpoint['lists_zero_out_dicts']
        #print(zero_out_dict)

        unet = UNet4_new(init_features=16)

        unet.load_state_dict(checkpoint['unet_state_dict'])

        unet.to(device)

        unet_modules = list(unet._modules.items())
        # print("=== BEFORE REDUCTION ===")
        number_params_before_unet = count_total_params(unet)
        print("Total parameters in the unet before pruning:", number_params_before_unet)
        # print("Full model parameters:")
        # print_model_parameters(unet, "UNet")
        # print_model_parameters(segmenter, "Segmenter")


        unet = get_reduced_model(unet, zero_out_dict, device)

        number_params_after_unet = count_total_params(unet) - count_num_params_deleted(unet)
        print("Total parameters in the unet after pruning: ", number_params_after_unet)

        params_removed_percentage = 100 - (number_params_after_unet*100/number_params_before_unet)

        predictions, ground_truth, test_dice, test_tpr, test_lesion_tpr, test_precision, test_f1_score, test_hd, test_hd95, test_inference_time = helper_test(unet, test_dataloaders[args.data[i]], device, option=option)
    
        results.append([args.data[i], round_number, test_dice, test_tpr, test_lesion_tpr, test_precision, test_f1_score, test_hd95, 
        params_removed_percentage, test_inference_time])
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
    for col in range(2, 10):  # extend range to include new metrics
        avg = np.mean([float(row[col]) for row in results])
        avg_metrics.append(avg)

    results.append(avg_metrics)
    results1 = np.array(results)
    np.savetxt(folder_path + 'results.txt', results1, fmt='%s')
    print("Results saved to: ", folder_path + 'results.txt')
    #print("Client, Round, Dice, TPR, Lesion-level TPR, Lesion-level Precision, Lesion-level F1 Score, HD, HD95, Percent Removed")

    print("Client\tRound\tDice\tTPR\tLesion-level TPR\tLesion-level Precision\tLesion-level F1 Score\tHD95\t%TotalParams\tAvgInferenceTime")
    for i in range(len(results)):
        for j in range(len(results[i])):
            num = results[i][j]
            if not isinstance(num, str) and not isinstance(num, int):
                print("{:.3f}".format(num), end='\t')
            else:
                print(results[i][j], end='\t')
        print()