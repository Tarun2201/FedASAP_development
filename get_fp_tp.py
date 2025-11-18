import torch
import torch.nn as nn
import argparse
import ast
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
from ModelArchitecture.Losses import PixelwiseContrastiveLoss
from scipy.ndimage import find_objects, gaussian_filter
from skimage.measure import label, regionprops


def get_summary(model, datadict_train, device, clients=["client1", "client2", "client3", "client4", "client5"], buckets_size = [(0,2), (2, 4), (4, 10), (10, 30), (30, 60)], buckets_contrast=[(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1)]):
    model.eval()
    client_data = {}
    #print("A")
    for client in clients:
        train_dataloader = DataLoader(datadict_train[client], batch_size=1, shuffle=False, num_workers=1)
        buckets_num_fps = np.zeros(len(buckets_size))
        buckets_num_fns = np.zeros(len(buckets_size))

        #size_contrast_joint = np.zeros((len(buckets_size), len(buckets_contrast)))
        size_contrast_joint = np.ones((len(buckets_size), len(buckets_contrast)))*0.01 #for smoothing
        #print("B")
        for i, data in enumerate(train_dataloader):

            image = data['input'].to(device)
            gt = data['gt'].cpu().squeeze().numpy().astype(np.uint8)
            
            with torch.no_grad():
                output, _ = model.forward(image)
            outmap = output.cpu().detach().squeeze().numpy() > 0.5
            #print("C")
            #clean the image using dog
            image = image.cpu().detach().squeeze().numpy()
            blurred1 = gaussian_filter(image, sigma=0.5)
            blurred2 = gaussian_filter(image, sigma=1)
            dog = blurred1 - blurred2

            cleaned_image = image - dog
            #print("D")
            
            # Extract the coloured 
            pred_labelled = label(outmap)

            #print("Coloured map extracted for predictions")

            #for each set of coloured points in pred_labelled, check if there is some pixel in gt that is also coloured. If yes, colour this cluster 0 in the outmap.
            #do find_objects on pred_labelled
            objects = find_objects(pred_labelled)
            #print("E")
            #print("Objects found in the prediction map")

            for obj in objects:
                pred_cluster = pred_labelled[obj]
                pred_mask = pred_cluster
                gt_mask = gt[obj]

                if np.any(np.logical_and(pred_mask, gt_mask)):
                    pred_labelled[obj] = 0
            
            #print("F")

            #print("Relabelling done")
            
            # Extract the coloured regions from pred_labelled. These are the false positives.
            fp_regions = regionprops(pred_labelled)

            #print("Regions (FPs) extracted")
            #print("G")
            for region in fp_regions:   
                smal = region.major_axis_length/2

                # #Extract the bounding box coordinates
                # min_row, min_col, min_slice, max_row, max_col, max_slice = region.bbox

                # #Extract the contrast of the region
                # lesion_mask = pred_labelled[min_row:max_row, min_col:max_col, min_slice:max_slice] == region.label
                # background_mask = pred_labelled[min_row:max_row, min_col:max_col, min_slice:max_slice] == 0

                # lesion_contrast = np.mean(cleaned_image[min_row:max_row, min_col:max_col, min_slice:max_slice][lesion_mask])
                # background_contrast = np.mean(cleaned_image[min_row:max_row, min_col:max_col, min_slice:max_slice][background_mask])

                # weber_contrast = (lesion_contrast - background_contrast)/background_contrast

                for i, (lower, upper) in enumerate(buckets_size):
                    if lower <= smal < upper:
                        buckets_num_fps[i] += 1

                        # for j in range(len(buckets_contrast)):
                        #     lower_contrast, upper_contrast = buckets_contrast[j]
                        #     if lower_contrast <= weber_contrast < upper_contrast:
                        #         size_contrast_joint[i, j] += 1
                        # break
            
            gt_labelled = label(gt)
            objects = find_objects(gt_labelled)
            #print("H")
            for obj in objects:
                gt_cluster = gt_labelled[obj]
                #print("Unique labels in gt_cluster: ", np.unique(gt_cluster))
                #print the shape of gt cluster
                #print("Shape of gt cluster: ", gt_cluster.shape)
                #print("GT cluster", gt_cluster)
                gt_mask = gt_cluster
                #print()
                #print("Unique labels in gt_mask: ", np.unique(gt_mask))
                #print("Shape of outmap: ", outmap.shape)
                #print("Shape of outmap[obj]: ", outmap[obj].shape)
                #print("outmap[obj]: ", outmap[obj])
                pred_mask = outmap[obj]


                if np.any(np.logical_and(gt_mask, pred_mask)):
                    gt_labelled[obj] = 0
                    #print(np.unique(gt_labelled))
            
            # Extract the coloured regions from gt_labelled. These are the false negatives.
            fn_regions = regionprops(gt_labelled)
            # print("Number of unique labels in gt_labelled: ", len(np.unique(gt_labelled)))
            # print(np.unique(gt_labelled))
            #print("I")
            for region in fn_regions:
                smal = region.major_axis_length/2

                #Extract the bounding box coordinates
                min_row, min_col, min_slice, max_row, max_col, max_slice = region.bbox

                #Extract the contrast of the region
                lesion_mask = gt_labelled[min_row:max_row, min_col:max_col, min_slice:max_slice] == region.label
                background_mask = gt_labelled[min_row:max_row, min_col:max_col, min_slice:max_slice] == 0
                #print("lesion mask: ", lesion_mask)
                #print("background mask: ", background_mask)
                lesion_contrast = np.mean(cleaned_image[min_row:max_row, min_col:max_col, min_slice:max_slice][lesion_mask])
                background_contrast = np.mean(cleaned_image[min_row:max_row, min_col:max_col, min_slice:max_slice][background_mask])
                #print("I_internal")
                weber_contrast = (lesion_contrast - background_contrast)/background_contrast

                for i, (lower, upper) in enumerate(buckets_size):
                    if lower <= smal < upper:
                        buckets_num_fns[i] += 1

                        for j in range(len(buckets_contrast)):
                            lower_contrast, upper_contrast = buckets_contrast[j]
                            if lower_contrast <= weber_contrast < upper_contrast:
                                size_contrast_joint[i, j] += 1

                        break
            #print("J")
            
            
    #         gt_props = regionprops(gt_labeled)
    #         outmap_props = regionprops(outmap_labeled)
            
    #         # Create region arrays with visited flag
    #         gt_regions = [(prop, False) for prop in gt_props]
    #         outmap_regions = [(prop, False) for prop in outmap_props]

    #         # Process each ground truth region
    #         for gt_idx, (gt_region, _) in enumerate(gt_regions):
    #             gt_centroid = np.array(gt_region.centroid)
    #             gt_label = gt_region.label
                
    #             # Calculate centroids for outmap regions
    #             # unvisited_outmap_regions = [(region, idx) for idx, (region, visited) in enumerate(all_outmap_regions) if not visited]
    #             # if len(unvisited_outmap_regions) == 0:
    #             #     continue

    #             # outmap_centroids = [np.array(region.centroid) for region, _ in unvisited_outmap_regions]

    #             # # Find nearest unvisited outmap region using Euclidean distance
    #             # distances = [np.linalg.norm(gt_centroid - centroid) for centroid in outmap_centroids]
    #             # nearest_idx = np.argmin(distances)
    #             # nearest_region, outmap_array_idx = unvisited_outmap_regions[nearest_idx]

    #             closest_region, closest_idx, best_distance = None, -1, float('inf')

    #             for outmap_idx, (outmap_region, visited) in enumerate(outmap_regions):
    #                 if visited:
    #                     continue

    #                 #calc distance between centroids
    #                 outmap_centroid = np.array(outmap_region.centroid)
    #                 distance = np.linalg.norm(gt_centroid - outmap_centroid)

    #                 if distance < best_distance:
    #                     best_distance = distance
    #                     closest_region = outmap_region
    #                     closest_idx = outmap_idx

    #             if closest_region is None:
    #                 continue

    #             # Compute overlap

    #             outmap_label = closest_region.label
    #             gt_mask = (gt_labeled == gt_label)
    #             outmap_mask = (outmap_labeled == outmap_label)

    #             intersection = np.logical_and(gt_mask, outmap_mask).sum()
    #             union = np.logical_or(gt_mask, outmap_mask).sum()
    #             overlap = intersection / union
                
    #             gt_region.overlap = overlap
    #             closest_region.overlap = overlap

    #             # Mark regions as visited if overlap > 0.7
    #             if overlap > 0.7:
    #                 gt_regions[gt_idx] = (gt_region, True)
    #                 outmap_regions[closest_idx] = (closest_region, True)
            
    #         # Count false positives and false negatives
    #         for gt_region, visited in gt_regions:
    #             if not visited:
    #                 smal = gt_region.major_axis_length/2
    #                 for i, (lower, upper) in enumerate(buckets):
    #                     if lower <= smal < upper:
    #                         buckets_num_fns[i] += 1
    #                         break

    #         for outmap_region, visited in outmap_regions:
    #             if not visited:
    #                 smal = outmap_region.major_axis_length/2
    #                 for i, (lower, upper) in enumerate(buckets):
    #                     if lower <= smal < upper:
    #                         buckets_num_fps[i] += 1
    #                         break
            
    #     # #normalize the number of false positives and false negatives by the total number of false positives and false negatives
    #     # total_fps = np.sum(buckets_num_fps)
    #     # total_fns = np.sum(buckets_num_fns)

    #     # buckets_num_fps = buckets_num_fps/total_fps
    #     # buckets_num_fns = buckets_num_fns/total_fns

        print("Client ", client)
        print("Bucket number of false positives: ", buckets_num_fps)
        print("Bucket number of false negatives: ", buckets_num_fns)
        print("Contrast size joint: ", size_contrast_joint)

        #normalize by the number of samples
        buckets_num_fps = buckets_num_fps/len(train_dataloader)
        buckets_num_fns = buckets_num_fns/len(train_dataloader)
    
        #normalize the size_contrast_joint by by summing along each row and then dividing by the sum
        size_contrast_joint = size_contrast_joint/np.sum(size_contrast_joint, axis=1)[:, None]
        #normalize the buckets_num_fps and buckets_num_fns by the total number of false positives and false negatives
        total_fps = np.sum(buckets_num_fps)
        total_fns = np.sum(buckets_num_fns)

        buckets_num_fps = buckets_num_fps/total_fps
        buckets_num_fns = buckets_num_fns/total_fns

        print("Normalized bucket number of false positives: ", buckets_num_fps)
        print("Normalized bucket number of false negatives: ", buckets_num_fns)
        print("Normalized contrast size joint: ", size_contrast_joint)

        client_data[client] = {"fp_dist": buckets_num_fps, "fn_dist": buckets_num_fns, "size_contrast_joint": size_contrast_joint, "total_fps": total_fps, "total_fns": total_fns}
    
    return client_data

def get_summary1(model, datadict_train, device, clients=["client1", "client2", "client3", "client4", "client5"], buckets = [(0,2), (2, 4), (4, 10), (10, 30), (30, 60)]):
    model.eval()
    client_data = {}
    
    for client in clients:
        train_dataloader = DataLoader(datadict_train[client], batch_size=1, shuffle=False, num_workers=1)
        buckets_num_fps = np.zeros(len(buckets))
        buckets_num_fns = np.zeros(len(buckets))

        for i, data in enumerate(train_dataloader):

            image = data['input'].to(device)
            gt = data['gt'].cpu().squeeze().numpy()
            
            with torch.no_grad():
                output, _ = model.forward(image)
            outmap = output.cpu().detach().squeeze().numpy() > 0.5
            
            # Extract the coloured 
            pred_labelled = label(outmap)

            #print("Coloured map extracted for predictions")

            #for each set of coloured points in pred_labelled, check if there is some pixel in gt that is also coloured. If yes, colour this cluster 0 in the outmap.
            #do find_objects on pred_labelled
            objects = find_objects(pred_labelled)

            #print("Objects found in the prediction map")

            for obj in objects:
                pred_cluster = pred_labelled[obj]
                pred_mask = pred_cluster
                gt_mask = gt[obj]

                if np.any(np.logical_and(pred_mask, gt_mask)):
                    pred_labelled[obj] = 0

            #print("Relabelling done")
            
            # Extract the coloured regions from pred_labelled. These are the false positives.
            fp_regions = regionprops(pred_labelled)

            #print("Regions (FPs) extracted")

            for region in fp_regions:
                smal = region.major_axis_length/2
                for i, (lower, upper) in enumerate(buckets):
                    if lower <= smal < upper:
                        buckets_num_fps[i] += 1
                        break
            
            gt_labelled = label(gt)
            objects = find_objects(gt_labelled)
            
            for obj in objects:
                gt_cluster = gt_labelled[obj]
                # print("Unique labels in gt_cluster: ", np.unique(gt_cluster))
                # #print the shape of gt cluster
                # print("Shape of gt cluster: ", gt_cluster.shape)
                # print("GT cluster", gt_cluster)
                gt_mask = gt_cluster
                # print()
                # print("Unique labels in gt_mask: ", np.unique(gt_mask))
                # print("Shape of outmap: ", outmap.shape)
                # print("Shape of outmap[obj]: ", outmap[obj].shape)
                # print("outmap[obj]: ", outmap[obj])
                pred_mask = outmap[obj]


                if np.any(np.logical_and(gt_mask, pred_mask)):
                    gt_labelled[obj] = 0
                    # print(np.unique(gt_labelled))
            
            # Extract the coloured regions from gt_labelled. These are the false negatives.
            fn_regions = regionprops(gt_labelled)
            # print("Number of unique labels in gt_labelled: ", len(np.unique(gt_labelled)))
            # print(np.unique(gt_labelled))

            for region in fn_regions:
                smal = region.major_axis_length/2
                for i, (lower, upper) in enumerate(buckets):
                    if lower <= smal < upper:
                        buckets_num_fns[i] += 1
                        break

            
            
    #         gt_props = regionprops(gt_labeled)
    #         outmap_props = regionprops(outmap_labeled)
            
    #         # Create region arrays with visited flag
    #         gt_regions = [(prop, False) for prop in gt_props]
    #         outmap_regions = [(prop, False) for prop in outmap_props]

    #         # Process each ground truth region
    #         for gt_idx, (gt_region, _) in enumerate(gt_regions):
    #             gt_centroid = np.array(gt_region.centroid)
    #             gt_label = gt_region.label
                
    #             # Calculate centroids for outmap regions
    #             # unvisited_outmap_regions = [(region, idx) for idx, (region, visited) in enumerate(all_outmap_regions) if not visited]
    #             # if len(unvisited_outmap_regions) == 0:
    #             #     continue

    #             # outmap_centroids = [np.array(region.centroid) for region, _ in unvisited_outmap_regions]

    #             # # Find nearest unvisited outmap region using Euclidean distance
    #             # distances = [np.linalg.norm(gt_centroid - centroid) for centroid in outmap_centroids]
    #             # nearest_idx = np.argmin(distances)
    #             # nearest_region, outmap_array_idx = unvisited_outmap_regions[nearest_idx]

    #             closest_region, closest_idx, best_distance = None, -1, float('inf')

    #             for outmap_idx, (outmap_region, visited) in enumerate(outmap_regions):
    #                 if visited:
    #                     continue

    #                 #calc distance between centroids
    #                 outmap_centroid = np.array(outmap_region.centroid)
    #                 distance = np.linalg.norm(gt_centroid - outmap_centroid)

    #                 if distance < best_distance:
    #                     best_distance = distance
    #                     closest_region = outmap_region
    #                     closest_idx = outmap_idx

    #             if closest_region is None:
    #                 continue

    #             # Compute overlap

    #             outmap_label = closest_region.label
    #             gt_mask = (gt_labeled == gt_label)
    #             outmap_mask = (outmap_labeled == outmap_label)

    #             intersection = np.logical_and(gt_mask, outmap_mask).sum()
    #             union = np.logical_or(gt_mask, outmap_mask).sum()
    #             overlap = intersection / union
                
    #             gt_region.overlap = overlap
    #             closest_region.overlap = overlap

    #             # Mark regions as visited if overlap > 0.7
    #             if overlap > 0.7:
    #                 gt_regions[gt_idx] = (gt_region, True)
    #                 outmap_regions[closest_idx] = (closest_region, True)
            
    #         # Count false positives and false negatives
    #         for gt_region, visited in gt_regions:
    #             if not visited:
    #                 smal = gt_region.major_axis_length/2
    #                 for i, (lower, upper) in enumerate(buckets):
    #                     if lower <= smal < upper:
    #                         buckets_num_fns[i] += 1
    #                         break

    #         for outmap_region, visited in outmap_regions:
    #             if not visited:
    #                 smal = outmap_region.major_axis_length/2
    #                 for i, (lower, upper) in enumerate(buckets):
    #                     if lower <= smal < upper:
    #                         buckets_num_fps[i] += 1
    #                         break
            
    #     # #normalize the number of false positives and false negatives by the total number of false positives and false negatives
    #     # total_fps = np.sum(buckets_num_fps)
    #     # total_fns = np.sum(buckets_num_fns)

    #     # buckets_num_fps = buckets_num_fps/total_fps
    #     # buckets_num_fns = buckets_num_fns/total_fns

        client_data[client] = (buckets_num_fps, buckets_num_fns)
    
    return client_data


def main():
    path_suffix = [5*i for i in range(20)]
    buckets = [(0,2), (2, 4), (4, 10), (10, 30), (30, 60)]
    num_rounds = 100
    path_prefix = '/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6/Karan/projects/FedSamp/models/Federated/FedAvg/wmh_fedavg_lr001_redo/without_dropout_contrastive/temperature_0.07/unet/_state_dict'
    num_clients = 5
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    clients_data = [np.zeros((2, len(buckets), num_rounds//5)) for _ in range(num_clients)] #2 <- one for FPs and one for FNs
    size = (128, 128, 128)
    model = 'unet'
    global_model = helper_model(model_type=model, which_data="wmh", hyper_parameters={}, device=device, size=size)
    system_data_path = './fets_clients/'
    dataset = 'wmh'
    system = 63

    if dataset.lower() == 'wmh':
        if system == 63:
            system_data_path = '/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6/Karan/WMH/wmh_clients_old/'
        
        elif system == 131:
            system_data_path = '/mnt/a4ef64ea-1b6b-4423-b1d2-4794d2e97289/Karan/WMH/wmh_clients/'
        
        elif system == 64:
            system_data_path = '/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/Karan/WMH/wmh_clients/'
    
    clients = ['client1', 'client2', 'client3', 'client4', 'client5']
    datadict_train, _, _ = helper_federated_setup(train_clients=['client1', 'client2', 'client3', 'client4', 'client5'], val_clients=['client1', 'client2', 'client3', 'client4', 'client5'], test_clients=['client1', 'client2', 'client3', 'client4', 'client5'], system_data_path=system_data_path)
    for _, round_num in enumerate(path_suffix):
        path = path_prefix + str(round_num) + '.pth'
        global_model.load_state_dict(torch.load(path)['model_state_dict'])
        client_data = get_summary(global_model, datadict_train, device, clients=clients, buckets_size=buckets)
        client_data = [client_data[client] for client in clients]

        for i, info_dict in enumerate(client_data):
            
            print("Client ", i+1)
            fp_dist = info_dict["fp_dist"]
            total_fps = info_dict["total_fps"]
            total_fns = info_dict["total_fns"]
            fn_dist = info_dict["fn_dist"]


            print("Number of false positives: ", info_dict["total_fps"])
            print("Number of false negatives: ", info_dict["total_fns"])

            size_contrast_joint = info_dict["size_contrast_joint"]
            

            print("FP distribution: ", fp_dist)
            print("FN distribution: ", fn_dist)
            print("Size contrast joint: ", size_contrast_joint)

            
            for j in range(len(buckets)):
                clients_data[i][0, j, round_num//5] = fp_dist[j]*total_fps
                clients_data[i][1, j, round_num//5] = fp_dist[j]*total_fns
            
            print("Client ", _, " done")
        print(f"Round {round_num} done")
        #print("client data: ", client_data)
    np.save('clients_data.npy', clients_data)

main()