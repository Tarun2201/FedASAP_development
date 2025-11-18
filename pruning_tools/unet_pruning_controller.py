import torch
import numpy as np
import os
import json
from collections import defaultdict
from heapq import nsmallest
from operator import itemgetter
from pruning_tools.unet_filter_pruner import UNet5FilterPruner
from pruning_tools.unet_prune_layer import prune_conv_layer
import torch.nn.functional as F
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import logit
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from pruning_tools.features_linear import compute_entropy, compute_l1_norm, compute_l2_norm, compute_mean, compute_variance, compute_sparsity
import pandas as pd 
import joblib
import numpy as np

num_clients = 0
num_rounds = 1
class UNet5FilterTracker:
    """Tracks pruned filters for the UNet5 model"""
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.pruned_filters = defaultdict(list)  # {iteration: [(layer_name, filter_num)]}
        self.filter_map_file = os.path.join(save_dir, 'unet5_filter_tracking.json')
        self.remaining_filters_file = os.path.join(save_dir, 'unet5_remaining_filters.json')
        
    def add_pruned_filters(self, iteration, filters):
        """Record filters pruned in current iteration"""
        self.pruned_filters[str(iteration)].extend(filters)
        self.save_filter_map()
        
    def save_filter_map(self):
        """Save the pruning history to a JSON file"""
        with open(self.filter_map_file, 'w') as f:
            json.dump(self.pruned_filters, f, indent=2)
            
    def save_remaining_filters(self, model):
        """Save currently remaining filters in the model"""
        remaining_filters = {}
        filter_count = 0
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv3d):
                remaining_filters[name] = {
                    "out_channels": module.out_channels,
                    "filter_numbers": list(range(filter_count, filter_count + module.out_channels))
                }
                filter_count += module.out_channels
                        
        with open(self.remaining_filters_file, 'w') as f:
            json.dump(remaining_filters, f, indent=2)

def print_layer_values(data):
    """Print the importance values for each layer"""
    # Group data by layer
    layer_data = {}
    
    for item in data:
        layer = item[0]  # Layer name
        value = float(item[2])  # Importance value
        
        if layer not in layer_data:
            layer_data[layer] = []
        layer_data[layer].append(value)
    
    # Print results by layer
    for layer in sorted(layer_data.keys()):
        print(f"Layer {layer}")
        print(' '.join(str(x) for x in layer_data[layer]))
        print()

class UNet5PruningController:
    """
    Controller for pruning the UNet5 model
    Adapts the approach from the original PruningController class
    """
    def __init__(
        self, 
        unet5, 
        train_data_loader, 
        criterion, 
        prune_percentage=10, 
        mode='Taylor', 
        pruning_pth=None, 
        device=torch.device('cpu'),
        zero_out_dict=None,
        introduce_importance=False,
        importance_factor=0.1,
        glm_dir = None,
        num_clients_round=4
    ):
        self.unet5 = unet5
        self.train_data_loader = train_data_loader
        self.mode = mode
        self.pruner = UNet5FilterPruner(self.unet5, self.mode, device, zero_out_dict=zero_out_dict)
        self.criterion = criterion
        self.prune_percentage = prune_percentage
        self.pruning_pth = pruning_pth
        self.device = device
        self.filter_tracker = UNet5FilterTracker(os.path.dirname(pruning_pth) if pruning_pth else '.')
        self.introduce_importance = introduce_importance
        self.importance_factor = importance_factor
        self.zero_out_dict = zero_out_dict if zero_out_dict is not None else {}
        self.retention_thresholds = {}
        self.glm_dir = glm_dir
        self.num_clients_round = num_clients_round
        if introduce_importance:
            self._set_importance_probs()
        
    def _set_importance_probs(self):
        """Set importance probabilities for different layers"""
        # For UNet5, we can set different probabilities for different levels
        # Encoder levels (deeper -> higher importance)
        self.retention_thresholds["encoder1"] = 0.5 - 2*self.importance_factor
        self.retention_thresholds["encoder2"] = 0.5 - self.importance_factor
        self.retention_thresholds["encoder3"] = 0.5
        self.retention_thresholds["encoder4"] = 0.5 + self.importance_factor
        # self.retention_thresholds["encoder5"] = 0.5 + 2*self.importance_factor
        
        # Bottleneck has highest importance
        self.retention_thresholds["bottleneck"] = 0.5 + 2*self.importance_factor
        
        # Decoder levels (deeper -> higher importance)
        # self.retention_thresholds["decoder5"] = 0.5 + 2*self.importance_factor
        self.retention_thresholds["decoder4"] = 0.5 + self.importance_factor
        self.retention_thresholds["decoder3"] = 0.5
        self.retention_thresholds["decoder2"] = 0.5 - self.importance_factor
        self.retention_thresholds["decoder1"] = 0.5 - 2*self.importance_factor

    def train_batch(self, optimizer, batch, label, rank_filters):
        """Train a single batch and optionally rank filters"""
        batch = batch.to(self.device)
        label = label.to(self.device)

        self.unet5.zero_grad()

        if rank_filters:
            # Forward pass through the pruner to collect activations and gradients
            outputs, bottleneck = self.pruner.forward(batch)
            loss = self.criterion(outputs, label)
            loss.backward()
        else:
            # Normal forward pass
            outputs, _ = self.unet5(batch)
            loss = self.criterion(outputs, label)
            loss.backward()
            optimizer.step()
        
    def train_epoch(self, optimizer=None, rank_filters=False):
        max_h , max_w, max_d = 0, 0, 0
        for i,data in enumerate(self.train_data_loader):
            batch = data['input']
            label = data['gt']
            if batch.shape[2] > max_h:
                max_h = batch.shape[2]
            if batch.shape[3] > max_w:
                max_w = batch.shape[3]
            if batch.shape[4] > max_d:
                max_d = batch.shape[4]
        for i, data in enumerate(self.train_data_loader):
            inputs_list = data['input']   # list of B tensors of shape (C, h_i, w_i, d_i)
            labels_list = data['gt']      # same structure
            padded_inputs = []
            padded_labels = []
            for x, y in zip(inputs_list, labels_list):
                # ensure 4D
                if x.ndim == 3:
                    x = x.unsqueeze(0)
                    y = y.unsqueeze(0)

                _, h, w, d = x.shape
                pad_h = max_h - h
                pad_w = max_w - w
                pad_d = max_d - d
                # torch.nn.functional.pad takes padding (D_front, D_back, W_front, W_back, H_front, H_back)
                padding = (0, pad_d, 0, pad_w, 0, pad_h)

                padded_inputs.append(F.pad(x, padding))
                padded_labels.append(F.pad(y, padding))
            batch = torch.stack(padded_inputs, dim=0)  # → (B, C, max_h, max_w, max_d)
            label = torch.stack(padded_labels, dim=0)
            # print(batch.shape, label.shape)
            self.train_batch(optimizer, batch, label, rank_filters)

    def sampling_activations(self, dict_activation, dict_taylor):
    # Flatten the Taylor values and record mapping (key, index)
        all_taylor = []
        mapping = []  # Each element is a tuple: (key, index)
        original_total = 0
        
        # Process keys sorted by their integer value
        for key in sorted(dict_taylor.keys(), key=lambda k: int(k)):
            # Each dict_taylor[key] is a tensor of Taylor values
            taylor_tensor = dict_taylor[key]
            values = taylor_tensor.tolist()  # convert tensor to list for flattening
            for i, val in enumerate(values):
                all_taylor.append(val)
                mapping.append((key, i))
            original_total += len(values)
        
        # Convert the flattened list of Taylor values to a tensor
        all_taylor = torch.tensor(all_taylor, dtype=torch.float32)
        
        # Calculate sample sizes based on the original total number of filters
        num_lowest = max(1, int(original_total * 0.01))
        num_remaining_sample = max(1, int(original_total * 0.05))
        
        # Remove zeros: keep only indices where Taylor value is not zero
        nonzero_mask = all_taylor != 0
        nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0]
        if nonzero_indices.numel() == 0:
            # If no nonzero Taylor values exist, return empty dictionaries for all keys
            empty_dict = {k: torch.tensor([]) for k in dict_activation.keys()}
            return empty_dict, empty_dict, empty_dict

        # Filter Taylor values and mapping to include only nonzero ones
        filtered_taylor = all_taylor[nonzero_indices]
        filtered_mapping = [mapping[i] for i in nonzero_indices.tolist()]
        
        # Sort the filtered Taylor values in ascending order and get their indices
        sorted_order = torch.argsort(filtered_taylor)
        
        # Select the lowest group (lowest num_lowest values)
        lowest_indices = sorted_order[:min(num_lowest, len(sorted_order))]
        
        # From the remaining, randomly sample num_remaining_sample entries
        remaining_indices = sorted_order[min(num_lowest, len(sorted_order)):]
        if len(remaining_indices) > 0:
            permuted = remaining_indices[torch.randperm(len(remaining_indices))]
            sampled_remaining_indices = permuted[:min(num_remaining_sample, len(permuted))]
        else:
            sampled_remaining_indices = torch.tensor([], dtype=torch.long)
        
        # Combine the selected indices (relative to the filtered list)
        selected_indices = torch.cat([lowest_indices, sampled_remaining_indices])
        
        # Recover the mapping for the selected indices and corresponding Taylor values
        selected_mapping = [filtered_mapping[i] for i in selected_indices.tolist()]
        selected_taylor_values = filtered_taylor[selected_indices]
        
        # Build output dictionaries using the original key types
        selected_activation_dict = {k: [] for k in dict_activation.keys()}
        selected_taylor_dict = {k: [] for k in dict_taylor.keys()}
        # New dictionary: for each key, record a binary indicator (0 for lowest group, 1 for remaining)
        selected_type_dict = {k: [] for k in dict_activation.keys()}
        
        # Determine how many of the selected indices belong to the lowest group
        lowest_count = lowest_indices.numel()
        
        # Iterate through the selected mapping and assign corresponding values
        for i, (key, idx) in enumerate(selected_mapping):
            # Append the Taylor value (as a float)
            selected_taylor_dict[key].append(selected_taylor_values[i].item())
            # Append the corresponding activation by indexing the tensor for this key
            selected_activation_dict[key].append(dict_activation[key][:,idx])
            # Determine type: 0 if from lowest group, 1 if from remaining
            type_indicator = 0 if i < lowest_count else 1
            selected_type_dict[key].append(type_indicator)
        
        # Convert lists to tensors for consistency
        for key in selected_activation_dict:
            if selected_activation_dict[key]:
                # If activations are scalars or tensors, stack them along a new dimension.
                selected_activation_dict[key] = torch.stack(selected_activation_dict[key], dim=1)
            else:
                selected_activation_dict[key] = torch.tensor([])
        
        for key in selected_taylor_dict:
            if selected_taylor_dict[key]:
                selected_taylor_dict[key] = torch.tensor(selected_taylor_dict[key])
            else:
                selected_taylor_dict[key] = torch.tensor([])
        
        for key in selected_type_dict:
            if selected_type_dict[key]:
                selected_type_dict[key] = torch.tensor(selected_type_dict[key], dtype=torch.long)
            else:
                selected_type_dict[key] = torch.tensor([])
        
        return selected_activation_dict, selected_taylor_dict, selected_type_dict

    def get_candidates_to_prune(self, num_filters_to_prune):
        global num_rounds
        global num_clients
        """Get candidates for pruning based on importance scores"""
        self.pruner.reset()

        self.train_epoch(rank_filters=True)
        self.pruner.normalize_ranks_per_layer()
        returned_ranks = self.pruner.return_ranks
        if num_clients<self.num_clients_round: # put the number of clients you are using here 
            num_clients += 1
        else:
            num_clients = 1
            num_rounds += 1
        # Get pruning plan
        pruning_plan, avg_activations, avg_filter_ranks = self.pruner.lowest_ranking_filters(num_filters_to_prune)
        
        return pruning_plan, returned_ranks, avg_activations, avg_filter_ranks

    def total_num_filters(self):
        """Count total number of filters in the model"""
        filters = 0
        for name, module in self.unet5.named_modules():
            name = name.split('.')[-1]
            if isinstance(module, torch.nn.Conv3d) and name not in self.unet5.ignore_for_pruning and name not in 'final_conv':
                filters += module.out_channels
        return filters

    def prune(self):
        """Perform pruning on the model"""
        # Make sure all parameters are trainable
        for param in self.unet5.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = int(self.prune_percentage/100 * number_of_filters)
        
        # Get candidates for pruning
        prune_targets, returned_ranks, avg_activations, avg_filter_ranks = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
        print(avg_activations.keys(), avg_filter_ranks.keys())
        activation_save, taylor_save, labels_taylor = self.sampling_activations(avg_activations, avg_filter_ranks)
        print(activation_save.keys(), taylor_save.keys(), labels_taylor.keys())
        # Track pruned filters
        pruned_filters = [(layer_name, filter_index) for layer_name, filter_index, _ in prune_targets]
        self.filter_tracker.add_pruned_filters(len(self.filter_tracker.pruned_filters), pruned_filters)
        data_mapping = {"encoder1conv1": 0, "encoder1conv2": 1, "encoder2conv1": 2, "encoder2conv2": 3, "encoder3conv1": 4, "encoder3conv2": 5, "encoder4conv1": 6, "encoder4conv2": 7, "bottleneckconv1": 8, 
            "bottleneckconv2": 9,
            "decoder4conv1": 10,
            "decoder4conv2": 11,
            "decoder3conv1": 12,
            "decoder3conv2": 13,
            "decoder2conv1": 14,
            "decoder2conv2": 15,
            "decoder1conv1": 16,
        }
        # Group pruned filters by layer
        layers_pruned = {}
        for layer_name, filter_index, _ in prune_targets:
            if layer_name not in layers_pruned:
                layers_pruned[layer_name] = 0
            layers_pruned[layer_name] += 1
            
        # Log pruning information
        if self.pruning_pth:
            file_object = open(self.pruning_pth, 'a')
            file_object.write('\n')
            layers_pruned_str = json.dumps(layers_pruned)
            file_object.write(layers_pruned_str)
            file_object.close()

        # Move model to CPU for pruning
        unet5 = self.unet5.cpu()
        zero_out_dict = self.zero_out_dict
        # Apply pruning with importance-based filtering

        csv_rows = []
        if num_rounds<10:
            pass
        else:
            for i in activation_save:
                if activation_save[i].numel() == 0:
                    pass
                else:
                    for j in range(0, activation_save[i].shape[1]):
                        entropy = compute_entropy(activation_save[i][:,j]).detach().cpu().numpy()
                        l2_norm = compute_l2_norm(activation_save[i][:,j]).detach().cpu().numpy()
                        mean_activation = compute_mean(activation_save[i][:,j]).detach().cpu().numpy()
                        csv_rows.append([entropy.item(), l2_norm.item(), mean_activation.item(), labels_taylor[i][j].item(), num_rounds])
            # file_name = f'/mnt/b0305b0a-824d-48cb-a829-2a6766e6b45b/tarun2/fets_regression_data/features_data/activations_{num_rounds}_{num_clients}.csv'
            df = pd.DataFrame(csv_rows, columns=['entropy', 'l2_norm', 'mean', 'label', 'round'])
            # df.to_csv(file_name)
            X = df[['entropy', 'mean', 'l2_norm']]  # Features
            y = df['label']  # Using binary labels from 'label_1' column
            # Split data into training and testing sets (80/20 split)
            y = y.astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            report_rows = []

            X_design = sm.add_constant(X_train)
            family = Binomial(link=logit())  # for logistic regression
            glm_model = sm.GLM(y_train, X_design, family=family)
            glm_results = glm_model.fit()
            best_f1 = 0
            X_test_design = sm.add_constant(X_test)
            y_pred_proba = glm_results.predict(X_test_design)

            model_filename = f'{self.glm_dir}_glm_{num_rounds}_{num_clients}.pkl'
            to_save = {
                "model": glm_results,
                "best_threshold": 0.8, #change the threshold here
            }
            joblib.dump(to_save, model_filename)

        if num_rounds<11:
            pass
        else:
            for layer_name, filter_index, _ in prune_targets:
                filter_activation = avg_activations[data_mapping[layer_name]][0, filter_index, :, :, :]
                loaded_data = joblib.load(f'{self.glm_dir}_glm_{num_rounds-1}_{num_clients}.pkl')
                loaded_model = loaded_data['model']
                threshold = loaded_data['best_threshold']
                entropy = compute_entropy(filter_activation).detach().cpu().numpy()
                l2_norm = compute_l2_norm(filter_activation).detach().cpu().numpy()
                mean_activation = compute_mean(filter_activation).detach().cpu().numpy()
                sample = pd.DataFrame({
                    'entropy': [entropy],
                    'mean': [mean_activation],
                    'l2_norm': [l2_norm]
                })
                # Make prediction
                sample = np.asarray(sample, dtype=float)

                # reshape to (1, p)
                exog = sample.reshape(1, -1)

                # add the intercept column
                exog_design = sm.add_constant(exog, has_constant='add')

                # now predict—it will return a length-1 array of probabilities
                prediction = loaded_model.predict(exog_design)
                # if self.introduce_importance:
                #     prob = np.random.rand()
                #     if prob > self.set_probs[batch_index]:
                #         continue
                if prediction[0] > threshold:
                    continue
                # Apply zero-out to the filter
                unet5, zero_out_dict = prune_conv_layer(unet5, layer_name, filter_index, zero_out_dict, device=self.device)

        # Update model and move back to device
        self.unet5 = unet5
        self.unet5 = self.unet5.to(self.device)
        self.zero_out_dict = zero_out_dict
        # Clean up
        self.pruner.remove_hooks()
        del self.pruner

        return returned_ranks, zero_out_dict
