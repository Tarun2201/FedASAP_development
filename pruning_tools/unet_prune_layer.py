import torch
import numpy as np

IN = 0
OUT = 1

def zero_grad_hook(mask):
    """Returns a hook function that applies mask after gradient updates"""
    def hook(grad):
        return grad * mask
    return hook

def prune_conv_layer(model, layer_name, filter_index, zero_out_dict, device=torch.device('cpu')):
    """
    Prune a specific filter from a convolutional layer in the UNet model.
    Uses the pre-defined filter dependencies to zero out all affected filters.
    
    Args:
        model: The UNet model
        layer_name: Name of the layer to prune (e.g., "encoder1conv1")
        filter_index: Index of the filter to prune
        zero_out_dict: Dictionary tracking which filters have been zeroed out
        device: Device to use for computations
        
    Returns:
        Updated model and zero_out_dict
    """
    # Find the target layer in the model
    target_layer = None
    for name, module in model.named_modules():
        if name.endswith(layer_name):
            target_layer = module
            break
            
    if target_layer is None or not isinstance(target_layer, torch.nn.Conv3d):
        print(f"Warning: Layer {layer_name} not found or not a Conv3d layer")
        return model, zero_out_dict

    if layer_name not in zero_out_dict:
        zero_out_dict[layer_name] = set([(filter_index, OUT)])
    else:
        zero_out_dict[layer_name].add((filter_index, OUT))
    

    dependencies = model.filter_dependencies[layer_name]

    for dep_type, dep_layer_name, offset in dependencies:
        
        if dep_layer_name not in zero_out_dict:
            zero_out_dict[dep_layer_name] = set([(filter_index+offset, dep_type)])
        else:
            zero_out_dict[dep_layer_name].add((filter_index+offset, dep_type))
         
    return model, zero_out_dict