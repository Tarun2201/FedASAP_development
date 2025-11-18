import torch
import numpy as np

IN = 0
OUT = 1

def zero_grad_hook(mask):
    """Returns a hook function that applies mask after gradient updates"""
    def hook(grad):
        return grad * mask
    return hook

def zero_out_and_add_hook_equivalent_unstructured(model, mask_dict):
    """
    Apply masks to model parameters and add hooks to maintain masking during training.
    
    Args:
        model: The model to mask
        mask_dict: Dictionary mapping parameter names to their binary masks
        
    Returns:
        Masked model and list of hooks
    """
    hooks = []
    
    # Apply masks immediately
    for name, param in model.named_parameters():
        if name in mask_dict:
            param.data = param.data * mask_dict[name].to(param.device)
    
    # Define hook to apply mask after each gradient update
    def hook_factory(mask):
        def hook(grad):
            return grad * mask
        return hook
    
    # Register hooks for all masked parameters
    for name, param in model.named_parameters():
        if name in mask_dict:
            mask = mask_dict[name].to(param.device)
            h = param.register_hook(hook_factory(mask))
            hooks.append(h)
    
    return model, hooks

def zero_out_unstructured(model, mask_dict):
    for name, param in model.named_parameters():
        if name in mask_dict:
            param.data = param.data * mask_dict[name].to(param.device)
    
    return model

def zero_out_and_add_hook_equivalent(model, zero_out_dict, device=torch.device('cpu')):
    """
    Apply zero-out operations and add hooks to the UNet5 model based on the zero_out_dict.
    This is adapted for the UNet5 architecture.
    
    Args:
        model: The UNet5 model
        zero_out_dict: Dictionary tracking which filters have been zeroed out
        device: Device to use for computations
        
    Returns:
        Updated model with zero-out operations and hooks applied
    """
    hooks = []
    # Process each layer in the zero_out_dict
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
        
        # Create a mask for the weight tensor
        weight_mask = torch.ones_like(layer.weight)
        bias_mask = torch.ones_like(layer.bias) if layer.bias is not None else None

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
        
        # Apply masks for each pruned filter
        for filter_index, filter_type in pruned_filters:
            if filter_type == OUT:  # Output channel
                weight_mask[filter_index] = 0
                if layer.bias is not None and layer_type != "convT":
                    bias_mask[filter_index] = 0
            
            elif filter_type == IN:  # Input channel
                if layer_type == "conv":
                    weight_mask[:, filter_index] = 0
                else:
                    print(f"Warning: Input channel pruning not supported for {layer_type} layer {layer_name}")
        
        # Apply mask to batch norm parameters if needed
        if layer_type == 'bn':
            layer.running_mean = layer.running_mean * weight_mask
            layer.running_var = layer.running_var * weight_mask
            # Avoid division by zero in batch norm
            layer.running_var[layer.running_var == 0] = 1
        
        # Apply masks to weights and biases
        layer.weight.data = layer.weight.data * weight_mask
        if layer.bias is not None:
            layer.bias.data = layer.bias.data * bias_mask
        
        # Register hooks to ensure gradients respect the masks
        hooks.append(layer.weight.register_hook(zero_grad_hook(weight_mask)))
        if layer.bias is not None:
            hooks.append(layer.bias.register_hook(zero_grad_hook(bias_mask)))
        
    return model, hooks

