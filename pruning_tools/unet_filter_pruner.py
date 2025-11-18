import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

IN = 0
OUT = 1

class UNet5FilterPruner:
    """
    Filter pruner for UNet5 model
    Adapts the approach from the original FilterPruner class
    """
    def __init__(self, unet5, mode='Taylor', device=torch.device('cpu'), zero_out_dict=None):
        self.unet5 = unet5
        self.mode = mode
        self.device = device
        self.return_ranks = None
        self.hooks = []
        self.reset()
        self.zero_out_dict = zero_out_dict
        self.hook_fn = {
            'Taylor': self.compute_rank_taylor,
            'Random': self.compute_rank_random,
            'L1': self.compute_rank_l1,
            'L2': self.compute_rank_l2,
            'L1_std': self.compute_rank_l1_std
        }.get(self.mode)
        
    def reset(self):
        """Reset the filter ranks dictionary"""
        self.filter_ranks = {}
        self.avg_activations = {}
        
    def _process_block(self, block, input_tensor, activation_index, layer_name):
        """Process a block by applying each layer sequentially and registering hooks for convolutional layers.
        
        Args:
            block: The block module to process
            input_tensor: Input tensor for the block
            activation_index: Current activation index
            layer_name: Name of the layer being processed
            
        Returns:
            tuple: (output tensor, updated activation_index)
        """
        # Process each layer in the block sequentially
        output = input_tensor
        
        # Iterate through named modules in the block
        for name, module in block.named_modules():
            
            if name == "":
                continue
            output = module(output)

            if isinstance(module, nn.Conv3d):
                
                # Register hook for the output of this convolutional layer
                if name in self.unet5.ignore_for_pruning:
                    continue
                self.hooks.append(output.register_hook(self.hook_fn))
                self.activations.append(output)
                self.activation_to_layer_name[activation_index] = name
                activation_index += 1
        
        return output, activation_index
    
    def forward(self, x):
        """Forward pass through the UNet5 model with hooks for filter pruning"""
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer_name = {}
        activation_index = 0
        
        # Encoder path
        enc1, activation_index = self._process_block(self.unet5.encoder1, x, activation_index, "encoder1")
        
        enc2_input = self.unet5.pool(enc1)
        enc2, activation_index = self._process_block(self.unet5.encoder2, enc2_input, activation_index, "encoder2")
        
        enc3_input = self.unet5.pool(enc2)
        enc3, activation_index = self._process_block(self.unet5.encoder3, enc3_input, activation_index, "encoder3")
        
        enc4_input = self.unet5.pool(enc3)
        enc4, activation_index = self._process_block(self.unet5.encoder4, enc4_input, activation_index, "encoder4")
        
        # enc5_input = self.unet5.pool(enc4)
        # enc5, activation_index = self._process_block(self.unet5.encoder5, enc5_input, activation_index, "encoder5")
        
        # Bottleneck
        bottleneck_input = self.unet5.pool(enc4)
        bottleneck, activation_index = self._process_block(self.unet5.bottleneck, bottleneck_input, activation_index, "bottleneck")
        
        # Decoder with skip connections
        # dec5_up = self.unet5.upconv5(bottleneck)
        # dec5_input = torch.cat([dec5_up, enc5], dim=1)
        # dec5, activation_index = self._process_block(self.unet5.decoder5, dec5_input, activation_index, "decoder5")

        dec4_up = self.unet5.upconv4(bottleneck)
        dec4_input = torch.cat([dec4_up, enc4], dim=1)
        dec4, activation_index = self._process_block(self.unet5.decoder4, dec4_input, activation_index, "decoder4")
        
        dec3_up = self.unet5.upconv3(dec4)
        dec3_input = torch.cat([dec3_up, enc3], dim=1)
        dec3, activation_index = self._process_block(self.unet5.decoder3, dec3_input, activation_index, "decoder3")
        
        dec2_up = self.unet5.upconv2(dec3)
        dec2_input = torch.cat([dec2_up, enc2], dim=1)
        dec2, activation_index = self._process_block(self.unet5.decoder2, dec2_input, activation_index, "decoder2")
        
        dec1_up = self.unet5.upconv1(dec2)
        dec1_input = torch.cat([dec1_up, enc1], dim=1)
        dec1, activation_index = self._process_block(self.unet5.decoder1, dec1_input, activation_index, "decoder1")
        
        # Output
        output = self.unet5.final_conv(dec1)
        return torch.sigmoid(output), dec1
            
    def compute_rank_taylor(self, grad):
        """Compute Taylor expansion based importance score"""
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        
        taylor = activation * grad
        taylor = taylor.mean(dim=(0, 2, 3, 4)).data
        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
            if self.device != torch.device('cpu'):
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].to('cpu')
        
        self.filter_ranks[activation_index] += taylor.cpu()
        if activation_index not in self.avg_activations:
            self.avg_activations[activation_index] = self.activations[activation_index].clone()
        else:
            # print("okay, it's done")
            self.avg_activations[activation_index] = self.activations[activation_index] + self.avg_activations[activation_index]
        self.grad_index += 1
        
    def compute_rank_l1(self, grad):
        """Compute L1 norm based importance score"""
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        
        l1 = torch.norm(activation, p=1, dim=2)
        l1 = torch.norm(l1, p=1, dim=2)
        l1 = torch.norm(l1, p=1, dim=2)
        l1 = l1.mean(dim=0).data
        
        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
            if self.device != torch.device('cpu'):
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].to('cpu')
        
        self.filter_ranks[activation_index] += l1.cpu()
        self.grad_index += 1
        
    def compute_rank_l1_std(self, grad):
        """Compute L1 norm standard deviation based importance score"""
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        
        l1 = torch.norm(activation, p=1, dim=2)
        l1 = torch.norm(l1, p=1, dim=2)
        l1 = torch.norm(l1, p=1, dim=2)
        l1 = l1.std(dim=0).data
        
        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
            if self.device != torch.device('cpu'):
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].to('cpu')
        
        self.filter_ranks[activation_index] += l1.cpu()
        self.grad_index += 1
        
    def compute_rank_l2(self, grad):
        """Compute L2 norm based importance score"""
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        
        l2 = torch.norm(activation, p=2, dim=2)
        l2 = torch.norm(l2, p=2, dim=2)
        l2 = torch.norm(l2, p=2, dim=2)
        l2 = l2.mean(dim=0).data
        
        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
            if self.device != torch.device('cpu'):
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].to(self.device)
        
        self.filter_ranks[activation_index] += l2.cpu()
        self.grad_index += 1
        
    def compute_rank_random(self, grad):
        """Compute random importance score"""
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        
        if activation.size()[1] != 1:
            taylor = activation * grad
            taylor = taylor.mean(dim=(0, 2, 3, 4)).data
            random = torch.rand(taylor.size())
        else:
            taylor = activation * grad
            taylor = taylor.mean(dim=(0, 2, 3, 4)).data
            random = torch.ones(taylor.size()) * 100
            
        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
            if self.device != torch.device('cpu'):
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].to(self.device)
        
        self.filter_ranks[activation_index] += random.cpu()
        self.grad_index += 1
        
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
            
    def normalize_ranks_per_layer(self):
        """Normalize the importance scores for each layer"""
        if not self.mode == 'Random':
            for i in self.filter_ranks:
                v = torch.abs(self.filter_ranks[i])
                v = v / np.sqrt(torch.sum(v * v))
                self.filter_ranks[i] = v.cpu()
        self.return_ranks = self.filter_ranks
        
    def lowest_ranking_filters(self, num):
        """Return the lowest ranking filters based on importance scores"""
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                if self.zero_out_dict is None:
                    self.zero_out_dict = {}
                layer_name = self.activation_to_layer_name[i]
                if layer_name in self.zero_out_dict:
                    if (j, OUT) in self.zero_out_dict[self.activation_to_layer_name[i]]: #out simply means that the output channel of that index is zeroed out
                        continue
                data.append((layer_name, j, self.filter_ranks[i][j])) #j is the filter index
        return sorted(data, key=lambda x: x[2])[:num], self.avg_activations, self.filter_ranks