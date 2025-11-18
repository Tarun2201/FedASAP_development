import torch

def compute_entropy(activation, num_bins=30):
    """
    Computes the Shannon entropy of the activation values using torch.
    
    Parameters:
        activation (torch.Tensor): The activation values as a torch tensor.
        num_bins (int): Number of bins for histogram estimation.
        
    Returns:
        torch.Tensor: Entropy value (scalar tensor).
    """
    # Flatten the activation tensor
    # print("Activation shape:", activation.shape)
    flat_activation = activation.view(-1)
    
    # For torch.histc, we need to specify min and max. We can use the min/max of the data.
    a_min = flat_activation.min().item()
    a_max = flat_activation.max().item()
    
    # Prevent a_max == a_min which would cause issues with histc.
    if a_max == a_min:
        return torch.tensor(0.0)
    
    # Compute histogram counts across the specified number of bins.
    hist = torch.histc(flat_activation, bins=num_bins, min=a_min, max=a_max)
    
    # Convert counts to probabilities by dividing by the total count.
    total = hist.sum()
    prob = hist / total
    
    # Filter out zero probabilities to avoid log(0).
    nonzero_prob = prob[prob > 0]
    
    # Compute Shannon entropy: -sum(p * log(p)).
    entropy = -torch.sum(nonzero_prob * torch.log(nonzero_prob))
    return entropy

def compute_l1_norm(activation):
    """
    Computes the L1 norm (sum of absolute values) of the activation.
    
    Parameters:
        activation (torch.Tensor): The activation values.
        
    Returns:
        torch.Tensor: L1 norm (scalar tensor).
    """
    return torch.sum(torch.abs(activation))

def compute_l2_norm(activation):
    """
    Computes the L2 norm (Euclidean norm) of the activation.
    
    Parameters:
        activation (torch.Tensor): The activation values.
        
    Returns:
        torch.Tensor: L2 norm (scalar tensor).
    """
    return torch.sqrt(torch.sum(activation ** 2))

def compute_mean(activation):
    """
    Computes the mean (average) of the activation values.
    
    Parameters:
        activation (torch.Tensor): The activation values.
        
    Returns:
        torch.Tensor: Mean value (scalar tensor).
    """
    return torch.mean(activation)

def compute_variance(activation):
    """
    Computes the variance of the activation values.
    
    Parameters:
        activation (torch.Tensor): The activation values.
        
    Returns:
        torch.Tensor: Variance (scalar tensor).
    """
    return torch.var(activation)

def compute_sparsity(activation, epsilon=1e-5):
    """
    Computes the sparsity measure of the activation.
    Sparsity is the ratio of elements with absolute value less than epsilon to the total number of elements.
    
    Parameters:
        activation (torch.Tensor): The activation values.
        epsilon (float): Threshold to determine near-zero values.
        
    Returns:
        torch.Tensor: Sparsity ratio (scalar tensor between 0 and 1).
    """
    flat_activation = activation.view(-1)
    num_near_zero = torch.sum(torch.lt(torch.abs(flat_activation), epsilon).float())
    total_elements = flat_activation.numel()
    return num_near_zero / total_elements

