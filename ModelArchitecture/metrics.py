import numpy as np
from scipy.spatial.distance import directed_hausdorff, cdist

def Hausdorff_Distance_95(pred, gt):
    # Ensure inputs are boolean arrays
    pred = pred > 0.5
    gt = gt > 0
    
    # Handle empty masks
    if np.sum(pred) == 0 and np.sum(gt) == 0:
        return 0
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return np.sqrt(np.sum(np.array(pred.shape) ** 2))
    
    # Get coordinates of boundaries
    pred_points = np.argwhere(pred)
    gt_points = np.argwhere(gt)
    
    # Calculate distances using broadcasting
    dists_pred_to_gt = np.linalg.norm(pred_points[:, np.newaxis] - gt_points[np.newaxis, :], axis=2)
    
    # Calculate minimums in both directions
    min_dists_pred_to_gt = np.min(dists_pred_to_gt, axis=1)
    min_dists_gt_to_pred = np.min(dists_pred_to_gt, axis=0)
    
    # Combine all distances and calculate 95th percentile
    all_distances = np.concatenate([min_dists_pred_to_gt, min_dists_gt_to_pred])
    return np.percentile(all_distances, 95)

def Dice_Score(pred,target,threshold=0.5):
    smooth = 1
    pred_cmb = (pred > threshold).astype(float)
   
    true_cmb = (target > 0).astype(float)

    pred_vect = pred_cmb.reshape(-1)
    true_vect = true_cmb.reshape(-1)
    intersection = np.sum(pred_vect * true_vect)
    if np.sum(pred_vect) == 0 and np.sum(true_vect) == 0:
        dice_score = (2. * intersection + smooth) / (np.sum(pred_vect) + np.sum(true_vect) + smooth)
    else:
        dice_score = (2. * intersection) / (np.sum(pred_vect) + np.sum(true_vect))
        #print('intersection, true_vect, pred_vect')
        #print(intersection, np.sum(true_vect), np.sum(pred_vect))
    return dice_score

# Voxel Wise
def TPR(pred,target,threshold=0.5):
    eps = 1e-7
    pred = pred.reshape(-1)
    target = target.reshape(-1) 
    target = (target > 0).astype(float)

    TP = np.sum(target*(pred>threshold))
    P = np.sum(target)
    return (TP+eps)/(P+eps)

# Voxel Wise
def TNR(pred,target,threshold=0.5):
    eps = 1e-7
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    target = (target > 0).astype(float)

    TN = np.sum((1-target)*(pred<=threshold))
    N = np.sum(1-target)
    return (TN+eps)/(N+eps)

# Voxel Wise
def FPR(pred,target,threshold=0.5):
    eps = 1e-7
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    target = (target > 0).astype(float)

    FP = np.sum((1-target)*(pred>threshold))
    N = np.sum(1-target)
    return (FP+eps)/(N+eps)

# Voxel Wise
def F1_score(pred,target,threshold=0.5):
    eps = 1e-7
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    target = (target > 0).astype(float)

    TP = np.sum(target*(pred>threshold))
    FP = np.sum((1-target)*(pred>threshold))
    FN = np.sum(target*(pred<threshold))
    
    return (2*TP+eps)/(2*TP+FP+FN+eps)


def Hausdorff_Distance(pred, target, threshold=0.5):
    pred_cmb = (pred > threshold).astype(float)
    true_cmb = (target > 0).astype(float)
    pred_points = np.argwhere(pred_cmb)
    true_points = np.argwhere(true_cmb)
    
    d1 = directed_hausdorff(pred_points, true_points)[0]
    d2 = directed_hausdorff(true_points, pred_points)[0]
    
    return max(d1, d2)

def Hausdorff_Distance_95(pred, target, threshold=0.5):
    pred_cmb = (pred > threshold).astype(float)
    true_cmb = (target > 0).astype(float)
    
    # Get the coordinates of the points in the predicted and true sets
    pred_points = np.argwhere(pred_cmb)
    true_points = np.argwhere(true_cmb)
    
    if len(pred_points) == 0 or len(true_points) == 0:
        return np.inf
    
    # Compute pairwise distances
    distances = cdist(pred_points, true_points, 'euclidean')
    
    # Get the 95th percentile distance from predictions to ground truth
    hd95_pred_to_true = np.percentile(distances.min(axis=1), 95)
    
    # Get the 95th percentile distance from ground truth to predictions
    hd95_true_to_pred = np.percentile(distances.min(axis=0), 95)
    
    # Return the maximum of the two
    return max(hd95_pred_to_true, hd95_true_to_pred)
