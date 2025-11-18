import torch

DIM_FACTOR = 32 # 16

def collate_pad(batch_in):
    #note that each element in the batch is of shape (1, d1, d2, d3)
    #batch_in is a list of dictionaries. each dictionary has the keys 'input' and 'gt'
    #calculate the max dimensions of the images in the batch
    d1 = max([image_dict['input'].shape[1] for image_dict in batch_in])
    d2 = max([image_dict['input'].shape[2] for image_dict in batch_in])
    d3 = max([image_dict['input'].shape[3] for image_dict in batch_in])

    #make sure that the dimensions are divisible by DIM_FACTOR
    d1 = d1 + (DIM_FACTOR - d1 % DIM_FACTOR) if d1 % DIM_FACTOR != 0 else d1
    d2 = d2 + (DIM_FACTOR - d2 % DIM_FACTOR) if d2 % DIM_FACTOR != 0 else d2
    d3 = d3 + (DIM_FACTOR - d3 % DIM_FACTOR) if d3 % DIM_FACTOR != 0 else d3

    images = []
    gts = []
    paths = []
    affines = []

    path_in = True
    return_orig = True if "affine" in batch_in[0].keys() else False

    if "path" not in batch_in[0].keys():
        path_in = False

    for image_dict in batch_in:
        image = image_dict['input']
        gt = image_dict['gt']
        affine = image_dict['affine'] if return_orig else None
        if path_in:
            path = image_dict['path']

        images.append(torch.nn.functional.pad(image, (0, d3 - image.shape[3], 0, d2 - image.shape[2], 0, d1 - image.shape[1])))
        gts.append(torch.nn.functional.pad(gt, (0, d3 - gt.shape[3], 0, d2 - gt.shape[2], 0, d1 - gt.shape[1])))
        if path_in:
            paths.append(path)
        if return_orig:
            affines.append(affine)
    
    return {'input':torch.stack(images), 'gt':torch.stack(gts), 'path':paths, 'affine':affines}

D1, D2, D3 = 336, 256, 112

def collate_padmax(batch_in):
    #note that each element in the batch is of shape (1, d1, d2, d3)
    #batch_in is a list of dictionaries. each dictionary has the keys 'input' and 'gt'
    #calculate the max dimensions of the images in the batch
    d1 = D1
    d2 = D2
    d3 = D3

    images = []
    gts = []
    paths = []

    for image_dict in batch_in:
        image = image_dict['input']
        gt = image_dict['gt']
        path = image_dict['path']

        images.append(torch.nn.functional.pad(image, (0, d3 - image.shape[3], 0, d2 - image.shape[2], 0, d1 - image.shape[1])))
        gts.append(torch.nn.functional.pad(gt, (0, d3 - gt.shape[3], 0, d2 - gt.shape[2], 0, d1 - gt.shape[1])))
        paths.append(path)
    
    return {'input':torch.stack(images), 'gt':torch.stack(gts), 'path':paths}

def collate_resize(batch_in):

    #note that each element in the batch is of shape (1, d1, d2, d3)
    
    #batch_in is a list of dictionaries. each dictionary has the keys 'input' and 'gt'
    #calculate the max dimensions of the images in the batch
    d1 = max([image_dict['input'].shape[1] for image_dict in batch_in])
    d2 = max([image_dict['input'].shape[2] for image_dict in batch_in])
    d3 = max([image_dict['input'].shape[3] for image_dict in batch_in])

    #make sure that the dimensions are divisible by DIM_FACTOR
    d1 = d1 + (DIM_FACTOR - d1 % DIM_FACTOR) if d1 % DIM_FACTOR != 0 else d1
    d2 = d2 + (DIM_FACTOR - d2 % DIM_FACTOR) if d2 % DIM_FACTOR != 0 else d2
    d3 = d3 + (DIM_FACTOR - d3 % DIM_FACTOR) if d3 % DIM_FACTOR != 0 else d3
    images = []
    gts = []

    for image_dict in batch_in:
        image = image_dict['input']
        gt = image_dict['gt']
        
        #how to extend the dimensions of the image from (1,d1,d2,d3) to (1, 1, d1, d2, d3)

        image = image.unsqueeze(0)
        gt = gt.unsqueeze(0)
        images.append(torch.nn.functional.interpolate(image, size=(d1, d2, d3), mode='trilinear', align_corners=True).squeeze(0))
        gts.append(torch.nn.functional.interpolate(gt, size=(d1, d2, d3), mode='trilinear', align_corners=True).squeeze(0))

        #return a single dictionary with the keys 'input' and 'gt'

    return {'input':torch.stack(images), 'gt':torch.stack(gts)}
