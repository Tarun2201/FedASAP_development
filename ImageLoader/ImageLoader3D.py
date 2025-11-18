import glob
import torch
from torch.utils.data import Dataset
import nibabel as nib
import skimage.transform as skiform
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.io import imread
from tqdm import tqdm
from skimage.exposure import rescale_intensity,equalize_hist
import numbers
from time import sleep

class ImageLoader3D(Dataset):
    def __init__(self,paths,gt_paths, simulated_paths=[], simulated_gt_paths = [], paths_clean=None,image_size =(128,128,128),type_of_imgs = 'nifty',no_crop=False, transform=None,data='wmh',resize=True, return_size = False,return_orig=False, label_resize=True, num_sims=0):
        self.paths = paths
        self.gt_paths = gt_paths
        self.paths_clean = paths_clean
        self.simulated_paths = simulated_paths
        self.simulated_gt_paths = simulated_gt_paths
        self.transform = transform
        self.image_size = image_size
        self.type_of_imgs = type_of_imgs
        self.no_crop = no_crop
        self.resize = resize
        self.label_resize = label_resize
        self.return_size = return_size
        self.return_orig = return_orig
        self.data = data
        self.num_sims = num_sims
        if isinstance(image_size, numbers.Number):
            self.image_size = (int(image_size), int(image_size))
    def __len__(self,):
        return len(self.paths)

    def __getitem__(self,index):

        #print(self.return_orig)

        image = nib.load(self.paths[index])
        affine = image.affine
        pixdim = image.header['pixdim']
        image = image.get_fdata()
        shape = image.shape
        img_crop_para = []
        gt = nib.load(self.gt_paths[index]).get_fdata()
        orig_gt = gt

        sim_images = []
        sim_gts = []

        if self.num_sims > 0:
            for i in range(self.num_sims):
                sim_images.append(nib.load(self.simulated_paths[index*3+i]).get_fdata())
                sim_gts.append(nib.load(self.simulated_gt_paths[index*3+i]).get_fdata())

        # Define the preprocessing function
        def preprocess(image, gt):
            sub_min = 0
            if not self.no_crop:
                sub_min = np.min(image)
                if sub_min < 0:
                    image -= sub_min
                image, img_crop_para = self.tight_crop_data(image)
                shape = image.shape
                gt = gt[img_crop_para[0]:img_crop_para[0] + img_crop_para[1], 
                        img_crop_para[2]:img_crop_para[2] + img_crop_para[3], 
                        img_crop_para[4]:img_crop_para[4] + img_crop_para[5]]
                image = image + sub_min
            if self.resize:
                image = skiform.resize(image, self.image_size, order=1, preserve_range=True)
                if self.label_resize:
                    gt = skiform.resize(gt, self.image_size, order=0, preserve_range=True)
            if np.isnan(image).sum() or np.isnan(gt).sum():
                print('Nan image:', self.paths[index])

            #MIN-MAX Normalization to 0-1
            image -= image.min()
            image /= (image.max() + 1e-12)

            #MIN-MAX Normalization to -1 to 1
            #image = rescale_intensity(image, in_range=(image.min(), image.max()), out_range=(-1, 1))
            
            image = np.expand_dims(image, -1).astype(np.single)
            gt = np.expand_dims(gt > 0, -1).astype(np.single)
            return image, gt

        # Preprocess the main image and ground truth
        image, gt = preprocess(image, gt)

        # Preprocess each simulated image and corresponding ground truth
        sim_samples = []

        for sim_image, sim_gt in zip(sim_images, sim_gts):
            sim_image, sim_gt = preprocess(sim_image, sim_gt)
            sim_samples.append((sim_image, sim_gt))
        
        data_dict = {}
            
        data_dict['input'] = image
        data_dict['gt'] = gt
        data_dict['sim_samples'] = sim_samples
        
        if(self.return_orig):
            data_dict['affine'] = affine
            data_dict['orig'] = orig_gt
            data_dict['shape'] = shape
            data_dict['pixdim'] = pixdim
            data_dict['crop_para'] = img_crop_para
            data_dict['path'] = self.paths[index]
        
        #print([key for key in data_dict.keys()])

        if(self.transform):
            data_dict = self.transform(data_dict)
        
        #print("After transform: ", [key for key in data_dict.keys()])

        return data_dict
    
    def cut_zeros1d(self, im_array):
        '''
     Find the window for cropping the data closer to the brain
     :param im_array: input array
     :return: starting and end indices, and length of non-zero intensity values
        '''

        im_list = list(im_array > 0)
        start_index = im_list.index(1)
        end_index = im_list[::-1].index(1)
        length = len(im_array[start_index:]) - end_index
        return start_index, end_index, length

    def tight_crop_data(self, img_data):
        '''
     Crop the data tighter to the brain
     :param img_data: input array
     :return: cropped image and the bounding box coordinates and dimensions.
        '''

        row_sum = np.sum(np.sum(img_data, axis=1), axis=1)
        col_sum = np.sum(np.sum(img_data, axis=0), axis=1)
        stack_sum = np.sum(np.sum(img_data, axis=1), axis=0)
        rsid, reid, rlen = self.cut_zeros1d(row_sum)
        csid, ceid, clen = self.cut_zeros1d(col_sum)
        ssid, seid, slen = self.cut_zeros1d(stack_sum)
        return img_data[rsid:rsid + rlen, csid:csid + clen, ssid:ssid + slen], [rsid, rlen, csid, clen, ssid, slen]
    
