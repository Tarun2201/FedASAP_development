import json
import os
from typing import Dict, Hashable, Mapping, Tuple

import monai
import numpy as np
import torch
from easydict import EasyDict
from monai.utils import ensure_tuple_rep


class ConvertToMultiChannelBasedOnBratsClassesd(monai.transforms.MapTransform):
    """
    TC WT ET
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = [monai.utils.TransformBackends.TORCH, monai.utils.TransformBackends.NUMPY]

    def __init__(
        self,
        keys: monai.config.KeysCollection,
        is2019: bool = False,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.is2019 = is2019

    def converter(self, img: monai.config.NdarrayOrTensor):
        # TC WT ET
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        if self.is2019:
            result = [
                (img == 2) | (img == 3),
                (img == 1) | (img == 2) | (img == 3),
                (img == 2),
            ]
        else:
            # TC WT ET
            result = [
                (img == 1) | (img == 4),
                (img == 1) | (img == 4) | (img == 2),
                img == 4,
            ]
            # merge labels 1 (tumor non-enh) and 4 (tumor enh) and 2 (large edema) to WT
            # label 4 is ET
        return (
            torch.stack(result, dim=0)
            if isinstance(img, torch.Tensor)
            else np.stack(result, axis=0)
        )

    def __call__(
        self, data: Mapping[Hashable, monai.config.NdarrayOrTensor]
    ) -> Dict[Hashable, monai.config.NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


def load_brats2021_dataset_images(root, directory = "train"):
    images_list = []
    #directories in root
    root_dirs = os.listdir(root)
    roots = [root + "/" + root_dir + "/" + directory for root_dir in root_dirs]
    for root in roots:
        images_path = os.listdir(root)
        
        for path in images_path:
            if path == "README.md" or path == "partitioning_1.csv" or path == "partitioning_2.csv":
                continue
            image_path = root + "/" + path + "/" + path
            flair_img = image_path + "_flair.nii.gz"
            t1_img = image_path + "_t1.nii.gz"
            t1ce_img = image_path + "_t1ce.nii.gz"
            t2_img = image_path + "_t2.nii.gz"
            seg_img = image_path + "_seg.nii.gz"
            images_list.append(
                {"image": [flair_img, t1_img, t1ce_img, t2_img], "label": seg_img}
            )
    return images_list


def get_Brats_transforms(
    config: EasyDict,
) -> Tuple[monai.transforms.Compose, monai.transforms.Compose]:
    train_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys="image"),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(
                keys=["label"], is2019=config.trainer.is_brats2019
            ),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.SpatialPadD(
                keys=["image", "label"],
                spatial_size=(255, 255, config.trainer.image_size),
                method="symmetric",
                mode="constant",
            ),
            monai.transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            monai.transforms.CenterSpatialCropD(
                keys=["image", "label"],
                roi_size=ensure_tuple_rep(config.trainer.image_size, 3),
            ),
            monai.transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                num_samples=2,
                spatial_size=ensure_tuple_rep(config.trainer.image_size, 3),
                pos=1,
                neg=1,
                image_key="image",
                image_threshold=0,
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=0
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=1
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=2
            ),
            monai.transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
            monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            monai.transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys="image"),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(
                keys="label", is2019=config.trainer.is_brats2019
            ),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            monai.transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
        ]
    )
    return train_transform, val_transform



def get_dataloader(
    config: EasyDict, data_flag: str
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_images = load_brats2021_dataset_images(config.data_root, directory = "train")
    val_images = load_brats2021_dataset_images(config.data_root, directory = "val")
    
    config.trainer.is_brats2019 = False
    train_transform, val_transform = get_Brats_transforms(config)

    train_dataset = monai.data.Dataset(
        data=train_images,
        transform=train_transform,
    )
    val_dataset = monai.data.Dataset(
        data=val_images,
        transform=val_transform,
    )

    batch_size_train = len(train_dataset)
    batch_size_val = len(val_dataset)

    train_loader = monai.data.DataLoader(
        train_dataset,
        num_workers=config.trainer.num_workers,
        batch_size=batch_size_train,
        shuffle=True,
    )


    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=config.trainer.num_workers,
        batch_size=batch_size_val,
        shuffle=False,
    )

    return train_loader, val_loader
