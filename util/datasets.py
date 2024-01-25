# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
from PIL import Image
from torchvision import datasets, transforms
from torch import TensorType
from torch.utils.data import Dataset, Subset, ConcatDataset
import numpy as np
from torch import Generator, randperm
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from typing import Union, Tuple
from collections.abc import Iterable, Callable
from functools import partial

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

__transform_dict__ = {"center_crop":transforms.CenterCrop, "random_crop":transforms.RandomCrop,
        "resize":partial(transforms.Resize, interpolation=transforms.InterpolationMode.BICUBIC)}

def is_image_file(filename:str):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def pil_loader(path, mode='RGB'):
    return Image.open(path).convert(mode)

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def find_imgs_indir(dirname:Union[str, Iterable[str]]):
    if isinstance(dirname, str):
        filenames = [os.path.join(dirname,file) for file in sorted(os.listdir(dirname)) if is_image_file(file)]
        return filenames
    elif isinstance(dirname, Iterable):
        filenames = []
        for subdir in dirname:
            subfilenames = [os.path.join(subdir,file) for file in sorted(os.listdir(subdir)) if is_image_file(file)]
            filenames.extend(subfilenames)
        return filenames
    else:
        raise NotImplementedError("dirname must be str or Iterable, found {}.".format(type(dirname)))

class InpaintDataset(Dataset):
    def __init__(self, data_path:str, mask_path:str,
                  img_loader:Callable, mask_loader:Callable,
                  img_transform:Callable, mask_transform:Callable):
        self.imgs = find_imgs_indir(data_path)
        self.masks = find_imgs_indir(mask_path)
        self.img_loader = img_loader
        self.mask_loader = mask_loader
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __getitem__(self, index) -> Tuple[TensorType, TensorType]:
        img = self.img_loader(self.imgs[index])
        mask = self.mask_loader(self.masks[index])
        return self.img_transform(img), self.mask_transform(mask)

def subset_split(dataset, lengths, generator):
    """
    split a dataset into non-overlapping new datasets of given lengths. main code is from random_split function in pytorch
    """
    indices = randperm(sum(lengths), generator=generator).tolist()
    Subsets = []
    for offset, length in zip(np.add.accumulate(lengths), lengths):
        if length == 0:
            Subsets.append(None)
        else:
            Subsets.append(Subset(dataset, indices[offset - length : offset]))
    return Subsets

def make_dataset(opt:dict):
    phase:str = opt['phase']
    dataset_args:dict = opt['dataset']
    img_size = dataset_args['img_size']
    general_transforms = []
    if dataset_args['transform'] != 'none':
        general_transforms.append(__transform_dict__[dataset_args['transform']](size=img_size))
    general_transforms.append(transforms.ToTensor())
    img_tf = transforms.Compose(general_transforms + [transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
    mask_tf = transforms.Compose(general_transforms)
    img_loader = pil_loader
    mask_loader = partial(pil_loader, mode='L')
    if isinstance(dataset_args['data_path'], str):
        assert isinstance(dataset_args['mask_path'], str)
        dataset = InpaintDataset(dataset_args['data_path'], dataset_args['mask_path'],
            img_loader, mask_loader, img_tf, mask_tf)
        if phase == 'train':
            assert 'val_split' in dataset_args.keys()
            total_length = len(dataset)
            val_length = int(total_length * dataset_args['val_split'])
            if val_length > 0:
                train_length = total_length - val_length
                train_dataset, val_dataset = subset_split(dataset, [train_length, val_length], Generator().manual_seed(dataset_args['seed']))
                return train_dataset, val_dataset
            else:
                return dataset, None
        else:
            return dataset
    elif isinstance(dataset_args['data_path'], Iterable):
        assert isinstance(dataset_args['mask_path'], Iterable)
        if phase == 'train':
            train_dataset_list = []
            val_dataset_list = []
        else:
            dataset_list = []
        for data_path, mask_path in zip(dataset_args['data_path'], dataset_args['mask_path']):
            dataset = InpaintDataset(data_path, mask_path,
                img_loader, mask_loader, img_tf, mask_tf)
            if phase == 'train':
                total_length = len(dataset)
                val_length = int(total_length * dataset_args['val_split'])
                if val_length > 0:
                    train_length = total_length - val_length
                    train_dataset, val_dataset = subset_split(dataset, [train_length, val_length], Generator().manual_seed(dataset_args['seed']))
                train_dataset_list.append(train_dataset)
                val_dataset_list.append(val_dataset)
            else:
                dataset_list.append(dataset)
        if phase == 'train':
            return ConcatDataset(train_dataset_list), ConcatDataset(val_dataset_list)
        else:
            return ConcatDataset(dataset_list)
            
    

    