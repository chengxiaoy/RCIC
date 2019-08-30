import numpy as np
import pandas as pd

from PIL import Image
from numpy import random
import torch
import torch.nn as nn
import torch.utils.data as D
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

from torchvision import models, transforms as T
import torchvision
from sklearn.model_selection import train_test_split
import os
import sys
from collections import defaultdict
from random import sample, choice
from loss import trick

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, Resize,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose
)

rgb_train_csv_path = 'new_train.csv'
rgb_test_csv_path = 'new_test.csv'

train_csv_path = './data/train.csv'
test_csv_path = './data/test.csv'
img_dir = './data'


def val_pair(df_val):
    # build same pair for metric in val phase
    sirna_val = np.array(df_val.sirna)
    sirna_index = defaultdict(list)
    for i, sirna in enumerate(sirna_val):
        sirna_index[sirna].append(i)

    val = []
    left_index = []
    for i in range(1108):
        index1, index2 = sample(sirna_index[i], 2)
        val.append(df_val.iloc[index1])
        val.append(df_val.iloc[index2])
        sirna_index[i].remove(index1)
        sirna_index[i].remove(index2)
        left_index.extend(sirna_index[i])
    random.shuffle(left_index)
    for index in left_index:
        val.append(df_val.iloc[index])

    if len(val) % 2 == 1:
        val.remove(val[-1])

    val = pd.concat(val, ignore_index=True, axis=1).T
    print(val.shape)
    return val


def get_dataset(rgb=True, size=512, pair=False, six_channel=False):
    if rgb:
        rgb_df = pd.read_csv(rgb_train_csv_path)
        df_train, df_val = train_test_split(rgb_df, test_size=0.12, stratify=rgb_df.sirna, random_state=42)
        df_test = pd.read_csv(rgb_test_csv_path)

        # build same pair for metric in val phase
        if pair:
            df_val = val_pair(df_val)

        ds = ImagesDS(df_train, 'train', True, mode='train', augmentation=True, size=size)
        ds_val = ImagesDS(df_val, 'train', True, mode='train', size=size)
        ds_test = ImagesDS(df_test, 'test', True, mode='test', size=size)

        return ds, ds_val, ds_test
    else:
        rgb_df = pd.read_csv(train_csv_path)
        df_train, df_val = train_test_split(rgb_df, test_size=0.12, stratify=rgb_df.sirna, random_state=42)
        df_test = pd.read_csv(test_csv_path)
        if pair:
            # build same pair for metric in val phase
            df_val = val_pair(df_val)

        ds = ImagesDS(df_train, img_dir, False, mode='train', augmentation=True, size=size, six_channel=six_channel)
        ds_val = ImagesDS(df_val, img_dir, False, mode='train', augmentation=False, size=size, six_channel=six_channel)
        ds_test = ImagesDS(df_test, img_dir, False, mode='test', augmentation=False, size=size, six_channel=six_channel)
        return ds, ds_val, ds_test


class ImagesDS(D.Dataset):
    def __init__(self, df, img_dir, rgb, mode='train', augmentation=False, size=512, six_channel=False, site=1,
                 channels=[1, 2, 3, 4, 5, 6]):
        self.records = df.to_records(index=False)
        self.channels = channels
        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        if rgb:
            self.len = self.records.size // 2
        else:
            self.len = self.records.size
        self.augmentation = augmentation
        self.rgb = rgb
        self.size = size
        self.six_channel_augment = six_channel

    def _load_img_as_tensor(self, file_name, size):
        with Image.open(file_name) as img:
            img = T.Resize(size)(img)

            if not self.augmentation:
                # trans = T.Compose([
                #     T.FiveCrop(256),
                #     T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])),
                #
                # ])
                return T.ToTensor()(img)
                # return trans(img)
            else:
                transfrom = T.Compose([
                    # T.FiveCrop(256),
                    # T.Lambda(lambd=lambda crops: crops[random.randint(0, 4)]),
                    T.RandomRotation(90),
                    T.RandomHorizontalFlip(0.5),
                    T.RandomVerticalFlip(0.5),
                    trick.RandomErasing(),
                    T.ToTensor()
                ])
                return transfrom(img)

    def _get_img_path(self, index, channel, site):
        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
        return '/'.join([self.img_dir, self.mode, experiment, f'Plate{plate}', f'{well}_s{site}_w{channel}.png'])

    def six_channel_transform(self, arr, augment=False):
        if augment:
            aug = Compose([
                Resize(height=self.size, width=self.size),
                RandomRotate90(),
                Flip(),
                Transpose(),
                OneOf([
                    IAAAdditiveGaussianNoise(),
                    GaussNoise(),
                ], p=0.2),
                OneOf([
                    MotionBlur(p=.2),
                    MedianBlur(blur_limit=3, p=0.1),
                    Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                OneOf([
                    OpticalDistortion(p=0.3),
                    GridDistortion(p=.1),
                    IAAPiecewiseAffine(p=0.3),
                ], p=0.2),
                OneOf([
                    IAASharpen(),
                    IAAEmboss(),
                    RandomBrightnessContrast(),
                ], p=0.3),
            ], p=1)
        else:
            aug = Compose([
                Resize(height=self.size, width=self.size),
            ], p=1)
        ret = aug(image=arr)['image']
        return ret

    def _transform(self, img, augumentation):
        if augumentation:
            trans = torchvision.transforms.Compose([
                # torchvision.transforms.RandomCrop(224),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize()
                # torchvision.transforms.RandomGrayscale(p=0.2),
                # torchvision.transforms.RandomRotation(90),
                # torchvision.transforms.RandomHorizontalFlip(0.5),
                # torchvision.transforms.RandomVerticalFlip(0.5),

            ])
            img = trans(img)
        else:
            trans = torchvision.transforms.Compose([

                # torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize()

                # torchvision.transforms.RandomGrayscale(p=0.2),
                # torchvision.transforms.RandomRotation(90),
                # torchvision.transforms.RandomHorizontalFlip(0.5),
                # torchvision.transforms.RandomVerticalFlip(0.5),

            ])
            img = trans(img)
        return img

    def __getitem__(self, index):
        if not self.rgb:
            site = index // self.len + 1
            index = index % self.len
            # site = random.choice([1, 2])
            # site = 1
            # tensor_path = "tensor/" + str(self.records[index].id_code) + "_" + str(site) + '_' + str(self.size) + '.pt'
            # if not os.path.exists(tensor_path):
            #     paths = [self._get_img_path(index, ch, site) for ch in self.channels]
            #     img = torch.cat([self._load_img_as_tensor(img_path, self.size) for img_path in paths])
            #     torch.save(img, tensor_path)
            # else:
            #     img = torch.load(tensor_path)
            paths = [self._get_img_path(index, ch, site) for ch in self.channels]
            if not self.six_channel_augment:
                img = torch.cat([self._load_img_as_tensor(img_path, self.size) for img_path in paths])
            else:
                six_channel_img = np.array([np.array(Image.open(path)) for path in paths]).transpose([1, 2, 0])
                img = self.six_channel_transform(six_channel_img, self.augmentation)
                img = self._transform(img, True)

            if self.mode == 'train':
                return img, int(self.records[index].sirna)
            else:
                return img, self.records[index].id_code
        else:
            if random.choice([0, 1]):
                index = self.len + index
            filename = self.records[index % self.len].filename
            img = Image.open(os.path.join(self.img_dir, filename))
            img = self._transform(img, self.augmentation)

            if self.mode == 'train':
                return img, int(self.records[index].sirna)
            else:
                return img, self.records[index].id_code

    def __len__(self):
        return self.len * 2
