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
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, RandomCrop,
)
import math

rgb_train_csv_path = 'new_train.csv'
rgb_test_csv_path = 'new_test.csv'

train_csv_path = './data/train.csv'
test_csv_path = './data/test.csv'
img_dir = './data'


def get_dataset(size=512, six_channel=False, train_aug=True, val_aug=False, test_aug=False,
                experment='all'):
    """
    :param size:
    :param pair:
    :param six_channel:
    :param train_aug:
    :param val_aug:
    :param test_aug:
    :param experment: one of the five value['HEPG2', 'HUVEC', 'RPE', 'U2OS','all']
    :return:
    """

    rgb_df = pd.read_csv(train_csv_path)
    df_train, df_val = train_test_split(rgb_df, test_size=0.12, stratify=rgb_df.sirna, random_state=42)
    df_test = pd.read_csv(test_csv_path)

    if experment != 'all':
        index = np.array([x.split('-')[0] for x in np.array(df_train.id_code)]) == experment
        df_train = df_train.iloc[index]

        index = np.array([x.split('-')[0] for x in np.array(df_val.id_code)]) == experment
        df_val = df_val.iloc[index]

    ds = ImagesDS(df_train, img_dir,  mode='train', augmentation=train_aug, size=size,
                  six_channel=six_channel)
    ds_val = ImagesDS(df_val, img_dir, mode='train', augmentation=val_aug, size=size,
                      six_channel=six_channel)
    ds_test = ImagesDS(df_test, img_dir, mode='test', augmentation=test_aug, size=size,
                       six_channel=six_channel)
    return ds, ds_val, ds_test


class ImagesDS(D.Dataset):
    def __init__(self, df, img_dir, mode='train', augmentation=False, size=512, six_channel=False, site=1,
                 channels=[1, 2, 3, 4, 5, 6]):
        self.records = df.to_records(index=False)
        self.channels = channels
        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        self.len = self.records.size
        self.augmentation = augmentation
        self.size = size
        self.six_channel_augment = six_channel

    def _load_img_as_tensor(self, file_name, size):
        with Image.open(file_name) as img:
            img = T.Resize(size)(img)

            if not self.augmentation:
                return T.ToTensor()(img)
            else:
                transfrom = T.Compose([
                    trick.RandomErasing(),
                    T.RandomRotation(90),
                    T.RandomHorizontalFlip(0.5),
                    T.RandomVerticalFlip(0.5),
                    T.ToTensor()
                ])

                # aug = Compose([
                #     # Resize(height=self.size, width=self.size),
                #     RandomRotate90(),
                #     Flip(),
                #     Transpose(),
                #     OneOf([
                #         IAAAdditiveGaussianNoise(),
                #         GaussNoise(),
                #     ], p=0.2),
                #     OneOf([
                #         MotionBlur(p=.2),
                #         MedianBlur(blur_limit=3, p=0.1),
                #         Blur(blur_limit=3, p=0.1),
                #     ], p=0.2),
                #     ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                #     OneOf([
                #         OpticalDistortion(p=0.3),
                #         GridDistortion(p=.1),
                #         IAAPiecewiseAffine(p=0.3),
                #     ], p=0.2),
                #     OneOf([
                #         IAASharpen(),
                #         IAAEmboss(),
                #         RandomBrightnessContrast(),
                #     ], p=0.3),
                # ], p=1)
                #
                # img = aug(image=np.array(img))['image']
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

    def __getitem__(self, index):

        site = index // self.len + 1
        index = index % self.len
        paths = [self._get_img_path(index, ch, site) for ch in self.channels]
        if not self.six_channel_augment:
            img = torch.cat([self._load_img_as_tensor(img_path, self.size) for img_path in paths])
        else:
            six_channel_img = np.array([np.array(Image.open(path)) for path in paths]).transpose([1, 2, 0])
            if self.augmentation:
                six_channel_img = trick.RandomErasing()(six_channel_img)
            img = self.six_channel_transform(six_channel_img, self.augmentation)
            img = T.ToTensor()(img)

        if self.mode == 'train':
            return img, int(self.records[index].sirna)
        else:
            return img, self.records[index].id_code

    def __len__(self):
        return self.len * 2
