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
from tqdm import tqdm

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, Resize,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, RandomCrop,
)
import math

rgb_train_csv_path = 'new_train.csv'
rgb_test_csv_path = 'new_test.csv'

pixel_csv = './data/pixel_stats.csv'

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

    pixel_stats = pd.read_csv(pixel_csv)
    pixel_info = {}
    for i in tqdm(range(len(pixel_stats))):
        experiment_ = pixel_stats.iloc[i]['experiment']
        plate = pixel_stats.iloc[i]['plate']
        well = pixel_stats.iloc[i]['well']
        site = pixel_stats.iloc[i]['site']
        channel = pixel_stats.iloc[i]['channel']
        pic_id = "_".join([experiment_, str(plate), str(well), str(site), str(channel)])
        pixel_info[pic_id] = [float(pixel_stats.iloc[i]['mean']), float(pixel_stats.iloc[i]['std'])]

    if experment != 'all':
        index = np.array([x.split('-')[0] for x in np.array(df_train.id_code)]) == experment
        df_train = df_train.iloc[index]

        index = np.array([x.split('-')[0] for x in np.array(df_val.id_code)]) == experment
        df_val = df_val.iloc[index]

        index = np.array([x.split('-')[0] for x in np.array(df_test.id_code)]) == experment
        df_test = df_test.iloc[index]

    ds = ImagesDS(df_train, img_dir, pixel_info, mode='train', augmentation=train_aug, size=size,
                  six_channel=six_channel)
    ds_val = ImagesDS(df_val, img_dir, pixel_info, mode='train', augmentation=val_aug, size=size,
                      six_channel=six_channel)
    ds_test = ImagesDS(df_test, img_dir, pixel_info, mode='test', augmentation=test_aug, size=size,
                       six_channel=six_channel)
    return ds, ds_val, ds_test


class ImagesDS(D.Dataset):
    def __init__(self, df, img_dir, pixel_state, mode='train', augmentation=False, size=512, six_channel=False, site=1,
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
        self.pixel_state = pixel_state

    def _load_img_as_tensor(self, file_name, size):

        infos = file_name.split("/")
        experiment = infos[-3]
        plate = infos[-2][5:]
        well = infos[-1].split("_")[0]
        site = infos[-1].split("_")[1][1:]
        channel = infos[-1].split("_")[2].split(".")[0][1:]
        pic_id = "_".join([experiment, plate, well, site, channel])

        mean, std = self.pixel_state[pic_id]

        with Image.open(file_name) as img:
            img = T.Resize(size)(img)

            if not self.augmentation:
                img = T.ToTensor()(img)
                return T.Normalize(mean / 255, std / 255)(img),

            else:
                transfrom = T.Compose([

                    T.RandomRotation(90),
                    T.RandomHorizontalFlip(0.5),
                    T.RandomVerticalFlip(0.5),
                    trick.RandomErasing(),
                    T.ToTensor(),
                    T.Normalize(mean/255, std/255)
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

    def six_channel_transform(self, arr):

        aug = Compose([
            # Resize(height=self.size, width=self.size),
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

        ret = aug(image=arr)['image']
        return ret

    def __getitem__(self, index):

        site = index // self.len + 1
        index = index % self.len
        paths = [self._get_img_path(index, ch, site) for ch in self.channels]
        if not self.six_channel_augment:
            img = torch.cat([self._load_img_as_tensor(img_path, self.size) for img_path in paths])
        else:
            six_channel_img = np.array([np.array(T.Resize(self.size)(Image.open(path))) for path in paths]).transpose(
                [1, 2, 0])
            if self.augmentation:
                six_channel_img = trick.RandomErasing()(six_channel_img)
                six_channel_img = self.six_channel_transform(six_channel_img)
            img = T.ToTensor()(six_channel_img)

        if self.mode == 'train':
            return img, int(self.records[index].sirna)
        else:
            return img, self.records[index].id_code

    def __len__(self):
        return self.len * 2
