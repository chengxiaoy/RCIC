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

rgb_train_csv_path = 'new_train.csv'
rgb_test_csv_path = 'new_test.csv'

train_csv_path = './data/train.csv'
test_csv_path = './data/test.csv'
img_dir = './data'


def get_dataset(rgb=True, size=512):
    if rgb:
        rgb_df = pd.read_csv(rgb_train_csv_path)
        df_train, df_val = train_test_split(rgb_df, test_size=0.1, stratify=rgb_df.sirna, random_state=42)
        df_test = pd.read_csv(rgb_test_csv_path)

        ds = ImagesDS(df_train, 'train', True, mode='train', augmentation=True, size=size)
        ds_val = ImagesDS(df_val, 'train', True, mode='train', size=size)
        ds_test = ImagesDS(df_test, 'test', True, mode='test', size=size)

        return ds, ds_val, ds_test
    else:
        rgb_df = pd.read_csv(train_csv_path)
        df_train, df_val = train_test_split(rgb_df, test_size=0.05, stratify=rgb_df.sirna, random_state=42)
        df_test = pd.read_csv(test_csv_path)
        ds = ImagesDS(df_train, img_dir, False, mode='train', augmentation=True, size=size)
        ds_val = ImagesDS(df_val, img_dir, False, mode='train', size=size)
        ds_test = ImagesDS(df_test, img_dir, False, mode='test', size=size)
        return ds, ds_val, ds_test


class ImagesDS(D.Dataset):
    def __init__(self, df, img_dir, rgb, mode='train', augmentation=False, size=512, site=1,
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

    @staticmethod
    def _load_img_as_tensor(file_name, size):
        with Image.open(file_name) as img:
            img = T.CenterCrop(size)(img)
            return T.ToTensor()(img)

    def _get_img_path(self, index, channel, site):
        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
        return '/'.join([self.img_dir, self.mode, experiment, f'Plate{plate}', f'{well}_s{site}_w{channel}.png'])

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
            img = torch.cat([self._load_img_as_tensor(img_path, self.size) for img_path in paths])

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
