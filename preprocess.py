import os
import sys
import zipfile

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

sys.path.append('rxrx1-utils')
import rxrx.io as rio

from multiprocessing import Pool

from collections import namedtuple

Cell = namedtuple('Cell', ['code', 'split', 'experiment', 'plate', 'well', 'site', 'extension'])

for folder in ['train', 'test']:
    os.makedirs(folder)

train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
print(train_df.shape)
print(test_df.shape)


def convert_to_rgb(df, split, resize=True, new_size=224, extension='jpeg'):
    N = df.shape[0]
    cc_list = []
    for i in tqdm(range(N)):
        code = df['id_code'][i]
        experiment = df['experiment'][i]
        plate = df['plate'][i]
        well = df['well'][i]
        for site in [1, 2]:
            cc = Cell(split, code, experiment, plate, well, site, extension)
            cc_list.append(cc)

    p = Pool(16)
    p.map(convert_to_rgb_, cc_list)
    p.close()


def convert_to_rgb_(cell):
    split = cell.split
    code = cell.code
    site = cell.site
    extension = cell.extension
    experiment = cell.experiment
    plate = cell.plate
    well = cell.well

    save_path = f'{split}/{code}_s{site}.{extension}'

    im = rio.load_site_as_rgb(
        split, experiment, plate, well, site,
        base_path='./data/'
    )
    im = im.astype(np.uint8)
    im = Image.fromarray(im)

    im = im.resize((224, 224), resample=Image.BILINEAR)

    im.save(save_path)


convert_to_rgb(train_df, 'train')
convert_to_rgb(test_df, 'test')


def zip_and_remove(path):
    ziph = zipfile.ZipFile(f'{path}.zip', 'w', zipfile.ZIP_DEFLATED)

    for root, dirs, files in os.walk(path):
        for file in tqdm(files):
            file_path = os.path.join(root, file)
            ziph.write(file_path)
            os.remove(file_path)

    ziph.close()


zip_and_remove('train')
zip_and_remove('test')


def build_new_df(df, extension='jpeg'):
    new_df = pd.concat([df, df])
    new_df['filename'] = pd.concat([
        df['id_code'].apply(lambda string: string + f'_s1.{extension}'),
        df['id_code'].apply(lambda string: string + f'_s2.{extension}')
    ])

    return new_df


new_train = build_new_df(train_df)
new_test = build_new_df(test_df)

new_train.to_csv('new_train.csv', index=False)
new_test.to_csv('new_test.csv', index=False)
