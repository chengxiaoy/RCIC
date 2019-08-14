import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as D
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, MultiStepLR

from torchvision import models, transforms as T
import torchvision
import time
import copy

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
import sys
from rcic_data import *
from model import get_basic_model, get_model
import joblib
from tensorboardX import SummaryWriter
from datetime import datetime
from evaluate import facade

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


class Config():
    train_batch_size = 24
    val_batch_size = 24

    device_ids = [2, 3]
    use_rgb = False
    backbone = 'densenet201'
    head_type = 'arcface'
    classes = 1108
    pic_size = 384

    def __repr__(self):
        return "batch_size_{}_picsize_{}_backbone_{}_head_{}_rgb_{}".format(
            self.train_batch_size, self.pic_size, self.backbone, self.head_type, self.use_rgb)


config = Config()

# model part
model = get_model(config.backbone, config.use_rgb, config.head_type)
model = model.to(device)
model = torch.nn.DataParallel(model, device_ids=[2, 3])

# data part
ds, ds_val, ds_test = get_dataset(config.use_rgb, size=config.pic_size)
loader = D.DataLoader(ds, batch_size=config.train_batch_size, shuffle=True, num_workers=16)
val_loader = D.DataLoader(ds_val, batch_size=config.val_batch_size, shuffle=True, num_workers=16)
tloader = D.DataLoader(ds_test, batch_size=config.val_batch_size, shuffle=False, num_workers=16)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = ExponentialLR(optimizer, gamma=0.95)

experiment_name = datetime.now().strftime('%b%d_%H-%M') + "_" + str(config)
writer = SummaryWriter(logdir=os.path.join("board/", experiment_name))


def train_model(model, criterion, optimizer, scheduler, dataloaders, writer, num_epochs):
    min_loss = float('inf')
    max_acc = 0.0

    for epoch in range(num_epochs):
        running_loss = 0.0

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                for i, (input, target) in enumerate(dataloaders[phase]):
                    input = input.to(device)
                    target = target.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(True):
                        embedding = model(input, target)
                        loss = criterion(embedding, target)
                        loss.backward()
                        optimizer.step()
                        running_loss = running_loss + loss.item()

                epoch_loss = running_loss / len(dataloaders[phase])
                writer.add_text('Text', '{} Loss: {:.4f} '.format(phase, epoch_loss),
                                epoch)
                print('{} Loss: {:.4f} '.format(phase, epoch_loss))
                scheduler.step()
            else:
                model.eval()
                embeddings = []
                labels = []
                for i, (input, target) in enumerate(dataloaders[phase]):
                    input = input.to(device)
                    target = target.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(False):
                        embedding = model(input, target)
                        embeddings.append(embedding)
                        labels.append(target)
                        loss = criterion(embedding, target)
                        running_loss = running_loss + loss.item()
                epoch_loss = running_loss / len(dataloaders[phase])
                writer.add_text('Text', '{} Loss: {:.4f} '.format(phase, epoch_loss),
                                epoch)
                print('{} Loss: {:.4f} '.format(phase, epoch_loss))
                embeddings = torch.cat(embeddings)
                labels = torch.cat(labels)
                accuracy, best_threshold, roc_curve_tensor = facade(embeddings, labels)
                board_val(writer, accuracy, best_threshold, roc_curve_tensor, epoch)

                if epoch_loss < min_loss:
                    min_loss = epoch_loss
                    torch.save(model.state_dict(), experiment_name + ".pth")


def board_val(writer, accuracy, best_threshold, roc_curve_tensor, step):
    writer.add_scalar('accuracy', accuracy, step)
    writer.add_scalar('best_threshold', best_threshold, step)
    writer.add_image('roc_curve', roc_curve_tensor, step)


train_model(model, criterion, optimizer, lr_scheduler, {'train': loader, 'val': val_loader}, writer, 50)
