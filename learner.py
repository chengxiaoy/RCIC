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

model.load_state_dict(torch.load('models/Aug14_15-43_batch_size_24_picsize_384_backbone_densenet201_head_arcface_rgb_False.pth'))
# model = model.module

# data part
ds, ds_val, ds_test = get_dataset(config.use_rgb, size=config.pic_size)
loader = D.DataLoader(ds, batch_size=config.train_batch_size, shuffle=True, num_workers=16)
val_loader = D.DataLoader(ds_val, batch_size=config.val_batch_size, shuffle=False, num_workers=16)
tloader = D.DataLoader(ds_test, batch_size=config.val_batch_size, shuffle=False, num_workers=16)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# lr_scheduler = ExponentialLR(optimizer, gamma=0.95)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

experiment_name = datetime.now().strftime('%b%d_%H-%M') + "_" + str(config)
writer = SummaryWriter(logdir=os.path.join("board/", experiment_name))


def train_model(model, criterion, optimizer, scheduler, dataloaders, writer, num_epochs):
    min_loss = float('inf')
    max_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())


    for epoch in range(num_epochs):
        running_loss = 0.0

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train','val']:
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
                writer.add_scalar('train_loss', epoch_loss, epoch)
                writer.add_text('Text', '{} Loss: {:.4f} '.format(phase, epoch_loss),
                                epoch)
                print('{} Loss: {:.4f} '.format(phase, epoch_loss))

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
                writer.add_scalar('val/loss', epoch_loss, epoch)
                writer.add_text('Text', '{} Loss: {:.4f} '.format(phase, epoch_loss),
                                epoch)
                print('{} Loss: {:.4f} '.format(phase, epoch_loss))
                embeddings = torch.cat(embeddings)
                labels = torch.cat(labels)
                accuracy, best_threshold, roc_curve_tensor = facade(embeddings, labels)
                print('{} acc: {:.4f} '.format(phase, accuracy))
                print('{} thr: {:.4f} '.format(phase, best_threshold))


                board_val(writer, accuracy, best_threshold, roc_curve_tensor, epoch)


                if accuracy>max_acc:
                    max_acc = accuracy
                    torch.save(model.state_dict(), 'models/' + experiment_name + ".pth")
                    best_model_wts = copy.deepcopy(model.state_dict())



                # if epoch_loss < min_loss:
                #     min_loss = epoch_loss
                #     torch.save(model.state_dict(), 'models/' + experiment_name + ".pth")

                scheduler.step(accuracy)

    model.load_state_dict(best_model_wts)


def board_val(writer, accuracy, best_threshold, roc_curve_tensor, step):
    writer.add_scalar('val/accuracy', accuracy, step)
    writer.add_scalar('best_threshold', best_threshold, step)
    writer.add_image('roc_curve', roc_curve_tensor, step)


# train_model(model, criterion, optimizer, lr_scheduler, {'train': loader, 'val': val_loader}, writer, 50)

train_embeddings = []
train_labels = []

model.eval()
with torch.no_grad():
    for i, (input, target) in enumerate(loader):
        input = input.to(device)
        target = target.to(device)
        embedding = model(input, target).cpu().numpy()
        train_embeddings.append(embedding)
        train_labels.append(target.cpu().numpy())

    for i, (input, target) in enumerate(val_loader):
        input = input.to(device)
        target = target.to(device)
        embedding = model(input, target).cpu().numpy()
        train_embeddings.append(embedding)
        train_labels.append(target.cpu().numpy())

    train_embeddings = np.concatenate(train_embeddings)
    train_labels = np.concatenate(train_labels)


    train_labels = np.array(train_labels)
    train_embeddings = np.array(train_embeddings)

    center_features = []
    for i in range(1108):
        index = train_labels == i
        center_feature = np.mean(train_embeddings[index], axis=0)
        center_features.append(center_feature)

    center_features = np.array(center_features)

    test_embeddings = []
    for i, (input, target) in enumerate(tloader):
        nput = input.to(device)
        target = target.to(device)
        embedding = model(input, target).cpu().numpy()
        test_embeddings.append(embedding)
    test_embeddings = np.concatenate(test_embeddings)

    assert len(test_embeddings) == 19897 * 2
    from sklearn.metrics.pairwise import cosine_similarity

    similarity = cosine_similarity(test_embeddings, center_features)
    confi = similarity.max(axis=1)
    preds = similarity.argmax(axis=1)

    true_idx = np.empty(0)
    for i in range(19897):
        if confi[i] > confi[i + 19897]:
            true_idx = np.append(true_idx, preds[i])
        else:
            true_idx = np.append(true_idx, preds[i + 19897])

submission = pd.read_csv('data/test.csv')
submission['sirna'] = true_idx.astype(int)
submission.to_csv('submission.csv', index=False, columns=['id_code', 'sirna'])
