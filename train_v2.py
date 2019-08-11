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
from model import get_basic_model
import joblib
from tensorboardX import SummaryWriter
from datetime import datetime

# sys.path.append('rxrx1-utils')
# import rxrx.io as rio

warnings.filterwarnings('ignore')

path_data = 'data'
# device = 'cuda'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 48
torch.manual_seed(0)
use_rgb = False
model_name = 'densenet201'
classes = 1108
pic_size = 512
experiment_name = str(use_rgb) + "_" + str(batch_size) + "_" + str(
    pic_size) + "_" + model_name + "_" + datetime.now().strftime('%b%d_%H-%M')

ds, ds_val, ds_test = get_dataset(use_rgb, size=pic_size)

model = get_basic_model(model_name, use_rgb)
model = model.to(device)

model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

# model.load_state_dict(torch.load('models/Model_False_24_512_densenet201_Aug07_13-01_35_val_acc=0.526287.pth'))
# model = model.module

loader = D.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=16)
val_loader = D.DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=16)
tloader = D.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=16)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
# lr_scheduler = ExponentialLR(optimizer, gamma=0.95)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)


# lr_scheduler = MultiStepLR(optimizer, [15, 25, 100], 0.1)

def train_model(model, criterion, optimizer, scheduler, dataloaders, writer, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    min_loss = float('inf')
    max_acc = 0.0

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        epoch_loss = {}
        epoch_acc = {}
        epoch_true_negative = {}
        epoch_false_positive = {}
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
                # model.apply(set_batchnorm_eval)
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            true_negative = 0
            false_positive = 0

            for i, (input, target) in enumerate(dataloaders[phase]):

                input = input.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):

                    theta, theta_ = model(input, target)
                    loss = criterion(theta, target)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    running_loss = running_loss + loss.item()
                    label = torch.max(theta_.data, 1)[1]

                    for i, j in zip(label, target.data.cpu().numpy()):
                        if len(label.shape) == 1:
                            if i == j:
                                running_corrects += 1
                            elif i:
                                true_negative += 1
                            else:
                                false_positive += 1
                        else:
                            if i[0] == j[0]:
                                running_corrects += 1
                            elif i[0]:
                                true_negative += 1
                            else:
                                false_positive += 1

            epoch_true_negative[phase] = true_negative / (len(dataloaders[phase]) * batch_size)
            epoch_false_positive[phase] = false_positive / (len(dataloaders[phase]) * batch_size)
            epoch_loss[phase] = running_loss / len(dataloaders[phase])
            epoch_acc[phase] = running_corrects / (len(dataloaders[phase]) * batch_size)

            writer.add_text('Text', '{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss[phase], epoch_acc[phase]),
                            epoch)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss[phase], epoch_acc[phase]))
            print('{} true_negative:{:.4f} false_positive:{:.4f}'.format(phase, epoch_true_negative[phase],
                                                                         epoch_false_positive[phase]))

            # deep copy the model
            if phase == 'val' and epoch_loss[phase] < min_loss:
                min_loss = epoch_loss[phase]
                torch.save(model.state_dict(), str(model) + "loss.pth")

            if phase == 'val' and epoch_acc[phase] > max_acc:
                max_acc = epoch_acc[phase]
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), str(model) + ".pth")
        writer.add_scalars('data/true_negative',
                           {'train': epoch_true_negative['train'], 'val': epoch_true_negative['val']}, epoch)
        writer.add_scalars('data/false_positive',
                           {'train': epoch_false_positive['train'], 'val': epoch_false_positive['val']}, epoch)

        writer.add_scalars('data/loss', {'train': epoch_loss['train'], 'val': epoch_loss['val']}, epoch)
        writer.add_scalars('data/acc', {'train': epoch_acc['train'], 'val': epoch_acc['val']}, epoch)
        # writer.add_scalar('data/loss', scheduler.get_lr())
        scheduler.step(epoch_acc['val'])

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('min loss : {:4f}'.format(min_loss))
    print('max acc : {:4f}'.format(max_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)

    # save model
    torch.save(model.state_dict(), str(model) + ".pth")

    return max_acc


writer = SummaryWriter(logdir=os.path.join("board/", experiment_name))

train_model(model, criterion, optimizer, lr_scheduler, {'train': loader, 'val': val_loader}, writer, 50)

model.eval()

preds = np.empty(0)
confi = np.empty(0)
with torch.no_grad():
    for x, _ in tqdm(tloader):
        x = x.to(device)
        output = model(x)
        idx = output.max(dim=-1)[1].cpu().numpy()
        confidence = output.max(dim=-1)[0].cpu().numpy()
        preds = np.append(preds, idx, axis=0)
        confi = np.append(confi, confidence, axis=0)

joblib.dump([confi, preds], "res.pkl")

true_idx = np.empty(0)
for i in range(19897):
    if confi[i] > confi[i + 19897]:
        true_idx = np.append(true_idx, preds[i])
    else:
        true_idx = np.append(true_idx, preds[i + 19897])

submission = pd.read_csv('data/test.csv')
submission['sirna'] = true_idx.astype(int)
submission.to_csv('submission.csv', index=False, columns=['id_code', 'sirna'])
