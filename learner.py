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
from loss.advance_loss import ArcFaceLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Config():
    train_batch_size = 64
    val_batch_size = 64
    test_batch_size = 64

    device_ids = [0, 1, 2, 3]
    use_rgb = False
    backbone = 'resnet_50'
    head_type = 'arcface'
    classes = 1108
    pic_size = 448

    stage1_epoch = 30
    stage2_epoch = 30

    stage1_lr = 0.0001
    stage2_lr = 0.0001

    def __repr__(self):
        return "lr1_{}_lr2_{}_bs_{}_ps_{}_backbone_{}_head_{}_rgb_{}".format(self.stage1_lr,
                                                                             self.stage2_lr,
                                                                             self.train_batch_size,
                                                                             self.pic_size,
                                                                             self.backbone, self.head_type,
                                                                             self.use_rgb)


class Learner:
    def __init__(self, config):
        self.config = config
        self.experiment_name = datetime.now().strftime('%b%d_%H-%M') + "-" + str(config)
        self.experiment_time = datetime.now().strftime('%b%d_%H-%M')

    def build_model(self, weight_path=None):
        model = get_model(self.config.backbone, self.config.use_rgb, 'line')
        model = model.to(device)
        model = torch.nn.DataParallel(model, device_ids=self.config.device_ids)
        if weight_path is not None:
            model.load_state_dict(
                torch.load(weight_path))
        return model

    def stage_one(self):
        model = self.build_model(
            weight_path='models/stage1_Aug22_06-57-lr1_0.0001_lr2_0.0001_bs_32_ps_448_backbone_resnet_50_head_arcface_rgb_False.pth')

        ds, ds_val, ds_test = get_dataset(self.config.use_rgb, size=self.config.pic_size, pair=False)
        loader = D.DataLoader(ds, batch_size=self.config.train_batch_size, shuffle=True, num_workers=16)
        val_loader = D.DataLoader(ds_val, batch_size=self.config.val_batch_size, shuffle=False, num_workers=16)

        criterion = nn.CrossEntropyLoss()
        # criterion = trick.LabelSmoothing(1108, 0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.stage1_lr)

        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

        writer = SummaryWriter(logdir=os.path.join("board", "stage1_" + self.experiment_name))
        s1_pretrained_model = train_model(model, criterion, optimizer, lr_scheduler,
                                          {'train': loader, 'val': val_loader}, writer,
                                          self.config.stage1_epoch, "stage1_" + self.experiment_name, self.config)

        return s1_pretrained_model

    def stage_two(self, s1_pretrained_model):

        model = s1_pretrained_model.module

        model.set_head_type('arcface')
        model = torch.nn.DataParallel(model, device_ids=self.config.device_ids)

        ds, ds_val, ds_test = get_dataset(self.config.use_rgb, size=self.config.pic_size, pair=True)
        loader = D.DataLoader(ds, batch_size=self.config.train_batch_size, shuffle=True, num_workers=16, drop_last=True)
        val_loader = D.DataLoader(ds_val, batch_size=self.config.val_batch_size, shuffle=False, num_workers=16,
                                  drop_last=True)

        # criterion = nn.CrossEntropyLoss()
        criterion = ArcFaceLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.stage2_lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

        writer = SummaryWriter(logdir=os.path.join("board", "stage2_" + self.experiment_name))
        s2_model = train_model_s2(model, criterion, optimizer, lr_scheduler,
                                  {'train': loader, 'val': val_loader}, writer,
                                  self.config.stage1_epoch, "stage2_" + self.experiment_name, self.config)

        return s2_model

    def angle_evaluate(self, model):
        train_embeddings = []
        train_labels = []
        ds, ds_val, ds_test = get_dataset(self.config.use_rgb, size=self.config.pic_size, pair=False)
        loader = D.DataLoader(ds, batch_size=self.config.train_batch_size, shuffle=True, num_workers=16)
        val_loader = D.DataLoader(ds_val, batch_size=self.config.val_batch_size, shuffle=False, num_workers=16)
        tloader = D.DataLoader(ds_test, batch_size=self.config.val_batch_size, shuffle=False, num_workers=16)

        model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(loader):
                input = input.to(device)
                target = target.to(device)
                embedding, cos = model(input, target).cpu().numpy()
                train_embeddings.append(embedding)
                train_labels.append(target.cpu().numpy())

            for i, (input, target) in enumerate(val_loader):
                input = input.to(device)
                target = target.to(device)
                embedding, cos = model(input, target).cpu().numpy()
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

            joblib.dump(train_embeddings, "train_embeddings.pkl")
            joblib.dump(train_labels, 'train_labels.pkl')
            joblib.dump(center_features, 'center_features.pkl')

            test_embeddings = []
            for i, (input, target) in enumerate(tloader):
                input = input.to(device)
                # target = target.to(device)
                embedding, cos = model(input, target).cpu().numpy()
                test_embeddings.append(embedding)
            test_embeddings = np.concatenate(test_embeddings)

            joblib.dump(test_embeddings, 'test_embeddings.pkl')

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
        submission.to_csv('submission_angle.csv', index=False, columns=['id_code', 'sirna'])

    def confi_evaluate(self, model, avg=False):

        model.eval()
        ds, ds_val, ds_test = get_dataset(self.config.use_rgb, size=self.config.pic_size, pair=False)

        tloader = D.DataLoader(ds_test, batch_size=self.config.test_batch_size, shuffle=False, num_workers=16)

        preds = np.empty(0)
        confi = np.empty(0)
        with torch.no_grad():
            for x, _ in tqdm(tloader):
                x = x.to(device)
                if len(x.size()) == 5:
                    bs, ncrops, c, h, w = x.size()
                    output = model(x.view(-1, c, h, w), _)  # fuse batch size and ncrops
                    if avg:
                        output_avg = output.view(bs, ncrops, -1).mean(1)  # avg over crops
                        idx = output_avg.max(dim=-1)[1].cpu().numpy()
                        confidence = output_avg.max(dim=-1)[0].cpu().numpy()
                        preds = np.append(preds, idx, axis=0)
                        confi = np.append(confi, confidence, axis=0)
                    else:
                        idx = output.max(dim=-1)[1].cpu().numpy()
                        confidence = output.max(dim=-1)[0].cpu().numpy()

                        idx = idx.reshape(-1, 5)
                        confidence = confidence.reshape(-1, 5)
                        confi_max_idx = confidence.argmax(axis=1)
                        confi_max = confidence.max(axis=1)
                        idx_max = idx[range(len(idx)), confi_max_idx]
                        print(idx_max)
                        preds = np.append(preds, idx_max, axis=0)
                        confi = np.append(confi, confi_max, axis=0)
                else:
                    output = model(x, _)
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


def train_model(model, criterion, optimizer, scheduler, dataloaders, writer, num_epochs, name, config):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    min_loss = float('inf')
    max_acc = 0.0

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        lr = float(optimizer.param_groups[0]['lr'])
        print("Learning rate: {}".format(lr))

        epoch_loss = {}
        epoch_acc = {}
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

            for i, (input, target) in enumerate(dataloaders[phase]):
                # if phase == 'train':
                input = input.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    theta = model(input, target)
                    loss = criterion(theta, target)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    running_loss = running_loss + loss.item()
                    label = torch.max(theta.data, 1)[1]
                    for i, j in zip(label, target.data.cpu().numpy()):
                        if len(label.shape) == 1:
                            if i == j:
                                running_corrects += 1
                        else:
                            if i[0] == j[0]:
                                running_corrects += 1
                # else:
                #     input = input.to(device)
                #     target = target.to(device)
                #     optimizer.zero_grad()
                #     bs, ncrops, c, h, w = input.size()
                #     result = model(input.view(-1, c, h, w), target)  # fuse batch size and ncrops
                #     result_avg = result.view(bs, ncrops, -1).mean(1)  # avg over crops
                #
                #     with torch.set_grad_enabled(phase == 'train'):
                #         loss = criterion(result_avg, target)
                #         if phase == 'train':
                #             loss.backward()
                #             optimizer.step()
                #         running_loss = running_loss + loss.item()
                #         label = torch.max(result_avg.data, 1)[1]
                #         for i, j in zip(label, target.data.cpu().numpy()):
                #             if len(label.shape) == 1:
                #                 if i == j:
                #                     running_corrects += 1
                #             else:
                #                 if i[0] == j[0]:
                #                     running_corrects += 1

            epoch_loss[phase] = running_loss / len(dataloaders[phase])
            epoch_acc[phase] = running_corrects / (len(dataloaders[phase]) * config.train_batch_size)

            writer.add_text('Text', '{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss[phase], epoch_acc[phase]),
                            epoch)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss[phase], epoch_acc[phase]))

            if phase == 'val' and epoch_acc[phase] > max_acc:
                max_acc = epoch_acc[phase]
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "models/" + name + ".pth")

        writer.add_scalars('data/loss', {'train': epoch_loss['train'], 'val': epoch_loss['val']}, epoch)
        writer.add_scalars('data/acc', {'train': epoch_acc['train'], 'val': epoch_acc['val']}, epoch)
        scheduler.step(epoch_acc['val'])

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('min loss : {:4f}'.format(min_loss))
    print('max acc : {:4f}'.format(max_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


def train_model_s2(model, criterion, optimizer, scheduler, dataloaders, writer, num_epochs, name, config):
    min_loss = float('inf')
    max_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0

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
                        embedding, cos = model(input, target)
                        loss = criterion(cos, target)
                        loss.backward()
                        optimizer.step()
                        running_loss = running_loss + loss.item()
                        label = torch.max(cos.data, 1)[1]

                        for i, j in zip(label, target.data.cpu().numpy()):
                            if len(label.shape) == 1:
                                if i == j:
                                    running_corrects += 1
                            else:
                                if i[0] == j[0]:
                                    running_corrects += 1

                epoch_loss = running_loss / len(dataloaders[phase])
                writer.add_scalar('train/loss', epoch_loss, epoch)
                epoch_acc = running_corrects / (
                        len(dataloaders[phase]) * config.train_batch_size)
                writer.add_scalar('train/acc', epoch_acc, epoch)
                writer.add_text('Text', '{} Loss: {:.4f} '.format(phase, epoch_loss),
                                epoch)
                print('{} Loss: {:.4f} '.format(phase, epoch_loss))
                print('{} theta Acc: {:.4f}'.format(phase, epoch_acc))

            else:
                running_corrects = 0
                model.eval()
                embeddings = []
                labels = []
                for i, (input, target) in enumerate(dataloaders[phase]):
                    input = input.to(device)
                    target = target.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(False):
                        embedding, cos = model(input, target)
                        embeddings.append(embedding)
                        labels.append(target)
                        loss = criterion(cos, target)
                        running_loss = running_loss + loss.item()
                        label = torch.max(cos.data, 1)[1]

                        for i, j in zip(label, target.data.cpu().numpy()):
                            if len(label.shape) == 1:
                                if i == j:
                                    running_corrects += 1
                            else:
                                if i[0] == j[0]:
                                    running_corrects += 1
                epoch_loss = running_loss / len(dataloaders[phase])
                writer.add_scalar('val/loss', epoch_loss, epoch)
                epoch_acc = running_corrects / (
                        len(dataloaders[phase]) * config.train_batch_size)
                writer.add_scalar('val/acc', epoch_acc, epoch)
                writer.add_text('Text', '{} Loss: {:.4f} '.format(phase, epoch_loss),
                                epoch)
                print('{} Loss: {:.4f} '.format(phase, epoch_loss))
                print('{} theta Acc: {:.4f}'.format(phase, epoch_acc))

                embeddings = torch.cat(embeddings)
                labels = torch.cat(labels)
                accuracy, best_threshold, roc_curve_tensor = facade(embeddings, labels)
                print('{} metric_acc: {:.4f} '.format(phase, accuracy))
                print('{} thr: {:.4f} '.format(phase, best_threshold))

                board_val(writer, accuracy, best_threshold, roc_curve_tensor, epoch)

                if accuracy > max_acc:
                    max_acc = accuracy
                    torch.save(model.state_dict(), 'models/' + name + ".pth")
                    best_model_wts = copy.deepcopy(model.state_dict())
                scheduler.step(accuracy)

    model.load_state_dict(best_model_wts)
    return model


def board_val(writer, accuracy, best_threshold, roc_curve_tensor, step):
    writer.add_scalar('val/metric_acc', accuracy, step)
    writer.add_scalar('best_threshold', best_threshold, step)
    writer.add_image('roc_curve', roc_curve_tensor, step)


if __name__ == "__main__":
    config = Config()

    learner = Learner(config)
    # s1_model = learner.stage_one()
    # learner.confi_evaluate(s1_model)
    s1_model = learner.build_model(
        weight_path='models/stage1_Aug23_09-17-lr1_0.0001_lr2_0.0001_bs_32_ps_448_backbone_resnet_50_head_arcface_rgb_False.pth')
    # learner.confi_evaluate(s1_model)

    s2_model = learner.stage_two(s1_model)
    learner.angle_evaluate(s2_model)
