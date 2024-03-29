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


# ['HEPG2', 'HUVEC', 'RPE', 'U2OS','all']
class Config():
    train_batch_size = 32
    val_batch_size = 32
    test_batch_size = 32

    device_ids = [0, 1]
    use_rgb = False
    backbone = 'resnet_50'
    head_type = 'arcface'
    classes = 1108
    pic_size = 512

    stage1_epoch = 50
    stage2_epoch = 50

    stage1_lr = 0.0003
    stage2_lr = 0.0001
    six_channel_aug = False
    experment = 'all'

    def __repr__(self):
        return "lr1_{}_lr2_{}_bs_{}_ps_{}_backbone_{}_head_{}_six_channel_aug_{}_experment_{}".format(
            self.stage1_lr,
            self.stage2_lr,
            self.train_batch_size,
            self.pic_size,
            self.backbone,
            self.head_type,
            self.six_channel_aug,
            self.experment)


class Learner:
    def __init__(self, config):
        self.config = config
        self.experiment_name = datetime.now().strftime('%b%d_%H-%M') + "-" + str(config)
        self.experiment_time = datetime.now().strftime('%b%d_%H-%M')

    def build_model(self, weight_path=None, mode='line'):
        model = get_model(self.config.backbone, self.config.use_rgb, mode)
        model = model.to(device)
        model = torch.nn.DataParallel(model, device_ids=self.config.device_ids)
        if weight_path is not None:
            model.load_state_dict(
                torch.load(weight_path))
        return model

    def stage_one(self):
        model = self.build_model(weight_path='models/stage1_Sep02_02-39-lr1_0.0001_lr2_0.0001_bs_32_ps_448_backbone_resnet_50_head_arcface_rgb_False_six_channel_aug_False.pth')
        ds, ds_val, ds_test = get_dataset(size=self.config.pic_size,
                                          six_channel=self.config.six_channel_aug, experment=self.config.experment)
        loader = D.DataLoader(ds, batch_size=self.config.train_batch_size, shuffle=True, num_workers=16)
        val_loader = D.DataLoader(ds_val, batch_size=self.config.val_batch_size, shuffle=False, num_workers=16)

        criterion = nn.CrossEntropyLoss()
        # criterion = trick.LabelSmoothing(1108, 0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.stage1_lr)

        # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=4, verbose=True)
        lr_scheduler = MultiStepLR(optimizer, [20, 30], 0.1)

        writer = SummaryWriter(logdir=os.path.join("board", "stage1_" + self.experiment_name))
        s1_pretrained_model = train_model(model, criterion, optimizer, lr_scheduler,
                                          {'train': loader, 'val': val_loader}, writer,
                                          self.config.stage1_epoch, "stage1_" + self.experiment_name, self.config)

        return s1_pretrained_model

    def stage_two(self, s1_pretrained_model):

        model = s1_pretrained_model.module

        model.set_head_type('arcface')
        model = torch.nn.DataParallel(model, device_ids=self.config.device_ids)

        ds, ds_val, ds_test = get_dataset(size=self.config.pic_size,
                                          six_channel=self.config.six_channel_aug, experment=self.config.experment)
        loader = D.DataLoader(ds, batch_size=self.config.train_batch_size, shuffle=True, num_workers=16, drop_last=True)
        val_loader = D.DataLoader(ds_val, batch_size=self.config.val_batch_size, shuffle=False, num_workers=16,
                                  drop_last=True)

        # criterion = nn.CrossEntropyLoss()
        criterion = ArcFaceLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.stage2_lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=4, verbose=True)
        # lr_scheduler = MultiStepLR(optimizer, [20, 30], 0.1)

        writer = SummaryWriter(logdir=os.path.join("board", "stage2_" + self.experiment_name))
        s2_model = train_model_s2(model, criterion, optimizer, lr_scheduler,
                                  {'train': loader, 'val': val_loader}, writer,
                                  self.config.stage2_epoch, "stage2_" + self.experiment_name, self.config)

        return s2_model

    def angle_evaluate(self, model):
        train_embeddings = []
        train_labels = []
        ds, ds_val, ds_test = get_dataset(size=self.config.pic_size,
                                          six_channel=self.config.six_channel_aug, train_aug=False, val_aug=False,
                                          test_aug=False)
        loader = D.DataLoader(ds, batch_size=self.config.test_batch_size, shuffle=True, num_workers=16)
        val_loader = D.DataLoader(ds_val, batch_size=self.config.test_batch_size, shuffle=False, num_workers=16)
        tloader = D.DataLoader(ds_test, batch_size=self.config.test_batch_size, shuffle=False, num_workers=16)

        model.eval()
        with torch.no_grad():
            # for i, (input, target) in tqdm(enumerate(loader)):
            #     input = input.to(device)
            #     target = target.to(device)
            #     embedding, cos = model(input, target)
            #     embedding = embedding.cpu().numpy()
            #     cos = cos.cpu().numpy()
            #     train_embeddings.append(embedding)
            #     train_labels.append(target.cpu().numpy())
            #
            # for i, (input, target) in tqdm(enumerate(val_loader)):
            #     input = input.to(device)
            #     target = target.to(device)
            #     embedding, cos = model(input, target)
            #     embedding = embedding.cpu().numpy()
            #     cos = cos.cpu().numpy()
            #     train_embeddings.append(embedding)
            #     train_labels.append(target.cpu().numpy())
            #
            # train_embeddings = np.concatenate(train_embeddings)
            # train_labels = np.concatenate(train_labels)
            #
            # train_labels = np.array(train_labels)
            # train_embeddings = np.array(train_embeddings)
            #
            # center_features = []
            # for i in range(1108):
            #     index = train_labels == i
            #     center_feature = np.mean(train_embeddings[index], axis=0)
            #     center_features.append(center_feature)
            #
            # center_features = np.array(center_features)
            #
            # joblib.dump(train_embeddings, "train_embeddings.pkl")
            # joblib.dump(train_labels, 'train_labels.pkl')
            # joblib.dump(center_features, 'center_features.pkl')

            test_embeddings = []
            cosine = []
            preds = np.empty(0)
            confi = np.empty(0)
            for i, (input, target) in tqdm(enumerate(tloader)):
                input = input.to(device)
                # target = target.to(device)
                embedding, cos = model(input, target)
                embedding = embedding.cpu().numpy()

                idx = cos.max(dim=-1)[1].cpu().numpy()
                confidence = cos.max(dim=-1)[0].cpu().numpy()
                preds = np.append(preds, idx, axis=0)
                confi = np.append(confi, confidence, axis=0)

                cosine.append(cos.cpu().numpy())
                test_embeddings.append(embedding)

            true_idx = np.empty(0)
            for i in range(19897):
                if confi[i] > confi[i + 19897]:
                    true_idx = np.append(true_idx, preds[i])
                else:
                    true_idx = np.append(true_idx, preds[i + 19897])

            submission = pd.read_csv('data/test.csv')
            submission['sirna'] = true_idx.astype(int)
            submission.to_csv(self.config.experment + '_s2_submission.csv', index=False, columns=['id_code', 'sirna'])

            test_embeddings = np.concatenate(test_embeddings)
            cosine = np.concatenate(cosine)
            joblib.dump(cosine, self.config.experment + '_cos.pkl')

            joblib.dump(test_embeddings, 'test_embeddings.pkl')

            assert len(test_embeddings) == 19897 * 2
        #     from sklearn.metrics.pairwise import cosine_similarity
        #
        #     similarity = cosine_similarity(test_embeddings, center_features)
        #     confi = similarity.max(axis=1)
        #     preds = similarity.argmax(axis=1)
        #
        #     true_idx = np.empty(0)
        #     for i in range(19897):
        #         if confi[i] > confi[i + 19897]:
        #             true_idx = np.append(true_idx, preds[i])
        #         else:
        #             true_idx = np.append(true_idx, preds[i + 19897])
        #
        # submission = pd.read_csv('data/test.csv')
        # submission['sirna'] = true_idx.astype(int)
        # submission.to_csv(self.config.experment + '_submission_angle.csv', index=False, columns=['id_code', 'sirna'])

    def confi_evaluate(self, model, avg=False):

        model.eval()
        ds, ds_val, ds_test = get_dataset(size=self.config.pic_size,
                                          six_channel=self.config.six_channel_aug)

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

    def data_leak_evaluate_mask(self):
        train_csv = pd.read_csv('./data/train.csv')
        test_csv = pd.read_csv('./data/test.csv')
        sub = pd.read_csv("dl_submission.csv")

        plate_groups = np.zeros((1108, 4), int)
        for sirna in range(1108):
            grp = train_csv.loc[train_csv.sirna == sirna, :].plate.value_counts().index.values
            assert len(grp) == 3
            plate_groups[sirna, 0:3] = grp
            plate_groups[sirna, 3] = 10 - grp.sum()

        all_test_exp = test_csv.experiment.unique()
        group_plate_probs = np.zeros((len(all_test_exp), 4))
        for idx in range(len(all_test_exp)):
            preds = sub.loc[test_csv.experiment == all_test_exp[idx], 'sirna'].values
            pp_mult = np.zeros((len(preds), 1108))
            pp_mult[range(len(preds)), preds] = 1

            sub_test = test_csv.loc[test_csv.experiment == all_test_exp[idx], :]
            assert len(pp_mult) == len(sub_test)

            for j in range(4):
                mask = np.repeat(plate_groups[np.newaxis, :, j], len(pp_mult), axis=0) == \
                       np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)

                group_plate_probs[idx, j] = np.array(pp_mult)[mask].sum() / len(pp_mult)
        exp_to_group = group_plate_probs.argmax(1)
        predicted_sides = joblib.load('cos.pkl')

        # todo get the predicate from the model

        def select_plate_group(pp_mult, idx):
            sub_test = test_csv.loc[test_csv.experiment == all_test_exp[idx], :]
            assert len(pp_mult) == len(sub_test)
            mask = np.repeat(plate_groups[np.newaxis, :, exp_to_group[idx]], len(pp_mult), axis=0) != \
                   np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)
            pp_mult[mask] = 0
            return pp_mult

        for idx in range(len(all_test_exp)):
            # print('Experiment', idx)
            indices = (test_csv.experiment == all_test_exp[idx])

            predicted1 = predicted_sides[:19897]
            predicted2 = predicted_sides[19897:]
            confi = []
            preds = []
            for predicted in [predicted1, predicted2]:
                preds_side = predicted[indices, :].copy()
                preds_side = select_plate_group(preds_side, idx)

                confi.append(preds_side.max(axis=1))
                preds.append(preds_side.argmax(axis=1))

            confi = np.concatenate(confi)
            preds = np.concatenate(preds)
            true_idx = np.empty(0)
            for i in range(indices.sum()):
                if confi[i] > confi[i + indices.sum()]:
                    true_idx = np.append(true_idx, preds[i])
                else:
                    true_idx = np.append(true_idx, preds[i + indices.sum()])

            sub.loc[indices, 'sirna'] = true_idx.astype(int)

        sub.to_csv('dl_submission.csv', index=False, columns=['id_code', 'sirna'])


def train_model(model, criterion, optimizer, scheduler, dataloaders, writer, num_epochs, name, config):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    min_loss = float('inf')
    max_acc = 0.0
    early_stop = 0
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
            if phase == 'val':
                early_stop += 1
            if phase == 'val' and epoch_acc[phase] > max_acc:
                early_stop = 0
                max_acc = epoch_acc[phase]
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "models/" + name + ".pth")

        writer.add_scalars('data/loss', {'train': epoch_loss['train'], 'val': epoch_loss['val']}, epoch)
        writer.add_scalars('data/acc', {'train': epoch_acc['train'], 'val': epoch_acc['val']}, epoch)
        scheduler.step(epoch_acc['val'])

        if early_stop > 10:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('min loss : {:4f}'.format(min_loss))
    print('max acc : {:4f}'.format(max_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


def train_model_s2(model, criterion, optimizer, scheduler, dataloaders, writer, num_epochs, model_name, config):
    min_loss = float('inf')
    max_acc = 0.0
    max_theta_acc = 0.0

    best_model_wts = copy.deepcopy(model.state_dict())

    early_stop = 0
    for epoch in range(num_epochs):

        # if epoch == 0:
        #     for name, child in model.module.backbone.named_children():
        #         for param in child.parameters():
        #             param.requires_grad = False
        #
        # else:
        #     for name, child in model.module.named_children():
        #         for param in child.parameters():
        #             param.requires_grad = True

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
                running_loss = 0.0
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
                epoch_acc = running_corrects / (
                        len(dataloaders[phase]) * config.train_batch_size)

                writer.add_scalar('val/loss', epoch_loss, epoch)

                writer.add_scalar('val/acc', epoch_acc, epoch)
                writer.add_text('Text', '{} Loss: {:.4f} '.format(phase, epoch_loss),
                                epoch)
                print('{} Loss: {:.4f} '.format(phase, epoch_loss))
                print('{} theta Acc: {:.4f}'.format(phase, epoch_acc))
                early_stop += 1
                if epoch_acc > max_theta_acc:
                    early_stop = 0
                    max_theta_acc = epoch_acc
                    torch.save(model.state_dict(), 'models/' + model_name + "_theta.pth")
                    best_model_wts = copy.deepcopy(model.state_dict())

                if config.experment == 'all':
                    embeddings = torch.cat(embeddings)
                    labels = torch.cat(labels)
                    accuracy, best_threshold, roc_curve_tensor = facade(embeddings, labels)
                    print('{} metric_acc: {:.4f} '.format(phase, accuracy))
                    print('{} thr: {:.4f} '.format(phase, best_threshold))

                    board_val(writer, accuracy, best_threshold, roc_curve_tensor, epoch)

                # if accuracy > max_acc:
                #     max_acc = accuracy
                #     torch.save(model.state_dict(), 'models/' + name + ".pth")
                #     best_model_wts = copy.deepcopy(model.state_dict())

                scheduler.step(epoch_acc)

        if early_stop > 10:
            break

    model.load_state_dict(best_model_wts)
    return model


def board_val(writer, accuracy, best_threshold, roc_curve_tensor, step):
    writer.add_scalar('val/metric_acc', accuracy, step)
    writer.add_scalar('best_threshold', best_threshold, step)
    writer.add_image('roc_curve', roc_curve_tensor, step)


def evaluate(model, vloader):
    model.eval()
    running_corrects = 0

    preds = np.empty(0)
    confi = np.empty(0)
    targets = []
    count = 0
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(vloader)):
            count += len(input)
            input = input.to(device)
            targets.append(target.cpu().numpy())
            embedding, cos = model(input, target)

            idx = cos.max(dim=-1)[1].cpu().numpy()
            confidence = cos.max(dim=-1)[0].cpu().numpy()
            preds = np.append(preds, idx, axis=0)
            confi = np.append(confi, confidence, axis=0)

    true_idx = np.empty(0)
    print(count)
    half_count = count // 2
    for i in range(half_count):
        if confi[i] > confi[i + half_count]:
            true_idx = np.append(true_idx, preds[i])
        else:
            true_idx = np.append(true_idx, preds[i + half_count])

    targets = np.concatenate(targets)
    assert len(true_idx) * 2 == len(targets)
    for i, j in zip(true_idx, targets):
        if i == j:
            running_corrects += 1

    epoch_acc = running_corrects / (
            len(vloader) * 32)
    return epoch_acc


def inference(model_path_dict):
    for experment in ['HEPG2', 'HUVEC', 'RPE', 'U2OS']:
        config = Config()
        config.experment = experment
        config.six_channel_aug = False
        if model_path_dict[experment].__contains__('resnext_50'):
            config.backbone = "resnext_50"
        elif model_path_dict[experment].__contains__('resnet_50'):
            config.backbone = "resnet_50"
        learner = Learner(config)
        model = learner.build_model(weight_path=model_path_dict[experment], mode='arcface')

        ds, ds_val, ds_test = get_dataset(size=config.pic_size,
                                          six_channel=config.six_channel_aug, experment=config.experment)
        tloader = D.DataLoader(ds_test, batch_size=config.test_batch_size, shuffle=False, num_workers=16)

        preds = np.empty(0)
        confi = np.empty(0)
        count = 0
        model.eval()
        with torch.no_grad():
            for i, (input, target) in tqdm(enumerate(tloader)):
                count += len(input)
                input = input.to(device)
                embedding, cos = model(input, target)
                idx = cos.max(dim=-1)[1].cpu().numpy()
                confidence = cos.max(dim=-1)[0].cpu().numpy()
                preds = np.append(preds, idx, axis=0)
                confi = np.append(confi, confidence, axis=0)

        true_idx = np.empty(0)
        print(count)
        half_count = count // 2
        for i in range(half_count):
            if confi[i] > confi[i + half_count]:
                true_idx = np.append(true_idx, preds[i])
            else:
                true_idx = np.append(true_idx, preds[i + half_count])

        test_csv_path = 'data/test.csv'
        df_test = pd.read_csv(test_csv_path)
        index = np.array([x.split('-')[0] for x in np.array(df_test.id_code)]) == experment

        sub = pd.read_csv('dl_submission.csv')
        pred = np.array(sub.sirna)
        pred[index] = true_idx
        sub['sirna'] = pred.astype(int)
        sub.to_csv('dl_submission.csv', index=False, columns=['id_code', 'sirna'])


def merge_submission():
    sub = pd.read_csv('submission.csv')
    pred = np.array(sub.sirna)
    full_embedding = np.zeros((39794, 1108))
    for experment in ['HEPG2', 'HUVEC', 'RPE', 'U2OS']:
        # for experment in ['HEPG2', 'RPE', 'U2OS']:
        file_name = experment + "_s2_submission.csv"
        embedding = joblib.load(experment + "_cos.pkl")

        sub_df = pd.read_csv(file_name)
        index = np.array([x.split('-')[0] for x in np.array(sub_df.id_code)]) == experment
        pred[index] = np.array(sub_df.sirna)[index]
        index = np.concatenate([index, index])

        full_embedding[index] = embedding[index]

    sub['sirna'] = pred.astype(int)
    sub.to_csv('s2_submission.csv', index=False, columns=['id_code', 'sirna'])
    joblib.dump(full_embedding, 'cos.pkl')


if __name__ == "__main__":
    # file_paths = {
    #     'HEPG2': 'models/stage2_Sep23_17-44-lr1_3e-06_lr2_0.0001_bs_32_ps_512_backbone_resnet_50_head_arcface_six_channel_aug_False_experment_HEPG2_theta.pth',
    #     'HUVEC': 'models/stage2_Sep23_20-39-lr1_3e-06_lr2_0.0001_bs_32_ps_512_backbone_resnet_50_head_arcface_six_channel_aug_False_experment_HUVEC_theta.pth',
    #     'RPE': 'models/stage2_Sep24_03-11-lr1_3e-06_lr2_0.0001_bs_32_ps_512_backbone_resnet_50_head_arcface_six_channel_aug_False_experment_RPE_theta.pth',
    #     'U2OS': 'models/stage2_Sep24_05-35-lr1_3e-06_lr2_0.0001_bs_32_ps_512_backbone_resnet_50_head_arcface_six_channel_aug_False_experment_U2OS_theta.pth'}
    # #
    # inference(file_paths)


    config = Config()
    learner = Learner(config)
    model = learner.stage_one()
    # s1_model = learner.build_model(
    #     weight_path='models/stage1_Sep03_07-08-lr1_0.0001_lr2_0.0001_bs_32_ps_448_backbone_densenet201_head_arcface_rgb_False_six_channel_aug_False.pth')
    # learner.confi_evaluate(s1_model)
    # s2_model = learner.stage_two(s1_model)

    # for experment in ['RPE', 'U2OS']:
    # for experment in ['HEPG2', 'HUVEC', 'RPE', 'U2OS']:
    #     config = Config()
    #     config.experment = experment
    #     config.six_channel_aug = False
    #     learner = Learner(config)
    #     # file_paths = {
    #     #     'HEPG2': 'models/stage2_Sep10_20-38-lr1_0.0001_lr2_0.0001_bs_32_ps_448_backbone_resnet_50_head_arcface_rgb_False_six_channel_aug_True_experment_HEPG2_theta.pth',
    #     #     'HUVEC': 'models/stage2_Sep10_23-22-lr1_0.0001_lr2_0.0001_bs_32_ps_448_backbone_resnet_50_head_arcface_rgb_False_six_channel_aug_True_experment_HUVEC_theta.pth',
    #     #     'RPE': 'models/stage2_Sep11_05-47-lr1_0.0001_lr2_0.0001_bs_32_ps_448_backbone_resnet_50_head_arcface_rgb_False_six_channel_aug_True_experment_RPE_theta.pth',
    #     #     'U2OS': 'models/stage2_Sep11_08-29-lr1_0.0001_lr2_0.0001_bs_32_ps_448_backbone_resnet_50_head_arcface_rgb_False_six_channel_aug_True_experment_U2OS_theta.pth'}
    #     #
    #     # file_paths2 = {
    #     #     'HEPG2': 'models/stage2_Sep12_02-31-lr1_0.0001_lr2_0.0001_bs_32_ps_448_backbone_resnet_50_head_arcface_rgb_False_six_channel_aug_False_experment_HEPG2_theta.pth',
    #     #     'HUVEC': 'models/stage2_Sep12_06-09-lr1_0.0001_lr2_0.0001_bs_32_ps_448_backbone_resnet_50_head_arcface_rgb_False_six_channel_aug_False_experment_HUVEC_theta.pth',
    #     #     'RPE': 'models/stage2_Sep12_12-27-lr1_0.0001_lr2_0.0001_bs_32_ps_448_backbone_resnet_50_head_arcface_rgb_False_six_channel_aug_False_experment_RPE_theta.pth',
    #     #     'U2OS': 'models/stage2_Sep12_14-44-lr1_0.0001_lr2_0.0001_bs_32_ps_448_backbone_resnet_50_head_arcface_rgb_False_six_channel_aug_False_experment_U2OS_theta.pth'}
    #
    #     # model = learner.stage_one()
    #     # model = learner.build_model(
    #     #     weight_path='models/stage1_Sep17_06-57-lr1_0.0001_lr2_0.0001_bs_32_ps_448_backbone_resnext_50_head_arcface_six_channel_aug_False_experment_all.pth',
    #     # )
    #     # model = learner.build_model(mode='arcface')
    #     s2_model = learner.stage_two(model)

    # learner.angle_evaluate(s2_model)

    # config = Config()
    # learner = Learner(config)
    # merge_submission()
    # learner.data_leak_evaluate_mask()
