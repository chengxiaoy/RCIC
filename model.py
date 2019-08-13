from torchvision import models
from torch import nn
import torch
from torch.nn import Module

from loss.advance_loss import Arcface, l2_norm, CusAngleLinear, CusAngleLoss


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class My_Model(Module):

    def __init__(self, backbone, model_name, classes):
        super(My_Model, self).__init__()
        self.backbone = backbone

        if model_name.startswith('resnet'):
            self.num_ftrs = backbone.fc.in_features
            backbone.fc = Identity()
        elif model_name.startswith('dense'):
            self.num_ftrs = backbone.classifier.in_features
            backbone.classifier = Identity()

        # self.head = Arcface(embedding_size=self.num_ftrs, classnum=classes)
        self.head = CusAngleLinear(in_features=self.num_ftrs, out_features=classes)

    def forward(self, input, labels):
        input = self.backbone(input)
        input = l2_norm(input)
        output, theta = self.head(input)
        return output, theta

    def __repr__(self):
        return self.__class__.__name__


def get_model(model_name, use_rgb, classes=1108, pretrained=True):
    backbone = get_backbone(model_name, use_rgb, classes, pretrained)
    my_model = My_Model(backbone, model_name, classes)
    return my_model


def get_backbone(model_name, use_rgb, classes=1108, pretrained=True):
    if model_name == 'resnet_18':
        model = models.resnet18(pretrained=pretrained)
    elif model_name == 'resnet_101':
        model = models.resnet101(pretrained=pretrained)
    elif model_name == 'densenet201':
        model = models.densenet201(pretrained=pretrained)
    elif model_name == 'densenet169':
        model = models.densenet169(pretrained=pretrained)
    else:
        model = None

    if use_rgb:
        return model
    else:
        if model_name.startswith('resnet'):
            trained_kernel = model.conv1.weight
            new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :] = torch.stack([torch.mean(trained_kernel, 1)] * 6, dim=1)
            model.conv1 = new_conv
            return model
        elif model_name.startswith('dense'):
            trained_kernel = model.features.conv0.weight
            new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :] = torch.stack([torch.mean(trained_kernel, 1)] * 6, dim=1)
            model.features.conv0 = new_conv
            return model


def get_basic_model(model_name, use_rgb, classes=1108, pretrained=True):
    if model_name == 'resnet_18':
        model = models.resnet18(pretrained=pretrained)
    elif model_name == 'resnet_101':
        model = models.resnet101(pretrained=pretrained)
    elif model_name == 'densenet201':
        model = models.densenet201(pretrained=pretrained)
    elif model_name == 'densenet169':
        model = models.densenet169(pretrained=pretrained)
    else:
        model = None
    if model_name.startswith('resnet'):
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, classes)
    elif model_name.startswith('dense'):
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, classes)

    if use_rgb:
        return model
    else:
        if model_name.startswith('resnet'):
            trained_kernel = model.conv1.weight
            new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :] = torch.stack([torch.mean(trained_kernel, 1)] * 6, dim=1)
            model.conv1 = new_conv
            return model
        elif model_name.startswith('dense'):
            trained_kernel = model.features.conv0.weight
            new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :] = torch.stack([torch.mean(trained_kernel, 1)] * 6, dim=1)
            model.features.conv0 = new_conv
            return model
