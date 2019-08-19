from torchvision import models
from torch import nn
import torch
from torch.nn import Module
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter

from loss.advance_loss import Arcface, l2_norm, CusAngleLinear, CusAngleLoss


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class My_Model(Module):

    def __init__(self, backbone, model_name, classes,  head_type='line',embedding_size=512):
        super(My_Model, self).__init__()

        self.pre_process = nn.BatchNorm2d(6)
        self.backbone = backbone

        if model_name.startswith('resnet'):
            self.num_ftrs = backbone.fc.in_features
            backbone.fc = Identity()
        elif model_name.startswith('dense'):
            self.num_ftrs = backbone.classifier.in_features
            backbone.classifier = Identity()

        self.head_type = head_type
        self.output_layer = Sequential(
            # BatchNorm2d(512),
            Dropout(0.3),
            Flatten(),
            Linear(self.num_ftrs, int(embedding_size)),
            BatchNorm1d(int(embedding_size)))

        self.dropout = nn.Dropout(0.2)

        self.line = nn.Linear(self.num_ftrs, classes)
        self.arcface = Arcface(embedding_size=int(embedding_size), classnum=classes)

    def set_head_type(self, head_type):
        self.head_type = head_type

    def forward(self, input, labels):
        # bn 6 channels
        input = self.pre_process(input)
        output = self.backbone(input)

        if self.head_type == 'line':
            output = self.dropout(output)
            return self.line(output)
        if self.head_type == 'arcface':
            output = self.output_layer(output)
            output = l2_norm(output)
            if self.training:
                output = self.arcface(output, labels)
            return output

    def __repr__(self):
        return self.__class__.__name__


def get_model(model_name, use_rgb, head_type, classes=1108, pretrained=True):
    backbone = get_backbone(model_name, use_rgb, classes, pretrained)
    my_model = My_Model(backbone, model_name, classes, head_type)
    return my_model


def get_backbone(model_name, use_rgb, classes=1108, pretrained=True):
    if model_name == 'resnet_18':
        model = models.resnet18(pretrained=pretrained)
    elif model_name == 'resnet_50':
        model = models.resnet50(pretrained=pretrained)
    elif model_name == 'resnet_101':
        model = models.resnet101(pretrained=pretrained)
    elif model_name == 'densenet201':
        model = models.densenet201(pretrained=pretrained)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
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
