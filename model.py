from torchvision import models
from torch import nn
import torch


def get_model(model_name, use_rgb, classes=1108, pretrained=True):
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



