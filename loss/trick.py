import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import *
import random
import math
from PIL import Image
import torch.nn.functional as F


class LabelSmoothing(nn.Module):
    "Implement label smoothing.  size表示类别总数  "

    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.confidence = 1.0 - smoothing  # if i=y的公式
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, x, target):
        """
        x表示输入 (N，M)N个样本，M表示总类数，每一个类的概率log P
        target表示label（M，）
        """

        assert x.size(1) == self.size
        x = self.log_softmax(x)
        true_dist = x.data.clone()  # 先深复制过来
        true_dist.fill_(self.smoothing / (self.size - 1))  # otherwise的公式
        # print true_dist
        # 变成one-hot编码，1表示按列填充，
        # target.data.unsqueeze(1)表示索引,confidence表示填充的数字
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


def shuffle_minibatch(inputs, targets, target_size=1108, hw_size=448, mixup=True):
    """Shuffle a minibatch and do linear interpolation between images and labels.
    Args:
        inputs: a numpy array of images with size batch_size x H x W x 3.
        targets: a numpy array of labels with size batch_size x 1.
        mixup: a boolen as whether to do mixup or not. If mixup is True, we
            sample the weight from beta distribution using parameter alpha=1,
            beta=1. If mixup is False, we set the weight to be 1 and 0
            respectively for the randomly shuffled mini-batches.
    """
    batch_size = inputs.shape[0]
    rp1 = torch.randperm(batch_size)
    inputs1 = inputs[rp1]
    targets1 = targets[rp1]
    targets1_1 = targets1.unsqueeze(1)

    rp2 = torch.randperm(batch_size)
    inputs2 = inputs[rp2]
    targets2 = targets[rp2]
    targets2_1 = targets2.unsqueeze(1)

    y_onehot = torch.FloatTensor(batch_size, target_size)
    y_onehot.zero_()
    targets1_oh = y_onehot.scatter_(1, targets1_1, 1)

    y_onehot2 = torch.FloatTensor(batch_size, target_size)
    y_onehot2.zero_()
    targets2_oh = y_onehot2.scatter_(1, targets2_1, 1)

    if mixup is True:
        a = np.random.beta(1, 1, [batch_size, 1])
    else:
        a = np.ones((batch_size, 1))

    b = np.tile(a[..., None, None], [1, 3, hw_size, hw_size])

    inputs1 = inputs1 * torch.from_numpy(b).float()
    inputs2 = inputs2 * torch.from_numpy(1 - b).float()

    c = np.tile(a, [1, target_size])
    targets1_oh = targets1_oh.float() * torch.from_numpy(c).float()
    targets2_oh = targets2_oh.float() * torch.from_numpy(1 - c).float()

    inputs_shuffle = inputs1 + inputs2
    targets_shuffle = targets1_oh + targets2_oh

    return inputs_shuffle, targets_shuffle


class RandomErasing(object):
    def __init__(self, EPSILON=0.5, sl=0.02, sh=0.4, r1=0.3, mean=None):
        if mean is None:
            mean = [0.0, 0.0, 0.0]
        self.EPSILON = EPSILON
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.EPSILON:
            return img

        for attempt in range(100):
            width, height = img.size
            area = width * height

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < width and h < height:
                x1 = random.randint(0, height - h)
                y1 = random.randint(0, width - w)

                img = np.array(img)

                img[y1:y1 + w, x1:x1 + h] = self.mean[0]
                # img[y1:y1 + w, x1:x1 + h, 1] = self.mean[1]
                # img[y1:y1 + w, x1:x1 + h, 2] = self.mean[2]

                img = Image.fromarray(img.astype('uint8'))
                return img

        return img
