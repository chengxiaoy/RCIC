from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
import pdb
from torch import nn
from torch.autograd import Variable


class CusAngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4, phiflag=True):
        super(CusAngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.m = m
        self.phiflag = phiflag
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, x):
        eps = 1e-12

        with torch.no_grad():
            self.fc.weight.div_(torch.norm(self.fc.weight, dim=1, keepdim=True))

        x_norm = F.normalize(x, dim=1)
        x_len = x.norm(2, 1, True).clamp_min(eps)
        # cos_theta = self.fc(x_norm)

        # cos_theta = torch.matmul(x_norm, F.normalize(self.weight))

        cos_theta = F.linear(x_norm, F.normalize(self.weight))

        cos_m_theta = self.mlambda[self.m](cos_theta)

        theta = Variable(cos_theta.data.acos())
        k = (self.m * theta / math.pi).floor()
        n_one = k * 0.0 - 1
        phi_theta = (n_one ** k) * cos_m_theta - 2 * k
        cos_theta = cos_theta * x_len
        phi_theta = phi_theta * x_len
        return cos_theta, phi_theta


class CusAngleLoss(nn.Module):
    def __init__(self):
        super(CusAngleLoss, self).__init__()
        self.iter = 0

        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0
        self.gamma = 0

    def forward(self, input, labels):
        with torch.autograd.set_detect_anomaly(True):
            self.iter += 1

            target = labels.view(-1, 1)  # size=(B,1)
            cos_theta, phi_theta = input
            index = torch.empty(cos_theta.shape)  # size=(B,Classnum)
            index.scatter_(1, target.data.view(-1, 1), 1)
            index = index.byte()
            index = Variable(index)

            # self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.iter))
            output = cos_theta * 1.0  # size=(B,Classnum)
            # output[index] -= cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
            # output[index] += phi_theta[index] * (1.0 + 0) / (1 + self.lamb)

            output[index] = output[index] - cos_theta[index] * (1.0 + 0)
            output[index] = output[index] + phi_theta[index] * (1.0 + 0)

            loss = F.cross_entropy(output, target.squeeze())

            # softmax loss

            # logit = F.log_softmax(output)
            #
            # logit = logit.gather(1, target).view(-1)
            # pt = logit.data.exp()
            #
            # loss = -1 * (1 - pt) ** self.gamma * logit
            # loss = loss.mean()
            return loss


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Arcface(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=51332, s=64., m=0.5):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m  # the margin value, default is 0.5
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)

    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings, kernel_norm)
        #         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm)  # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0  # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output, cos_theta
