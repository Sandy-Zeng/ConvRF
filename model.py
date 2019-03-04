import torch.nn as nn
import torch.utils.model_zoo as model_zoo

'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class VGGRF(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features):
        super(VGGRF, self).__init__()
        self.features = features

         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                uniform_weights = torch.Tensor(np.random.rand(m.kernel_size[0]*m.kernel_size[1]*m.in_channels,m.out_channels))
                normalized_weights = F.softmax(uniform_weights,dim=0).view(m.weight.shape)
                m.weight = torch.nn.Parameter(normalized_weights)
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def conv(self, x,conv=True):
        if conv:
            x = self.features(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # self.classifier.fit(x,y)
        return x


def make_layers(cfg, batch_norm=False,depth=3):
    layers = []
    in_channels = 1

    for n in range(depth):
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'MC':
                layers += [nn.MaxPool2d(kernel_size=3,stride=1,padding=1)]
            elif v == 'AC':
                layers += [nn.AvgPool2d(kernel_size=3,stride=1,padding=1)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
    'F': ['MC','AC',31,'M'],
    'G': ['AC',31,'M'],
}


def vggRF(mode='F',depth=3):
    """VGG 11-layer model (configuration "A")"""
    return VGGRF(make_layers(cfg[mode],depth=depth))
