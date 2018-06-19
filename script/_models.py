import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    """
    custom weights initialization called on netG and netD
    https://github.com/pytorch/examples/blob/master/dcgan/main.py
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class ResidualBlock(nn.Module):
    # one_to_many 논문에서 제시된 resunit 구조
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = self.bn1(x)
        residual = self.relu1(residual)
        residual = self.conv1(residual)
        residual = self.bn2(residual)
        residual = self.relu2(residual)
        residual = self.conv2(residual)
        return x + residual


def ResidualBlocks(channels, size):
    bundle = []
    for i in range(size):
        bundle.append(ResidualBlock(channels))
    return nn.Sequential(*bundle)


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """
    Custom convolutional layer for simplicity.
    bn 을 편하게 사용하기 위해 만든 함수
    """
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)