import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import math


# Music Conv Block conv2d+bn+elu
# input size (1*96*800)
# MCB1 (64,48,400)
# MCB2 (128,16,200)
# MCB3 (128,4,50)
# MCB4 (128,1,25)
class MusicConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)):
        super(MusicConvBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size,
                                padding=padding, stride=stride)

        self.bn = nn.BatchNorm2d(output_channels, affine=True)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        feat_conv = self.conv2d(x)
        feat_bn = self.bn(feat_conv)
        feat_elu = self.elu(feat_bn)
        return feat_elu


class RhythmConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=5, padding=2):
        super(RhythmConvBlock, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size,
                                padding=padding)

        self.bn = nn.BatchNorm1d(output_channels, affine=True)
        self.elu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat_conv = self.conv1d(x)
        feat_bn = self.bn(feat_conv)
        feat_elu = self.elu(feat_bn)
        return feat_elu


class DanceConvBlock(nn.Module):
    # 好像不太对
    def __init__(self, input_channels, output_channels, kernel_size=(1, 3), padding=(0, 1), pooling_kernel=(1, 2),
                 stride=1):
        super(DanceConvBlock, self).__init__()

        # self.conv_graph = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size,
        #                         padding=padding)
        #
        # self.downsample =

        self.conv_graph = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding,
                                    stride=stride)
        self.downsample = nn.MaxPool2d(kernel_size=pooling_kernel)

    def forward(self, x):
        feat_conv = self.conv_graph(x)
        feat_downsample = self.downsample(feat_conv)

        return feat_downsample


if __name__ == '__main__':
    print("ConvBlock_test")

    # data1 = torch.rand(2, 200)  # input (2,200)
    # conv_layer1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=5, padding=2, stride=4)  # RBC1
    # convolved_data = conv_layer1(data1)  # expect output (64,50)

    # data2 = torch.rand(64, 50)
    # conv_layer2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2, stride=5)  # RBC2
    # convolved_data = conv_layer2(data2)  # expect output (128,10)

    # data = torch.rand(5, 60) # input (5, 60)
    # conv_layer = nn.Conv1d(in_channels=5, out_channels=64, kernel_size=5, padding=2, stride=2)  # RBC3
    # convolved_data = conv_layer(data)  # expect output (64,30)

    # data = torch.rand(64, 30)  # input (64, 30)
    # conv_layer = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2, stride=3)  # RBC4
    # convolved_data = conv_layer(data)  # expect output (128,10)

    # 实际上kernel size为3时另有一组卷积核参数
    # data = torch.rand(1, 1, 96, 800)
    # conv1 = MusicConvBlock(input_channels=1, output_channels=64, kernel_size=(5, 5), padding=2, stride=2)
    # conv_data = conv1(data)

    # data = torch.rand(1, 64, 48, 400)
    # conv1 = MusicConvBlock(input_channels=64, output_channels=128, kernel_size=(5, 3), padding=(2, 1), stride=(3, 2))
    # conv_data = conv1(data)

    # data = torch.rand(1, 128, 16, 200)
    # conv1 = MusicConvBlock(input_channels=128, output_channels=128, kernel_size=(5, 3), padding=(2, 1), stride=(4, 4))
    # conv_data = conv1(data)

    # data = torch.rand(1, 128, 4, 50)
    # conv1 = MusicConvBlock(input_channels=128, output_channels=128, kernel_size=(5, 3), padding=(2, 1), stride=(4, 2))
    # conv_data = conv1(data)

    # input_data = torch.rand(3, 18, 240)
    # conv_graph = nn.Conv2d(3, 64, kernel_size=(1, 3), padding=(0, 1))
    # downsample = nn.MaxPool2d(kernel_size=(1, 2))
    # feat_conv = conv_graph(input_data)
    # feat_down = downsample(feat_conv)
