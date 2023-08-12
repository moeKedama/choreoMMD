import torch
import torch.nn as nn
from network.common_network import RhythmConvBlock


class MusicRhythmEmbedding(nn.Module):
    def __init__(self):
        super(MusicRhythmEmbedding, self).__init__()

        self.conv1 = RhythmConvBlock(input_channels=2, output_channels=64, kernel_size=5, padding=2, stride=4)
        self.conv2 = RhythmConvBlock(input_channels=64, output_channels=128, kernel_size=5, padding=2, stride=5)
        self.Dence = nn.Linear(128 * 10, 32)

    def forward(self, x):
        feat_conv1 = self.conv1(x)
        feat_conv2 = self.conv2(feat_conv1)
        feat_flatten = feat_conv2.reshape(feat_conv1.shape[0], -1)
        output = self.Dence(feat_flatten)

        return output


class DanceRhythmEmbedding(nn.Module):
    def __init__(self):
        super(DanceRhythmEmbedding, self).__init__()

        self.conv1 = RhythmConvBlock(input_channels=5, output_channels=64, kernel_size=5, padding=2, stride=2)
        self.conv2 = RhythmConvBlock(input_channels=64, output_channels=128, kernel_size=5, padding=2, stride=3)
        self.Dence = nn.Linear(128 * 10, 32)

    def forward(self, x):
        feat_conv1 = self.conv1(x)
        feat_conv2 = self.conv2(feat_conv1)
        feat_flatten = feat_conv2.reshape(feat_conv1.shape[0], -1)
        output = self.Dence(feat_flatten)

        return output


class RhythmSignatureEmbedding(nn.Module):
    def __init__(self):
        super(RhythmSignatureEmbedding, self).__init__()

        self.Dence1 = nn.Linear(64, 128)
        self.Dence2 = nn.Linear(128, 128)
        self.Dence3 = nn.Linear(128, 13)

        self.pre_logits = nn.Identity()  # 占空一个先, music-artist-classification-crnn使用的是softmax

    def forward(self, x):
        feat_dence1 = self.Dence1(x)
        feat_dence2 = self.Dence2(feat_dence1)
        feat_dence3 = self.Dence3(feat_dence2)

        output = self.pre_logits(feat_dence3)

        return output


if __name__ == '__main__':
    print("RhythmEmbedding_test")
    device = torch.device("cuda")
    # data1 = torch.rand(32, 2, 200).to(device)
    # MRE = MusicRhythmEmbedding().to(device)
    # feat1 = MRE(data1)
    data2 = torch.rand(32, 5, 60).to(device)
    DRE = DanceRhythmEmbedding().to(device)
    feat2 = DRE(data2)
