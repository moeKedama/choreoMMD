import torch.nn as nn
from network.common_network import MusicConvBlock

# 4 Music Conv Block conv2D+bn+elu
# then 2 GRU
# LAST Dense head
class MusicEncoder(nn.Module):
    def __init__(self):
        super(MusicEncoder, self).__init__()

        self.MCB1 = MusicConvBlock(input_channels=1, output_channels=64, kernel_size=(5, 5), padding=2,
                                   stride=2)  # (1,96,800) -> (64,48,400)
        self.MCB2 = MusicConvBlock(input_channels=64, output_channels=128, kernel_size=(5, 3), padding=(2, 1),
                                   stride=(3, 2))  # (64,48,400) -> (128,16,200)
        self.MCB3 = MusicConvBlock(input_channels=128, output_channels=128, kernel_size=(5, 3), padding=(2, 1),
                                   stride=(4, 4))  # (128,16,200) -> (128,4,50)
        self.MCB4 = MusicConvBlock(input_channels=128, output_channels=128, kernel_size=(5, 3), padding=(2, 1),
                                   stride=(4, 2))  # (128,4,50) -> (128,1,25)
        self.GRU1 = nn.GRU(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.GRU2 = nn.GRU(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.Dence = nn.Linear(128 * 25, 32)

    def forward(self, x):
        feat_mcb1 = self.MCB1(x)
        feat_mcb2 = self.MCB2(feat_mcb1)
        feat_mcb3 = self.MCB3(feat_mcb2)
        feat_mcb4 = self.MCB4(feat_mcb3)
        feat_sqz = feat_mcb4.squeeze(1)  # (128,1,25) -> (128,25)
        _, hn1 = self.GRU1(feat_sqz)
        _, hn2 = self.GRU1(hn1)
        print(hn2.shape)
        feat_flatten = hn2.view(hn2.shape[0], -1)
        output = self.Dence(feat_flatten)

        return output
