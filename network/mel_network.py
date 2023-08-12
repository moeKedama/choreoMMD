import torch
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
        feat_sqz = feat_mcb4.squeeze(2)  # (batch,128,1,25) -> (batch,128,25)
        feat_trans = feat_sqz.permute(0, 2, 1)
        o1, hn1 = self.GRU1(feat_trans)
        o2, hn2 = self.GRU2(o1)
        print(o2.shape)
        # feat_flatten = o2.view(o2.shape[0], -1) # 因为使用了permute导致内存不连续,需要使用contiguous()或者改为reshape()
        feat_flatten = o2.reshape(o2.shape[0], -1)
        output = self.Dence(feat_flatten)

        return output


if __name__ == '__main__':
    print("MusicEncoder_test")
    device = torch.device('cuda')

    data = torch.rand(32, 1, 96, 800).to(device)

    # MCB1 = MusicConvBlock(input_channels=1, output_channels=64, kernel_size=(5, 5), padding=2,
    #                       stride=2).to(device)  # (1,96,800) -> (64,48,400)
    # MCB2 = MusicConvBlock(input_channels=64, output_channels=128, kernel_size=(5, 3), padding=(2, 1),
    #                       stride=(3, 2)).to(device)  # (64,48,400) -> (128,16,200)
    # MCB3 = MusicConvBlock(input_channels=128, output_channels=128, kernel_size=(5, 3), padding=(2, 1),
    #                       stride=(4, 4)).to(device)  # (128,16,200) -> (128,4,50)
    # MCB4 = MusicConvBlock(input_channels=128, output_channels=128, kernel_size=(5, 3), padding=(2, 1),
    #                       stride=(4, 2)).to(device)  # (128,4,50) -> (128,1,25)
    # GRU1 = nn.GRU(input_size=128, hidden_size=256, num_layers=1, batch_first=True).to(device)
    # GRU2 = nn.GRU(input_size=256, hidden_size=128, num_layers=1, batch_first=True).to(device)
    # Dence = nn.Linear(128 * 25, 32).to(device)
    #
    # feat_mcb1 = MCB1(data)
    # feat_mcb2 = MCB2(feat_mcb1)
    # feat_mcb3 = MCB3(feat_mcb2)
    # feat_mcb4 = MCB4(feat_mcb3)
    # feat_sqz = feat_mcb4.squeeze(2)  # (batch,128,1,25) -> (batch,128,25)
    # feat_trans = feat_sqz.permute(0, 2, 1)
    # o1, hn1 = GRU1(feat_trans)
    # o2, hn2 = GRU2(o1)
    # print(o2.shape)
    # feat_flatten = o2.view(o2.shape[0], -1) # 因为使用了permute导致内存不连续,需要使用contiguous()或者改为reshape()
    # feat_flatten = o2.reshape(o2.shape[0], -1)
    # feat_flatten = o2.reshape(32, -1)
    # output = Dence(feat_flatten)
    #

    Encoder = MusicEncoder().to(device)

    emb = Encoder(data)
