import torch
from torch import nn


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()

        self.batch_norm = nn.BatchNorm3d(in_channels) if batch_norm else lambda x: x

        self.conv = nn.Sequential(                                                          # B I   H   L   W
            nn.Conv3d(in_channels, out_channels, 3, 1, padding=1),                          # B O   H   L   W
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels, out_channels, (3, 2, 2), (1, 2, 2), padding=(1, 0, 0)), # B O   H   L/2 W/2
            nn.LeakyReLU(0.2),
            nn.Dropout3d(0.2, inplace=True)
        )
    
    def forward(self, x):
        # x     B   I   H   L   W

        x = self.batch_norm(x)      # B I   H   L   W
        out = self.conv(x)          # B O   H   L/2 W/2

        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels, batch_norms=(False, True)):
        super().__init__()

        if len(channels) != 3:
            raise Exception('You should pass 3 channels as in, hidden and out channels')

        if len(batch_norms) != 2:
            raise Exception('You should pass 2 batch_norms for 2 basic layers')

        self.conv = nn.Sequential(                                  # B I   H   L   W
            BasicBlock(channels[0], channels[1], batch_norms[0]),   # B M   H   L/2 W/2
            BasicBlock(channels[1], channels[2], batch_norms[1]),   # B O   H   L/4 W/4
        )

        self.downsample = nn.Sequential(                            # B I   H   L   W
            nn.Conv3d(channels[0], channels[2], 3, 1, padding=1),   # B O   H   L   W
            nn.LeakyReLU(0.2),
            nn.AvgPool3d((3, 4, 4), (1, 4, 4), padding=(1, 0, 0)),  # B O   H   L/4 W/4
        )
    
    def forward(self, x):
        # x     B   I   H   L   W

        out = self.conv(x)                  # B O   H   L/4 W/4
        downsampled = self.downsample(x)    # B O   H   L/4 W/4

        out += downsampled                  # B O   H   L/4 W/4

        return out


class MaskDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(                          # B 3   H   256 256
            ResidualBlock((3, 4, 8), (False, True)),        # B 8   H   64  64
            ResidualBlock((8, 16, 32), (True, True)),       # B 32  H   16  16
            ResidualBlock((32, 64, 128), (False, False)),   # B 128 H   4   4
            ResidualBlock((128, 256, 512), (False, False)), # B 512 H   1   1
        )

        # H B   512
        self.lstm = nn.LSTM(512, 256)                       # 1 B   256 + 1 B   256 =>  B   512

        self.decider = nn.Sequential(                       # B 512
            nn.Linear(512, 256),                            # B 256
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(256, 128),                            # B 128
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(128, 1),                              # B 1
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x     B   3   H   256  256

        x = self.conv(x)                # B 512 H   1   1
        x = x.squeeze(-1).squeeze(-1)   # B 512 H
        x = x.permute(2, 0, 1)          # H B   512
        _, (h, c) = self.lstm(x)        # 1 B   256 + 1 B   256
        out = torch.cat((h, c), dim=2)  # 1 B   512
        out = out.permute(1, 0, 2)      # B 1   512
        out = out.reshape(-1, 512)      # B 512
        out = self.decider(out)         # B 1
        out = out.flatten()             # B

        return out