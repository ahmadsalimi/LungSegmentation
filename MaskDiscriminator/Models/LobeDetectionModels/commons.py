import torch
from torch import nn
from typing import Union, Tuple, Iterable

class BasicBlock(nn.Module):

    def __init__(self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: Union[int, Tuple[int, int, int]], 
        stride: Union[int, Tuple[int, int, int]], 
        padding: Union[int, Tuple[int, int, int]] = 0, 
        batch_norm: bool = False):
        super().__init__()

        self.batch_norm = nn.BatchNorm3d(in_channels) if batch_norm else None

        self.conv: nn.Module = nn.Sequential(                                               # B I   H   L   W
            nn.Conv3d(in_channels, out_channels, 3, 1, padding=1),                          # B O   H   L   W
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding=padding),    # B O   H   L/2 W/2
            nn.LeakyReLU(0.2),
            nn.Dropout3d(0.2, inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x     B   I   H   L   W

        x = self.batch_norm(x) if self.batch_norm else x    # B I   H   L   W
        out: torch.Tensor = self.conv(x)                    # B O   H   L/2 W/2

        return out


class ResidualBlock(nn.Module):
    def __init__(self, 
        channels: Tuple[int, int, int], 
        kernel_size: Union[int, Tuple[int, int, int]], 
        stride: Union[int, Tuple[int, int, int]],
        down_sample_kernel_size: Union[int, Tuple[int, int, int]],
        down_sample_stride: Union[int, Tuple[int, int, int]],
        padding: Union[int, Tuple[int, int, int]] = 0,
        batch_norms: Tuple[bool, bool] = (False, True)):
        super().__init__()

        if len(channels) != 3:
            raise Exception('You should pass 3 channels as in, hidden and out channels')

        if len(batch_norms) != 2:
            raise Exception('You should pass 2 batch_norms for 2 basic layers')

        self.conv = nn.Sequential(                                                                                  # B I   H   L   W
            BasicBlock(channels[0], channels[1], kernel_size, stride, padding=padding, batch_norm=batch_norms[0]),  # B M   H   L/2 W/2
            BasicBlock(channels[1], channels[2], kernel_size, stride, padding=padding, batch_norm=batch_norms[1]),  # B O   H   L/4 W/4
        )

        self.downsample = nn.Sequential(                                                    # B I   H   L   W
            nn.Conv3d(channels[0], channels[2], 3, 1, padding=1),                           # B O   H   L   W
            nn.LeakyReLU(0.2),
            nn.AvgPool3d(down_sample_kernel_size, down_sample_stride, padding=padding),     # B O   H   L/4 W/4
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x     B   I   H   L   W

        out = self.conv(x)                  # B O   H   L/4 W/4
        downsampled = self.downsample(x)    # B O   H   L/4 W/4

        out += downsampled                  # B O   H   L/4 W/4

        return out
