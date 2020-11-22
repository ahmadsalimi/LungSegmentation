import torch
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple, Iterable
from typing import Union, Tuple, Iterable, Dict
from Models.Model import Model

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


class WholeMaskDiscriminator(Model):

    def __init__(self, batch_norms: Tuple[bool, bool, bool, bool, bool, bool, bool, bool]):
        super().__init__()

        kernel_size = (3, 2, 2)
        stride = (1, 2, 2)
        down_sample_kernel_size = (3, 4, 4)
        down_sample_stride = (1, 4, 4)
        padding = (1, 0, 0)

        self.conv = nn.Sequential(                                                                                                                              # 1 3   H   256 256
            ResidualBlock((3, 4, 8), kernel_size, stride, down_sample_kernel_size, down_sample_stride, padding=padding, batch_norms=batch_norms[:2]),           # 1 8   H   64  64
            ResidualBlock((8, 16, 32), kernel_size, stride, down_sample_kernel_size, down_sample_stride, padding=padding, batch_norms=batch_norms[2:4]),        # 1 32  H   16  16
            ResidualBlock((32, 64, 128), kernel_size, stride, down_sample_kernel_size, down_sample_stride, padding=padding, batch_norms=batch_norms[4:6]),      # 1 128 H   4   4
            ResidualBlock((128, 256, 512), kernel_size, stride, down_sample_kernel_size, down_sample_stride, padding=padding, batch_norms=batch_norms[6:]),     # 1 512 H   1   1
        )

        # H B   512
        self.lstm = nn.LSTM(512, 256)                       # 1 1   256 + 1 1   256 =>  1   512

        self.decider = nn.Sequential(                       # 1 512
            nn.Linear(512, 256),                            # 1 256
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(256, 128),                            # 1 128
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(128, 1),                              # 1 1
            nn.Sigmoid()
        )
    
    def forward(self, x_sample: torch.Tensor, x_label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # x_sample  1   1   3   H  256 256
        # x_label   1

        x = x_sample.squeeze(1)         # 1 3   H   256 256
        x = self.conv(x)                # 1 512 H   1   1
        x = x.squeeze(-1).squeeze(-1)   # 1 512 H
        x = x.permute(2, 0, 1)          # H 1   512
        hs, (h, c) = self.lstm(x)       # 1 1   256 + 1 1   256

        h = h + hs.mean(dim=0).unsqueeze(0)

        out = torch.cat((h, c), dim=2)  # 1 1   512
        out = out.permute(1, 0, 2)      # 1 1   512
        out = out.reshape(-1, 512)      # 1 512
        out = self.decider(out)         # 1 1
        out = out.flatten()             # 1

        if x_label is None:
            return {'positive_class_probability': out}
        
        loss = F.binary_cross_entropy(out, x_label)

        return {
            'positive_class_probability': out,
            'loss': loss
        }
    

class AttentionMaskDiscriminator(Model):

    def __init__(self, batch_norms: Tuple[bool, bool, bool, bool, bool, bool, bool, bool]):
        super().__init__()

        self.batch_norms = batch_norms

        kernel_size = (3, 2, 2)
        stride = (1, 2, 2)
        down_sample_kernel_size = (3, 4, 4)
        down_sample_stride = (1, 4, 4)
        padding = (1, 0, 0)

        self.conv = nn.Sequential(                                                                                                                              # 1 3   H   256 256
            ResidualBlock((3, 4, 8), kernel_size, stride, down_sample_kernel_size, down_sample_stride, padding=padding, batch_norms=batch_norms[:2]),           # 1 8   H   64  64
            ResidualBlock((8, 16, 32), kernel_size, stride, down_sample_kernel_size, down_sample_stride, padding=padding, batch_norms=batch_norms[2:4]),        # 1 32  H   16  16
            ResidualBlock((32, 64, 128), kernel_size, stride, down_sample_kernel_size, down_sample_stride, padding=padding, batch_norms=batch_norms[4:6]),      # 1 128 H   4   4
            ResidualBlock((128, 256, 512), kernel_size, stride, down_sample_kernel_size, down_sample_stride, padding=padding, batch_norms=batch_norms[6:]),     # 1 512 H   1   1
        )

        for param in self.conv.parameters():
            param.requires_grad = False

        self.attention = nn.Sequential(                     # H 512
            nn.Linear(512, 256),                            # H 256
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(256, 128),                            # H 128
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(128, 1),                              # H 1
            nn.Softmax(dim=0)
        )

        self.decider = nn.Sequential(                       # 1 512
            nn.Linear(512, 256),                            # 1 256
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(256, 128),                            # 1 128
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(128, 1),                              # 1 1
            nn.Sigmoid()
        )
    
    def forward(self, x_sample: torch.Tensor, x_label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # x_sample  1   1   3   H  256 256
        # x_label   1

        x = x_sample.squeeze(1)             # 1 3   H   256 256
        with torch.no_grad():
            x = self.conv(x)                # 1 512 H   1   1
        x = x.squeeze(-1).squeeze(-1)       # 1 512 H
        x = x.permute(2, 1, 0).flatten(1)   # H 512

        w = self.attention(x)               # H 1
        
        a = w.T.matmul(x)                   # 1 512

        out = self.decider(a)               # 1 1
        out = out.flatten()                 # 1

        if x_label is None:
            return {'positive_class_probability': out}
        
        loss = F.binary_cross_entropy(out, x_label)

        return {
            'positive_class_probability': out,
            'loss': loss
        }

    def get_other_model_to_load_from(self):
        """ Returns the model from which we are going to load the parameters for initialization.
        Default is the same model as we are using. """
        return WholeMaskDiscriminator(self.batch_norms)


class PatchMaskDiscriminator(Model):

    def __init__(self, batch_norms: Tuple[bool, bool, bool, bool, bool, bool, bool, bool]):
        super().__init__()

        self.conv = nn.Sequential(                                                                                                  # B 3   64  256 256
            ResidualBlock((3, 4, 8), 2, 2, 4, 4, padding=0, batch_norms=batch_norms[:2]),                                           # B 8   16  64  64
            ResidualBlock((8, 16, 32), 2, 2, 4, 4, padding=0, batch_norms=batch_norms[2:4]),                                        # B 32  4   16  16
            ResidualBlock((32, 64, 128), 2, 2, 4, 4, padding=0, batch_norms=batch_norms[4:6]),                                      # B 128 1   4   4
            ResidualBlock((128, 256, 512), (1, 2, 2), (1, 2, 2), (1, 4, 4), (1, 4, 4), padding=0, batch_norms=batch_norms[6:]),     # B 512 1   1   1
        )

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

    def forward(self, x_sample: torch.Tensor, x_label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # x_sample  B   P   3   64  256 256
        # x_label   B

        B, P = x_sample.shape[:2]

        x = x_sample.flatten(0, 1)  # B*P   3   64  256 256

        out = self.conv(x)          # B*P   512 1   1   1
        out = out.reshape(-1, 512)  # B*P   512
        out = self.decider(out)     # B*P   1
        out = out.reshape(B, P)     # B     P
        out = out.max(dim=1)[0]     # B

        if x_label is None:
            return {'positive_class_probability': out}

        loss = F.binary_cross_entropy(out, x_label)

        return {
            'positive_class_probability': out,
            'loss': loss
        }