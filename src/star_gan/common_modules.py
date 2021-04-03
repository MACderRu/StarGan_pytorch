import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class BaseConvBlock(nn.Module):
    def __init__(self,
                 input_features,
                 output_features,
                 kernel_size,
                 stride,
                 padding=None,
                 norm=False,
                 act=None):
        super().__init__()

        if padding is None:
            padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(input_features,
                              output_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)

        self.norm = nn.InstanceNorm2d(output_features, affine=True, track_running_stats=True) if norm else nn.Identity()

        if act:
            self.act = act()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class UpsampleBlock(nn.Module):
    def __init__(self,
                 input_features,
                 output_features,
                 kernel_size,
                 padding=None,
                 norm=None,
                 act=None,
                 spectral_normalize=False):
        super().__init__()

        if padding is None:
            padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(input_features,
                              output_features,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=1)
        if spectral_normalize:
            self.conv = spectral_norm(self.conv)

        self.norm = nn.InstanceNorm2d(output_features, affine=True, track_running_stats=True) if norm else nn.Identity()

        if act is None:
            self.act = nn.Identity()
        else:
            self.act = act()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.act(self.norm(self.conv(x)))


class DownsampleBlock(nn.Module):
    def __init__(self,
                 input_features,
                 output_features,
                 kernel_size,
                 norm=False,
                 act=None,
                 spectral_normalize=False):
        super().__init__()

        self.conv = nn.Conv2d(input_features,
                              output_features,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2,
                              stride=2)

        if spectral_normalize:
            self.conv = spectral_norm(self.conv)

        self.norm = nn.InstanceNorm2d(output_features, affine=True, track_running_stats=True) if norm else nn.Identity()

        if act is None:
            self.act = nn.Identity()
        else:
            self.act = act()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class BaseResidualBlock(nn.Module):
    def __init__(self, features, kernel_size, stride):
        super().__init__()

        self.block1 = BaseConvBlock(
            features,
            features,
            kernel_size=kernel_size,
            stride=stride,
            norm=True,
            act=nn.ReLU()
        )

        self.block2 = BaseConvBlock(
            features,
            features,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.act = nn.ReLU()

    def forward(self, x):
        z = self.block1(x)
        z = self.block2(z)
        return self.act(z + x)
