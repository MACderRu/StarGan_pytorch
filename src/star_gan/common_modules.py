import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class InstNorm2d(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(features, affine=True, track_running_stats=True)

    def forward(self, x):
        return self.norm(x)


class BaseConvBlock(nn.Module):
    def __init__(self,
                 input_features,
                 output_features,
                 kernel_size,
                 stride,
                 padding=None,
                 norm=False,
                 act=None,
                 bias=False):
        super().__init__()

        if padding is None:
            padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(input_features,
                              output_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)

        self.norm = InstNorm2d(output_features) if norm else nn.Identity()

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
                 norm=False,
                 act=None,
                 spectral_normalize=False):
        super().__init__()

        if padding is None:
            padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(input_features,
                              output_features,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=1,
                              bias=False)
        if spectral_normalize:
            self.conv = spectral_norm(self.conv)

        self.norm = InstNorm2d(output_features) if norm else nn.Identity()

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
                 padding=None,
                 norm=False,
                 act=None,
                 spectral_normalize=False):
        super().__init__()

        if padding is None:
            padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(input_features,
                              output_features,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=2,
                              bias=False)

        if spectral_normalize:
            self.conv = spectral_norm(self.conv)

        self.norm = InstNorm2d(output_features) if norm else nn.Identity()

        if act is None:
            self.act = nn.Identity()
        else:
            self.act = act()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class BaseResidualBlock(nn.Module):
    def __init__(self, features, kernel_size):
        super().__init__()

        padding = kernel_size // 2

        self.blocks = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=1, stride=1, bias=False),
            InstNorm2d(features // 2),
            nn.ReLU(),
            nn.Conv2d(features // 2, features // 2, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            InstNorm2d(features // 2),
            nn.ReLU(),
            nn.Conv2d(features // 2, features, kernel_size=1, stride=1, bias=False),
            InstNorm2d(features)
        )

        self.act = nn.ReLU()

    def forward(self, x):
        z = self.blocks(x)
        return self.act(z + x)
