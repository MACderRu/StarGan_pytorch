import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class BaseConvBlock(nn.Module):
    def __init__(self, in_features, out_features,
                 kernel_size, stride, padding=1,
                 norm=True, upsample=False, act='nn.ReLU', spectral_normalize=False):
        super().__init__()

        self.conv = nn.Conv2d(in_features,
                              out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)

        if spectral_normalize:
            self.conv = spectral_norm(self.conv)

        self.norm = nn.InstanceNorm2d(out_features, affine=True, track_running_stats=True) if norm else nn.Identity()
        try:
            self.act = eval(act)()
        except Exception:
            self.act = nn.Identity()

        self.upsample = upsample

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        return self.act(self.norm(self.conv(x)))


class BaseResidualBlock(nn.Module):
    def __init__(self, features, kernel_size, stride, padding=1, spectral_=False):
        super().__init__()

        self.block1 = BaseConvBlock(
            features,
            features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            spectral_normalize=spectral_
        )

        self.block2 = BaseConvBlock(
            features,
            features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            act="ne nado",
            spectral_normalize=spectral_
        )

        self.act = nn.LeakyReLU(0.01)

    def forward(self, x):
        z = self.block1(x)
        z = self.block2(z)
        return self.act(z + x)
