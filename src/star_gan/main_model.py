import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from .common_modules import (BaseConvBlock,
                             BaseResidualBlock,
                             DownsampleBlock,
                             UpsampleBlock)


class Generator(nn.Module):
    def __init__(self, label_size, residual_num, image_size):
        super().__init__()
        self.image_size = image_size
        self.down_sample = nn.Sequential(
            BaseConvBlock(3 + label_size, 64, kernel_size=7, stride=1, norm=True, act=nn.ReLU),
            DownsampleBlock(64, 128, kernel_size=4, norm=True, act=nn.ReLU),
            DownsampleBlock(128, 256, kernel_size=4, norm=True, act=nn.ReLU),
        )

        self.residual = nn.Sequential(
            *[
                BaseResidualBlock(256, kernel_size=3, stride=1) for _ in range(residual_num)
            ]
        )

        self.up_sample = nn.Sequential(
            UpsampleBlock(256, 128, kernel_size=4, norm=True, act=nn.ReLU),
            UpsampleBlock(128, 64, kernel_size=4, norm=True, act=nn.ReLU),
        )

        self.out_block = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x, labels):
        c1, c2 = labels.size()

        labels = labels.view(c1, c2, 1, 1).expand(c1, c2, self.image_size, self.image_size)

        x = torch.cat([x, labels], dim=1)
        x = self.down_sample(x)
        x = self.residual(x)
        x = self.up_sample(x)
        x = self.out_block(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, label_features, image_size):
        super().__init__()

        self.model = nn.Sequential(
            DownsampleBlock(3, 64, 4, act=nn.LeakyReLU, spectral_normalize=True),
            DownsampleBlock(64, 128, 4, act=nn.LeakyReLU, spectral_normalize=True),
            DownsampleBlock(128, 256, 4, act=nn.LeakyReLU, spectral_normalize=True),
            DownsampleBlock(256, 512, 4, act=nn.LeakyReLU, spectral_normalize=True),
            DownsampleBlock(512, 1024, 4, act=nn.LeakyReLU, spectral_normalize=True),
            DownsampleBlock(1024, 2048, 4, act=nn.LeakyReLU, spectral_normalize=True),
        )

        self.patch_discriminator_conv = spectral_norm(nn.Conv2d(2048, 1, kernel_size=3, padding=1))

        cls_size = image_size // 64
        self.classification_conv = spectral_norm(nn.Conv2d(2048, label_features, kernel_size=cls_size))
        self.global_pooling = nn.AvgPool2d(cls_size)

    def forward(self, x):
        x = self.model(x)
        patch_output = self.patch_discriminator_conv(x)
        cls_output = self.classification_conv(x)
        return patch_output, cls_output.view(cls_output.size(0), cls_output.size(1))


class StarGAN(nn.Module):
    def __init__(self,
                 lbl_features,
                 image_size,
                 residual_block_number=6,
                 ):
        super().__init__()

        self.G = Generator(
            label_size=lbl_features,
            residual_num=residual_block_number,
            image_size=image_size
        )

        self.D = Discriminator(
            label_features=lbl_features,
            image_size=image_size
        )

    def forward_g(self, image, label):
        return self.G(image, label)

    def forward_d(self, image):
        return self.D(image)

    def generate(self, image, label):
        return self.G(image, label).detach()

