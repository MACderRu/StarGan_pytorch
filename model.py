import torch
from torch import nn
import torch.nn.functional as F

# from .utils import compute_gradient_penalty, permute_labels
from torch.nn.utils import spectral_norm

class BaseConvBlock(nn.Module):
    def __init__(self, in_features, out_features,
                 kernel_size, stride, padding=1,
                 norm=True, upsample=False, act='nn.ReLU', spectral_=False):
        super().__init__()

        self.conv = nn.Conv2d(in_features,
                              out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        
        if spectral_:
            self.conv = spectral_norm(self.conv)


        self.norm = nn.InstanceNorm2d(out_features, affine=True, track_running_stats=True) if norm else nn.Identity()
        try:
            self.act = eval(act)()
        except BaseException as e:
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
            spectral_=spectral_
        )

        self.block2 = BaseConvBlock(
            features,
            features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            act="ne nado",
            spectral_=spectral_
        )
        
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x):
        z = self.block1(x)
        z = self.block2(z)
        return self.act(z + x)


class Generator(nn.Module):
    def __init__(self, input_features, output_features, residual_num, image_size):
        super().__init__()
        self.image_size = image_size
        self.down_sample = nn.Sequential(
            BaseConvBlock(input_features, 64, 7, 1, padding=3),
            BaseConvBlock(64, 128, 4, 2),
            BaseConvBlock(128, 256, 4, 2)
        )

        self.residual = nn.Sequential(
            *[
                BaseResidualBlock(256, kernel_size=3, stride=1) for _ in range(residual_num)
            ]
        )

        self.up_sample = nn.Sequential(
            BaseConvBlock(256, 128, kernel_size=3, stride=1, padding=1, upsample=True),
            BaseConvBlock(128, 64, kernel_size=3, stride=1, padding=1, upsample=True),
            BaseConvBlock(64, output_features,
                          kernel_size=7, stride=1, padding=3,
                          norm=False, upsample=False, act='nn.Tanh')
        )

    def forward(self, x, labels):

        c1, c2 = labels.size()

        labels = labels.view(c1, c2, 1, 1).expand(c1, c2, self.image_size, self.image_size)

        x = torch.cat([x, labels], dim=1)
        x = self.down_sample(x)
        x = self.residual(x)
        x = self.up_sample(x)
        return x

        
class Discriminator(nn.Module):
    def __init__(self, input_features, label_features, image_size):
        super().__init__()

        self.model = nn.Sequential(
            BaseConvBlock(input_features, 64, 4, 2, 1, norm=False, act='nn.LeakyReLU', spectral_=True),
            BaseConvBlock(64, 128, 4, 2, 1, act='nn.LeakyReLU', norm=False, spectral_=True),
            BaseConvBlock(128, 256, 4, 2, 1, act='nn.LeakyReLU', norm=False, spectral_=True),
            BaseConvBlock(256, 512, 4, 2, 1, act='nn.LeakyReLU', norm=False, spectral_=True),
            BaseConvBlock(512, 1024, 4, 2, 1, act='nn.LeakyReLU', norm=False, spectral_=True),
            BaseConvBlock(1024, 2048, 4, 2, 1, act='nn.LeakyReLU', norm=False, spectral_=True),
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
                 input_features,
                 lbl_features,
                 image_size,
                 residual_block_number=6,
                 ):
        super().__init__()

        self.G = Generator(
            input_features=input_features + lbl_features,
            output_features=3,
            residual_num=residual_block_number,
            image_size=image_size
        )

        self.D = Discriminator(
            input_features=input_features,
            label_features=lbl_features,
            image_size=image_size
        )

    def forward_g(self, image, label):
        return self.G(image, label)
        
    def forward_d(self, image):
        return self.D(image)

    def generate(self, image, label):
        return self.G(image, label)
