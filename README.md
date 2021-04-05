# StarGan_pytorch
My implementation of StarGan paper <https://arxiv.org/pdf/1711.09020.pdf>

torch version: 1.8

python >= 3.8

Some addition features:
- WGAN-GP is replaced with SpectralNormalization of Critic
- ConvTranspose is replaced by Upsample x2 + Conv2D
- Additional loss for Generator: reconstraction for fake image with small coeff

If you find this repo useful you are welcome to contribute :)
