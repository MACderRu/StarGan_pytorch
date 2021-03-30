import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import CelebA

from torch.autograd import grad


def permute_labels(labels):
    permute_mask = np.random.permutation(labels.size(0))
    return labels[permute_mask]


class LabelTransformer:
    def __init__(self, target_attributes, idx2attr_orig):
        self.trg_attr = target_attributes
        self.trg_attr2idx = {name: idx for idx, name in enumerate(target_attributes)}

        self.idx2attr = idx2attr_orig
        self.attr2idx = {v: k for k, v in idx2attr_orig.items()}

        self.mask = self._get_mask()
        self.label_dim = len(target_attributes)

    def get_one_hot(self, x):
        if x.ndim == 1:
            return x[self.mask]

        return x[:, self.mask]

    def _get_mask(self):
        mask = np.array([0 for _ in range(len(self.trg_attr))])
        for attribute in self.trg_attr:
            mask[self.trg_attr2idx[attribute]] = self.attr2idx[attribute]

        return mask


def load_celeba(path):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    try:
        data = CelebA(path, transform=transforms, target_type='attr', download=False)
    except Exception as e:
        data = CelebA('./', transform=transforms, target_type='attr', download=True)

    return data


def save_checkpoint(path: str, checkpoint: dict, epoch_num: int) -> None:
    # checkpoint = {
    #     'epoch': epoch_num,
    #     'model_state_dict': model_gan.state_dict(),
    #     'optimizer_g_state_dict': optimizers['G'].state_dict(),
    #     'optimizer_d_state_dict': optimizers['D'].state_dict()
    # }

    try:
        torch.save(checkpoint, path)
    except Exception as e:
        print(f"Failed to save checkpoint, epoch: {epoch_num}")


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu')
    return checkpoint


def train_model(model,):
    pass
