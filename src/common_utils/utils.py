import os
import sys
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import CelebA
from glob import glob

from .config import Config


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


def load_celeba(path, transforms=None):
    try:
        data = CelebA(path, transform=transforms, target_type='attr', download=False)
    except Exception as e:
        data = CelebA('./', transform=transforms, target_type='attr', download=True)

    return data


def save_checkpoint(path: str, checkpoint: dict) -> None:
    try:
        torch.save(checkpoint, path)
    except Exception as e:
        print(f"Failed to save checkpoint...")


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu')
    return checkpoint


def terminate_launch(message, logging_state):
    print(message)

    if logging_state:
        wandb.finish()

    sys.exit(0)


def find_last_run(ckpt_dir_path: str) -> int:
    if not os.listdir(ckpt_dir_path):
        return 0

    runs_list = glob(ckpt_dir_path + '/*')
    last_run_path = sorted(runs_list, key=lambda name: int(name.split('/')[-1][3:]))[-1]

    return int(last_run_path.split('/')[-1][3:])


def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
