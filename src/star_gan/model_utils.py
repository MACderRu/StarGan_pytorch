import os
import wandb
import typing as tp
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision
from torchvision import transforms
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torch.autograd import grad
from scipy import linalg
from tqdm import tqdm

from ..common_utils.utils import (save_checkpoint,
                                  load_celeba,
                                  find_last_run,
                                  LabelTransformer,
                                  LabelTransformerUpdated,
                                  optimizer_to)

from ..common_utils.config import Config
from ..star_gan.main_model import StarGAN


def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """
    :param discriminator: StarGan discriminator
    :param real_samples: Samples from P_real
    :param fake_samples: Samples from Q_generated
    :return: GP part
    """
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    x_hat = (alpha * real_samples.data + (1 - alpha) * fake_samples.data).requires_grad_(True)
    out, _ = discriminator(x_hat)
    weight = torch.ones(out.size()).to(real_samples.device)
    gradient_dxhat = grad(
        outputs=out,
        inputs=x_hat,
        grad_outputs=weight,
        retain_graph=True,
        create_graph=True,
        only_inputs=True
    )[0]
    grad_tensor = gradient_dxhat.view(real_samples.size(0), -1)
    norm = torch.norm(grad_tensor, dim=1)
    return ((norm - 1) ** 2).mean()


def get_description(desc, epoch, d_cls, d_adv, g_cls, g_adv, g_rec, g_rec_fake):
    return desc.format(epoch, g_cls, g_adv, g_rec, g_rec_fake, d_cls, d_adv)


def train_epoch(train_loader, model, optimizers, epoch_num, config, label_transformer, log_step=None):
    model.train()
    iter_idx = 0

    optimizer_d = optimizers['D']
    optimizer_g = optimizers['G']

    device = next(model.parameters())

    losses = {
        "D": {
            'adversarial': [],
            'classification': [],
        },
        "G": {
            'adversarial': [],
            'rec_fake': [],
            'rec_real': [],
            'classification': []
        }
    }

    description = "Epoch: {}: Loss G: cls {:.4f}, adv {:.4f}, rec_real {:.4f}, rec_fake {:.4f}; \
        Loss D: cls {:.4f}, adv {:.4f}"

    pbar = tqdm(train_loader, leave=False, desc=description.format(epoch_num, 0, 0, 0, 0, 0, 0))

    g_adv_loss = torch.tensor([0])
    g_cls_loss = torch.tensor([0])
    g_loss_rec_orig = torch.tensor([0])
    g_loss_rec_fake = torch.tensor([0])

    # gp_loss = torch.tensor([0])

    for image, label in pbar:
        image = image.to(device)
        label = label.to(device)

        iter_idx += 1
        true_labels = label_transformer.get_target(label).type(torch.float32)
        fake_labels = label_transformer.revert_one_label(true_labels)

        image_fake = model.forward_g(image, fake_labels).detach()
        fake_patch_out, fake_cls_out = model.forward_d(image_fake)
        real_patch_out, real_cls_out = model.forward_d(image)

        d_loss_adv = - real_patch_out.mean() + fake_patch_out.mean()
        d_loss_cls = F.binary_cross_entropy_with_logits(real_cls_out, true_labels, size_average=False) / real_cls_out.size(0)

        # gp_loss = compute_gradient_penalty(model.D, image, image_fake)

        d_loss = d_loss_adv + config['lambda_cls'] * d_loss_cls  # + config['lambda_gp'] * gp_loss
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        losses['D']['adversarial'].append(d_loss_adv.item())
        # losses['D']['gp_loss'].append(gp_loss.item())
        losses['D']['classification'].append(d_loss_cls.item())

        pbar.set_description(get_description(description,
                                             epoch_num,
                                             d_loss_cls.item(),
                                             d_loss_adv.item(),
                                             g_cls_loss.item(),
                                             g_adv_loss.item(),
                                             g_loss_rec_orig.item(),
                                             g_loss_rec_fake.item()))

        if iter_idx % config['n_critic'] == 0:
            image_fake = model.forward_g(image, fake_labels)

            g_loss_rec_fake = torch.mean((image_fake - image) ** 2)

            image_reconstructed = model.forward_g(image_fake, true_labels)

            g_loss_rec_orig = torch.mean(torch.abs(image - image_reconstructed))
            fake_patch_out, cls_fake = model.forward_d(image_fake)

            g_cls_loss = F.binary_cross_entropy_with_logits(cls_fake, fake_labels, size_average=False) / cls_fake.size(0)
            g_adv_loss = -fake_patch_out.mean()

            optimizer_g.zero_grad()
            loss = (g_adv_loss + config['lambda_cls'] * g_cls_loss
                    + config['lambda_rec_original'] * g_loss_rec_orig
                    + config['lambda_rec_fake'] * g_loss_rec_fake)
            loss.backward()
            optimizer_g.step()

            losses['G']['adversarial'].append(g_adv_loss.item())
            losses['G']['rec_fake'].append(g_loss_rec_fake.item())
            losses['G']['rec_real'].append(g_loss_rec_orig.item())
            losses['G']['classification'].append(g_cls_loss.item())

            pbar.set_description(get_description(description,
                                                 epoch_num,
                                                 d_loss_cls.item(),
                                                 d_loss_adv.item(),
                                                 g_cls_loss.item(),
                                                 g_adv_loss.item(),
                                                 g_loss_rec_orig.item(),
                                                 g_loss_rec_fake.item()))

        if log_step and iter_idx % log_step == 0:
            for type_m in losses:
                for loss_type in losses[type_m]:
                    wandb.log({
                        type_m + '/' + loss_type: np.mean(losses[type_m][loss_type][-log_step:])
                    })

    return losses


def fid_distance_from_activations(acts1, acts2):
    mu1 = np.mean(acts1, axis=0)
    sigma1 = np.cov(acts1, rowvar=False)

    mu2 = np.mean(acts2, axis=0)
    sigma2 = np.cov(acts2, rowvar=False)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    eps = 1e-8
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def upscale_twice_tensor(im_tensor):
    return F.interpolate(im_tensor, scale_factor=2, mode='bilinear', align_corners=True)


@torch.no_grad()
def validate(model_inc, model_gan, val_loader, label_transformer):
    device = next(model_inc.parameters()).device

    acts_real = np.zeros((len(val_loader.dataset), 1000))
    acts_fake = np.zeros((len(val_loader.dataset), 1000))

    pbar = tqdm(val_loader, desc='Running evaluation...', leave=False)
    for idx, (image, label) in enumerate(pbar):
        image = image.to(device)
        label = label.to(device)
        label = label_transformer.get_target(label).type(torch.float32)
        fake = model_gan.generate(image, label)

        _acts_real = model_inc(upscale_twice_tensor(image)).detach().cpu().numpy()
        _acts_fake = model_inc(upscale_twice_tensor(fake)).detach().cpu().numpy()

        acts_real[idx * val_loader.batch_size: idx * val_loader.batch_size + image.size(0)] = _acts_real
        acts_fake[idx * val_loader.batch_size: idx * val_loader.batch_size + image.size(0)] = _acts_fake

    return fid_distance_from_activations(acts_real, acts_fake)


def generate_batch_wandb(image, label, generator, target_attributes):
    generated_batches = [(image + 1) / 2]

    device = next(generator.parameters()).device
    for trg_idx, trg_attr in enumerate(target_attributes):
        old_column = label[:, trg_idx].clone()
        label[:, trg_idx] = 1.
        generated = (generator(image.to(device), label.to(device)).detach().cpu() + 1) / 2
        generated_batches.append(generated)
        label[:, trg_idx] = old_column

    grid_img = torchvision.utils.make_grid(torch.cat(generated_batches, dim=0), nrow=image.size(0))
    return grid_img.permute(1, 2, 0).numpy()


def train_model(config: Config, checkpoint: tp.Optional[dict] = None) -> None:
    train_params = config.training

    data_transforms = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = load_celeba(config.data.celeba.path, transforms=data_transforms)
    target_attributes = config.data.celeba.AttributeList
    original_attributes = dataset.attr_names
    label_transformer = LabelTransformerUpdated(original_attributes, target_attributes)

    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_params['batch_size'], shuffle=True)

    last_run_num = find_last_run(config.checkpoints.save_path)
    ckpt_save_path = Path(config.checkpoints.save_path) / f"run{last_run_num + 1}"
    os.mkdir(ckpt_save_path)

    # build_model
    model = StarGAN(
        lbl_features=label_transformer.label_dim,
        image_size=64,
        residual_block_number=6
    )

    optimizer_d = torch.optim.Adam(model.D.parameters(),
                                   lr=train_params['learning_rate_discriminator'],
                                   betas=train_params['adam_betas'])

    optimizer_g = torch.optim.Adam(model.G.parameters(),
                                   lr=train_params['learning_rate_generator'],
                                   betas=train_params['adam_betas'])

    optimizers = {
        'D': optimizer_d,
        'G': optimizer_g
    }

    start_epoch = 0

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        optimizer_to(optimizer_g, config.device)
        optimizer_to(optimizer_d, config.device)

        start_epoch = checkpoint['epoch']

    model.to(config.device)
    k = float(train_params['learning_rate_generator']) / (train_params['epochs_num'] - 10)
    lambda_lr = (lambda epoch: float(train_params['learning_rate_generator']) - k * (epoch - 10)
                 if epoch > 9 else float(train_params['learning_rate_generator']))

    best_fid_value = 1e6
    fid_calc_model = models.resnext50_32x4d(pretrained=True)
    fid_calc_model.to(config.device)

    test_im, test_labels = next(iter(val_loader))
    test_labels = label_transformer.get_target(test_labels).type(torch.float32)

    # train_loop
    for epoch_num in range(start_epoch, train_params['epochs_num']):
        for model_T in optimizers:
            for g in optimizers[model_T].param_groups:
                g['lr'] = lambda_lr(epoch_num)

        generated_val = generate_batch_wandb(test_im,
                                             test_labels,
                                             model.G,
                                             target_attributes)

        wandb.log({"Images/test_permutations": wandb.Image(generated_val)})

        if epoch_num > 10:
            train_params.lambda_rec_fake = 0.

        losses = train_epoch(train_loader,
                             model,
                             optimizers,
                             epoch_num + 1,
                             train_params,
                             label_transformer,
                             log_step=config.wandb.log_step)
        fid = validate(fid_calc_model, model, val_loader, label_transformer)

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_d': optimizers['D'].state_dict(),
            'optimizer_g': optimizers['G'].state_dict(),
            'epoch': epoch_num + 1,
            'fid_value': fid
        }

        save_checkpoint(Path(ckpt_save_path) / 'last.pth', checkpoint)

        if best_fid_value > fid:
            best_fid_value = fid
            save_checkpoint(Path(ckpt_save_path) / 'best.pth', checkpoint)

        wandb.log({"Validation/fid:": fid})
        wandb.log({"Model/lr": optimizer_d.param_groups[0]['lr']})
