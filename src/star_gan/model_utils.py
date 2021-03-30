import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad
from scipy import linalg
from tqdm.auto import tqdm

from src.common_utils.utils import permute_labels


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


# class DiscriminatorLoss:
#     def __init__(self,
#                  lambda_cls,
#                  lambda_rec):
#         self.cls_loss = F.binary_cross_entropy_with_logits
#
#     def _get_label_classification_loss(self, cls_out, cls_true):
#         return self.cls_loss(cls_out, cls_true)
#
#     def _get_adversarial_loss(self, fake, real):
#         return real.mean() - fake.mean()
#
#     def calc_loss(self, discriminator, real_im, fake_im):
#         real_out, _ = discriminator(real_im)
#         fake_out, _ = discriminator(fake_im)
#         L_adv = self._get_adversarial_loss(fake_out, real_out)
#
#         cls_loss = self._get_label_classification_loss(classification_out, cls_true)
#         return cls_loss


def get_description(desc, epoch, d_cls, d_adv, d_gp, g_cls, g_adv, g_rec):
    return desc.format(epoch, g_cls, g_adv, g_rec, d_cls, d_adv, d_gp)


def train_epoch(train_loader, model, optimizers, epoch_num, config, label_transformer, log=False):
    model.train()
    iter_idx = 0

    optimizer_d = optimizers['D']
    optimizer_g = optimizers['G']

    device = next(model.parameters())

    losses = {
        "D": {
            'adversarial': [],
            'classification': [],
            'gp_loss': []
        },
        "G": {
            'adversarial': [],
            'reconstruction': [],
            'classification': []
        }
    }

    description = "Epoch: {}: Loss G: cls {:.4f}, adv {:.4f}, rec {:.4f};\n \
                    Loss D: cls {:.4f}, adv {:.4f}, gp {:.4f}"

    pbar = tqdm(train_loader, leave=False, desc=description.format(epoch_num, 0, 0, 0, 0, 0, 0), ncols=850)

    g_adv_loss = torch.tensor([1000])
    g_cls_loss = torch.tensor([1000])
    g_rec_loss = torch.tensor([1000])

    gp_loss = torch.tensor([0])

    for image, label in pbar:
        image = image.to(device)
        label = label.to(device)

        iter_idx += 1
        true_labels = label_transformer.get_one_hot(label).type(torch.float32)
        fake_labels = permute_labels(true_labels)

        image_fake = model.forward_g(image, fake_labels).detach()
        fake_patch_out, fake_cls_out = model.forward_d(image_fake)
        real_patch_out, real_cls_out = model.forward_d(image)

        d_loss_adv = - real_patch_out.mean() + fake_patch_out.mean()
        d_loss_cls = F.binary_cross_entropy_with_logits(real_cls_out, true_labels.clone())

        # gp_loss = compute_gradient_penalty(model.D, image, image_fake)

        d_loss = d_loss_adv + config['lambda_cls'] * d_loss_cls  # + config['lambda_gp'] * gp_loss
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        losses['D']['adversarial'].append(d_loss_adv.item())
        losses['D']['gp_loss'].append(gp_loss.item())
        losses['D']['classification'].append(d_loss_cls.item())

        pbar.set_description(get_description(description,
                                             epoch_num,
                                             d_loss_cls.item(),
                                             d_loss_adv.item(),
                                             gp_loss.item(),
                                             g_cls_loss.item(),
                                             g_adv_loss.item(),
                                             g_rec_loss.item()))

        if iter_idx % config['generator_step'] == 0:
            image_fake = model.forward_g(image, fake_labels)
            image_reconstructed = model.forward_g(image_fake, true_labels)

            g_rec_loss = torch.mean(torch.abs(image - image_reconstructed))
            fake_patch_out, cls_fake = model.forward_d(image_fake)

            g_cls_loss = F.binary_cross_entropy_with_logits(cls_fake, fake_labels.clone())
            g_adv_loss = -fake_patch_out.mean()

            optimizer_g.zero_grad()
            loss = g_adv_loss + config['lambda_cls'] * g_cls_loss + config['lambda_rec'] * g_rec_loss
            loss.backward()
            optimizer_g.step()

            losses['G']['adversarial'].append(g_adv_loss.item())
            losses['G']['reconstruction'].append(g_rec_loss.item())
            losses['G']['classification'].append(g_cls_loss.item())

            pbar.set_description(get_description(description,
                                                 epoch_num,
                                                 d_loss_cls.item(),
                                                 d_loss_adv.item(),
                                                 gp_loss.item(),
                                                 g_cls_loss.item(),
                                                 g_adv_loss.item(),
                                                 g_rec_loss.item()))

        if iter_idx % 100 == 0 and log:
            for type_m in losses:
                for loss_type in losses[type_m]:
                    wandb.log({
                        type_m + '/' + loss_type: np.mean(losses[type_m][loss_type][-100:])
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


def interpolate_im(im_tensor):
    return F.interpolate(im_tensor, scale_factor=2, mode='bilinear', align_corners=False)


@torch.no_grad()
def validate(model_inc, model_gan, val_loader):
    device = next(model_inc.parameters()).device

    acts_real = np.zeros((len(val_loader.dataset), 1000))
    acts_fake = np.zeros((len(val_loader.dataset), 1000))

    for idx, (image, label) in tqdm(enumerate(val_loader), desc='Running evaluation...', leave=False):
        image = image.to(device)
        label = label.to(device)
        label = label_transformer.get_one_hot(label).type(torch.float32)
        fake = model_gan.generate(image, label)

        _acts_real = model_inc(interpolate_im(image)).detach().cpu().numpy()
        _acts_fake = model_inc(interpolate_im(fake)).detach().cpu().numpy()

        acts_real[idx * val_loader.batch_size: idx * val_loader.batch_size + image.size(0)] = _acts_real
        acts_fake[idx * val_loader.batch_size: idx * val_loader.batch_size + image.size(0)] = _acts_fake

    return fid_distance_from_activations(acts_real, acts_fake)

