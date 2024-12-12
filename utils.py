# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def elbo_loss(recon_x, x, mu, logvar, num_samples=1):
    """
    Compute the ELBO loss based on Kingma and al 2014.

    """

    # 1. Divergence KL
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    # 2. Reconstruction log-likelihood (Monte Carlo approximation if num_samples > 1)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)

    # Total ELBO loss
    elbo = kl_div + recon_loss
    return elbo, recon_loss, kl_div
