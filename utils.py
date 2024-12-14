# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


# Loss Function (Kingma et al., 2014)
def elbo_loss(x, x_recon, y, y_pred, mu, logvar):
    # Reconstruction loss (binary cross-entropy)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Classification loss (cross-entropy)
    class_loss = F.cross_entropy(y_pred, y)
    # Total loss
    return recon_loss + kl_loss + class_loss, recon_loss, kl_loss, class_loss
