import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm


# Loss Function (Kingma et al., 2014)
def elbo_loss(x, x_recon, y, y_pred, mu, logvar):
    # Reconstruction loss)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Classification loss
    class_loss = F.cross_entropy(y_pred, y)
    # Total loss
    return recon_loss + kl_loss + class_loss, recon_loss, kl_loss, class_loss


# plot test distribution of latent variable z with PCA
def plot_latent_space(model, test_loader, device):
    model.eval()
    latent_space = []
    labels = []
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x = x.to(device)
            z = model.get_z(x.unsqueeze(0))
            latent_space.append(z.cpu().numpy().squeeze(0))
            labels.append(y)
    print(np.array(latent_space).shape)
    labels = np.array(labels)
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(latent_space)
    plt.figure(figsize=(6, 4))
    plt.scatter(z_pca[:, 0], z_pca[:, 1], c=labels, s=5, marker='x', cmap='tab10')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('./Inference/latent_space_pca.pdf')
    plt.show()
