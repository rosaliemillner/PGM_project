import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class FineTunedResNet18(nn.Module):
    """
    Fine tuning a ResNet-18.
    """
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')

        # Change the first layer to accept to handle grayscale images
        self.resnet.conv1 = nn.Conv2d(
                    in_channels=1,  # grayscale
                    out_channels=self.resnet.conv1.out_channels,
                    kernel_size=self.resnet.conv1.kernel_size,
                    stride=self.resnet.conv1.stride,
                    padding=self.resnet.conv1.padding
                )

        # freeze parameters when fine-tuning
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace the fully connected layer with a new one,
        # corresponding to our number of classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 6)

        # Unfreeze the last layers to learn new classification task
        for param in self.resnet.layer4[-2:].parameters():
            param.requires_grad = True
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)

    def get_z(self, x):
        return self.resnet.avgpool(x)


# Encoder: q(z|x)
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)  # Adapté pour 1 canaux (RGB)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # Output: (64, 17, 17)
        self.fc1 = nn.Linear(64 * 17 * 17, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# Decoder: p(x|z)
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 64 * 17 * 17)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # Output: (32, 34, 34)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)  # Adapté pour 1 canaux (RGB)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = z.view(z.size(0), 64, 17, 17)
        z = F.relu(self.deconv1(z))
        z = torch.sigmoid(self.deconv2(z))  # Pixel values in [0, 1]
        return z


# Classifier: p(y|z)
class Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = self.fc2(z)
        return z


# VAE Model
class GBZ(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(GBZ, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.classifier = Classifier(latent_dim, num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        self.z = z
        # Decode
        x_recon = self.decoder(z)
        # Classify
        y_pred = self.classifier(z)

        return x_recon, y_pred, mu, logvar

    def get_z(self, x):
        self.forward(x)
        return self.z
