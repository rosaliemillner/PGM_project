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

        # Load the pre-trained ResNet-34 model
        self.resnet = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')

        self.resnet.conv1 = nn.Conv2d(
                    in_channels=1,  # grayscale
                    out_channels=self.resnet.conv1.out_channels,
                    kernel_size=self.resnet.conv1.kernel_size,
                    stride=self.resnet.conv1.stride,
                    padding=self.resnet.conv1.padding
                )

        # Freeze all parameters by default
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace the fully connected layer with a new one
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 6)

        # Unfreeze the last 'fine_tune_layers' layers
        for param in self.resnet.layer4[-2:].parameters():
            param.requires_grad = True
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        '''forward path'''
        return self.resnet(x)


# Encoder for q(z|x)
class Encoder(nn.Module):
    def __init__(self, input_channels, latent_dim, image_size):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)

        # Compute the size of the flattened feature map after convolutions
        conv_output_size = image_size // 4  # Assuming two layers of stride=2
        self.flatten_size = 64 * conv_output_size * conv_output_size

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar



# Decoder for p(x|z)
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels, image_size):
        super(Decoder, self).__init__()

        # Taille finale de l'image
        self.image_size = image_size

        # Calcul de la taille intermédiaire (avant déconvolutions)
        self.intermediate_size = image_size // 4  # Taille après deux convolutions transposées avec stride=2

        # Taille aplatie en entrée
        self.flattened_size = 64 * self.intermediate_size * self.intermediate_size

        # Définition des couches
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, self.flattened_size)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # Upsample
        self.deconv2 = nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)  # Upsample

    def forward(self, z):
        h = F.relu(self.fc1(z))  # Couche dense
        h = F.relu(self.fc2(h))  # Expandir vers l'entrée aplatie
        h = h.view(h.size(0), 64, self.intermediate_size, self.intermediate_size)  # Reshape
        h = F.relu(self.deconv1(h))  # Première convolution transposée
        return torch.sigmoid(self.deconv2(h))  # Deuxième convolution transposée


# Classifier for p(y|z)

class Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return F.log_softmax(self.fc2(h), dim=1)


# GBZ Model

class GBZ(nn.Module):
    def __init__(self, input_channels, latent_dim, num_classes, output_channels, image_size):
        super(GBZ, self).__init__()
        self.encoder = Encoder(input_channels, latent_dim, image_size)
        self.decoder = Decoder(latent_dim, output_channels, image_size)
        self.classifier = Classifier(latent_dim, num_classes)

    def forward(self, x):
        # Encode to latent space
        mu, logvar = self.encoder(x)

        # Reparameterize to get z
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # Reparameterization trick

        # Decode x and classify y
        recon_x = self.decoder(z)
        log_probs_y = self.classifier(z)

        return recon_x, log_probs_y, mu, logvar
