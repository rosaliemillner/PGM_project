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


# Encoder for q(z|x,y)
class Encoder(nn.Module):
    def __init__(self, input_channels, num_classes, latent_dim, image_size):
        super(Encoder, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calcul de la taille après les convolutions
        conv_output_size = image_size // 8  # Réduction par max-pooling (3 couches)
        flattened_size = 64 * conv_output_size * conv_output_size

        # MLP layers
        self.fc1 = nn.Linear(flattened_size + num_classes, 500)  # +num_classes pour inclure y
        self.fc2 = nn.Linear(500, 500)

        # Paramètres pour la distribution latente
        self.fc_mean = nn.Linear(500, latent_dim)  # Moyenne
        self.fc_logvar = nn.Linear(500, latent_dim)  # Log-variance

    def forward(self, x, y):
        # Passer x dans les couches convolutionnelles
        h = F.relu(self.conv1(x))
        h = self.pool(h)
        h = F.relu(self.conv2(h))
        h = self.pool(h)
        h = F.relu(self.conv3(h))
        h = self.pool(h)

        # Flatten l'image
        h = h.view(h.size(0), -1)

        # Concatenation de x avec y (one-hot encoding)
        if y.dim() == 1:  # Si y est un indice
            y = F.one_hot(y, num_classes=self.fc1.in_features - h.size(1)).float()
        h = torch.cat((h, y), dim=1)

        # Passer dans le MLP
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))

        # Calcul des moyennes et des log-variances
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)

        return mean, logvar


# Decoder for p(x|z)
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels, image_size):
        super(Decoder, self).__init__()

        # Taille finale de l'image
        self.image_size = image_size

        # Calcul de la taille intermédiaire (avant les déconvolutions)
        self.intermediate_size = image_size // 4  # Taille après deux convolutions transposées avec stride=2

        # Taille aplatie en entrée
        self.flattened_size = 64 * self.intermediate_size * self.intermediate_size

        # MLP avec 2 couches cachées de 500 unités chacune
        self.fc1 = nn.Linear(latent_dim, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, self.flattened_size)

        # Réseau deconvolutionnel
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, output_channels, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, z):
        # MLP étapes
        h = F.relu(self.fc1(z))  # Première couche cachée
        h = F.relu(self.fc2(h))  # Deuxième couche cachée
        h = F.relu(self.fc3(h))  # Projection vers la taille aplatie

        # Reshape vers un tenseur 4D pour le réseau deconvolutionnel
        h = h.view(h.size(0), 64, self.intermediate_size, self.intermediate_size)

        # Deconvolutions
        h = F.relu(self.deconv1(h))  # Première déconv
        h = F.relu(self.deconv2(h))  # Deuxième déconv
        return torch.sigmoid(self.deconv3(h))  # Dernière déconv avec activation sigmoïde


# Classifier for p(y|z)
class Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Classifier, self).__init__()
        self.hidden = nn.Linear(latent_dim, 500)  # Couche cachée avec 500 unités
        self.output = nn.Linear(500, num_classes)  # Couche de sortie

    def forward(self, z):
        h = F.relu(self.hidden(z))  # Activation ReLU après la couche cachée
        return F.log_softmax(self.output(h), dim=1)  # Log-softmax sur la couche de sortie


# GBZ Model
class GBZ(nn.Module):
    def __init__(self, input_channels, latent_dim, num_classes, output_channels, image_size):
        super(GBZ, self).__init__()
        self.encoder = Encoder(input_channels, latent_dim, num_classes, image_size)
        self.decoder = Decoder(latent_dim, output_channels, image_size)
        self.classifier = Classifier(latent_dim, num_classes)

    def forward(self, x, y):
        # Encode to latent space
        mu, logvar = self.encoder(x, y)

        # Reparameterize to get z
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # Reparameterization trick

        # Decode x and classify y
        recon_x = self.decoder(z)
        log_probs_y = self.classifier(z)

        return recon_x, log_probs_y, mu, logvar
