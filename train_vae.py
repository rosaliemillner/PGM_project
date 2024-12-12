import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import GBZ model and ELBO loss
from models import GBZ
from utils import elbo_loss


# Hyperparameters
input_channels = 1  # MNIST is grayscale
output_channels = 1  # Output is grayscale
latent_dim = 20  # Dimension of latent space
num_classes = 10  # Number of classes (MNIST)
batch_size = 64
learning_rate = 1e-3
num_epochs = 10
image_size = 28  # For MNIST, images are 28x28
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset and DataLoader
transform = transforms.Compose([
    transforms.ToTensor()  # Converts to range [0, 1]
])

train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, optimizer
model = GBZ(input_channels, latent_dim, num_classes, output_channels, image_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        # Forward pass
        recon_x, log_probs_y, mu, logvar = model(data)

        # Compute loss
        loss, recon_loss, kl_div = elbo_loss(recon_x, data, mu, logvar)

        train_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print average loss for this epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss / len(train_loader)}")

# Save the trained model
os.makedirs('./checkpoints', exist_ok=True)
torch.save(model.state_dict(), './checkpoints/vae_GBZ.pth')
print("Model saved to ./checkpoints/vae_GBZ.pth")
