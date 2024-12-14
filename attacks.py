import os
import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from tqdm import tqdm
from utils import elbo_loss


def fgsm_attack(image, label, model, epsilon, model_type='vae'):
    """
    Parameters:
        image (torch.Tensor): input image tensor (1, H, W)
        label (int): ground-truth label for the image
        model (torch.nn.Module): the model to attack
        epsilon (float): perturbation magnitude

    Returns: perturbed_image (torch.Tensor): Adversarially perturbed image
    """
    image.requires_grad = True

    if model_type == 'vae':
        recon_x, log_prob_y, mu, logvar = model(image)
        # Compute loss
        loss, _, _, _ = elbo_loss(image, recon_x, torch.LongTensor([label]), log_prob_y, mu, logvar)
        logits = log_prob_y
    else:
        outputs = model(image)
        logits = outputs

        loss = nn.CrossEntropyLoss()(logits, torch.tensor([label]))

    model.zero_grad()
    loss.backward()

    image_grad = image.grad.data

    perturbation = epsilon * image_grad.sign()  # creates perturbation
    perturbed_image = image + perturbation

    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image


def pixel_attack(image, label, model, num_pixels=10, max_iter=100, type_model='vae'):
    """
    Parameters:
        image (torch.Tensor): input image tensor (1,H,W)
        label (int): ground-truth label for the image
        model
        max_iter
    Returns adversarially perturbed image (torch.Tensor)
    """
    image_np = image.squeeze().numpy()

    def apply_perturbation(params):  # apply the perturbation to the image
        # x, y, value = params
        # x, y = int(x), int(y)
        perturbed = image_np.copy()
        for i in range(num_pixels):
            x = int(params[i * 3])  # x position of the pixel
            y = int(params[i * 3 + 1])  # y position of the pixel
            val = params[i * 3 + 2]  # perturbation value for the pixel
            perturbed[x, y] = val
        return perturbed

    def objective_function(params):
        perturbed = apply_perturbation(params)
        perturbed_tensor = torch.tensor(perturbed[None, :, :], dtype=torch.float32).unsqueeze(0)
        if type_model == 'vae':
            _, log_prob_y, _, _ = model(perturbed_tensor)
            confidence = torch.exp(log_prob_y)[0, label].item()  # Extract confidence
        elif type_model == 'convnet':
            with torch.no_grad():
                confidence = torch.softmax(model(perturbed_tensor), dim=1)[0, label].item()
        return -confidence  # We minimize confidence (maximize loss in optimization)

    # Optimization bounds for pixel location and RGB values
    bounds = []
    for _ in range(num_pixels):
        bounds.extend([
            (0, image_np.shape[0] - 1),  # x (0, 27)
            (0, image_np.shape[1] - 1),  # y (0, 27)
            (0, 1)                       # grayscale value (0, 1)
        ])

    result = differential_evolution(objective_function, bounds, maxiter=max_iter, disp=True, workers=1)

    perturbed_image_np = apply_perturbation(result.x)
    # perturbed_image_np = apply_perturbation(result.x)
    perturbed_image = torch.tensor(perturbed_image_np[None, :, :], dtype=torch.float32)

    return perturbed_image.unsqueeze(0)




# new one


# Function to evaluate model accuracy on perturbed image
def perturb_and_classify(model, image, pixels, device, model_type):
    # Clone the original image
    perturbed_image = image.clone()
    # Apply perturbation in grayscale
    for pixel in pixels:
        x, y, gray_value = pixel  # Only grayscale value
        perturbed_image[0, int(y), int(x)] = gray_value  # Single channel for grayscale

    # Classify perturbed image
    with torch.no_grad():
        if model_type == 'vae_gbz':
            _, y_pred, _, _ = model(perturbed_image.unsqueeze(0).to(device))
            pred_label = torch.argmax(y_pred, dim=1).item()
        elif model_type == 'convnet':
            outputs = model(perturbed_image.unsqueeze(0).to(device))
            pred_label = torch.argmax(outputs, dim=1).item()
    return pred_label


# Function to evaluate attack objective
def attack_loss(pixel_params, model, image, label, device, model_type):
    # Decode pixel params into pixel positions and values
    pixels = pixel_params.reshape(-1, 3)  # x, y, grayscale value
    pred_label = perturb_and_classify(model, image, pixels, device, model_type)
    return 1.0 if pred_label == label else 0.0  # Minimize this (success = 0)


# Perform One Pixel Attack
def one_pixel_attack(model, image, label, num_pixels, bounds, device, model_type):
    # Bounds for optimization (x, y positions and grayscale values)
    bounds = [(0, image.shape[2] - 1),  # x
              (0, image.shape[1] - 1),  # y
              (0, 1)] * num_pixels      # Grayscale value (0 to 1)

    # Optimize pixel locations and grayscale values
    result = differential_evolution(
        attack_loss,
        bounds,
        args=(model, image, label, device, model_type),
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7
    )

    # Extract optimized pixels
    optimal_pixels = result.x.reshape(-1, 3)  # x, y, grayscale
    return optimal_pixels, result.fun


# Visualize attack
def visualize_attack(original_image, perturbed_image, class_names, original_label,
                     attacked_label):
    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image.squeeze(0).cpu().numpy(), cmap='gray')
    plt.title(f"Original Image\nLabel: {class_names[original_label]}")
    plt.axis('off')

    # Perturbed image
    plt.subplot(1, 2, 2)
    plt.imshow(perturbed_image.squeeze(0).cpu().numpy(), cmap='gray')
    plt.title(f"Perturbed Image\nLabel: {class_names[attacked_label]}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# Apply One Pixel Attack on a sample
def run_one_pixel_attack(model, data_loader, class_names, num_pixels=1,
                         model_type='convnet',
                         device='cuda'):
    model = model.to(device)
    model.eval()

    perturbed_images = []
    # Select a single sample
    for image, label in tqdm(data_loader):
        image, label = image[0].to(device), label[0].item()

        # Perform attack
        optimal_pixels, _ = one_pixel_attack(
            model, image, label, num_pixels, bounds=None, device=device, model_type=model_type
        )

        # Apply perturbation to the image in grayscale
        perturbed_image = image.clone()
        for pixel in optimal_pixels:
            x, y, gray_value = pixel
            perturbed_image[0, int(y), int(x)] = gray_value  # Grayscale perturbation

        # Get predictions
        perturbed_image.unsqueeze(0)
        perturbed_images.append(perturbed_image)
    return perturbed_images
