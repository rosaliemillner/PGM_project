import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from utils import elbo_loss


def fgsm_attack(image, label, model, model_type='vae', epsilon=0.1):
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
        loss, _, _ = elbo_loss(recon_x, image, mu, logvar)
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


def more_pixel_attack(image, label, model, num_pixels=3, max_iter=100):
    """
    Parameters:
        image (torch.Tensor): input image tensor (1,H,W)
        label (int): ground-truth label for the image
        model
        max_iter
    Returns adversarially perturbed image (torch.Tensor)
    """
    image_np = image.squeeze().numpy()

    def predict(input_image): #predicts the class using the model
        input_tensor = torch.tensor(input_image[None, None, :, :]).float()
        input_tensor = transforms.Resize((224, 224))(input_tensor)  # for compatibility with ResNet
        input_tensor = input_tensor.repeat(1, 3, 1, 1)  # Convert our grayscale (1, H, W) -> (3, H, W)  #for compatibility with ResNet 

        with torch.no_grad():
            output = model(input_tensor)
        return torch.argmax(output).item()

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

    def objective_function(params): #objective function for optimization (misclassification confidence)
        perturbed = apply_perturbation(params)
        predicted_label = predict(perturbed)
        return 1.0 if predicted_label == label else 0.0

    # Optimization bounds for pixel location and RGB values
    bounds = []
    for _ in range(num_pixels):
        bounds.extend([
            (0, image_np.shape[0] - 1),  # x (0, 27)
            (0, image_np.shape[1] - 1),  # y (0, 27)
            (0, 1)                       # grayscale value (0, 1)
        ])

    result = differential_evolution(objective_function, bounds, maxiter=max_iter, disp=True)

    perturbed_image_np = apply_perturbation(result.x)
    perturbed_image = torch.tensor(perturbed_image_np[None, :, :])

    return perturbed_image
