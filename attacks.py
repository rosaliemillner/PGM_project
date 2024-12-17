import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from tqdm import tqdm
from utils import elbo_loss


def fgsm_attack(image, label, model, epsilon, model_type='vae_gbz', device='cpu'):
    if model_type == 'vae_gbz':
        device = 'cpu'
    else:
        device = device

    image.requires_grad = True

    if model_type == 'vae_gbz':
        recon_x, log_prob_y, mu, logvar = model(image.to(device))
        # Compute loss
        loss, _, _, _ = elbo_loss(image, recon_x, torch.LongTensor([label]), log_prob_y, mu, logvar)
        logits = log_prob_y
    else:
        outputs = model(image.to(device))
        logits = outputs

        loss = nn.CrossEntropyLoss()(logits, torch.tensor([label], device=device))

    model.zero_grad()
    loss.backward()

    image_grad = image.grad.data

    perturbation = epsilon * image_grad.sign()  # creates perturbation
    perturbed_image = image + perturbation

    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image.cpu()


# pgd attack
def pgd_attack(image, label, model, epsilon=0.1, alpha=0.01, num_iter=40, model_type='vae_gbz', device='cpu'):
    if model_type == 'vae_gbz':
        device = 'cpu'
    else:
        device = device
    model.to(device)
    model.eval()

    original_image = image.clone().to(device)
    perturbed_image = image.clone().detach().to(device).requires_grad_(True)

    for _ in range(num_iter):
        if model_type == 'vae_gbz':
            recon_x, log_prob_y, mu, logvar = model(perturbed_image)
            loss, _, _, _ = elbo_loss(perturbed_image, recon_x, torch.LongTensor([label], device=device), log_prob_y, mu, logvar)
            logits = log_prob_y
        else:
            outputs = model(perturbed_image)
            logits = outputs

            loss = nn.CrossEntropyLoss()(logits, torch.tensor([label], device=device))

        model.zero_grad()
        loss.backward()

        image_grad = perturbed_image.grad.data
        perturbed_image = perturbed_image + alpha * image_grad.sign()
        perturbation = torch.clamp(perturbed_image - original_image, -epsilon, epsilon)
        perturbed_image = torch.clamp(original_image + perturbation, 0, 1).detach_().requires_grad_(True)

    return perturbed_image.cpu()


# pixel attack
def pixel_attack(image, label, model, num_pixels=10, max_iter=100, type_model='vae'):
    image_np = image.squeeze().numpy()

    def apply_perturbation(params):
        perturbed = image_np.copy()
        for i in range(num_pixels):
            x = int(params[i * 3])
            y = int(params[i * 3 + 1])
            val = params[i * 3 + 2]
            perturbed[x, y] = val
        return perturbed

    def objective_function(params):
        perturbed = apply_perturbation(params)
        perturbed_tensor = torch.tensor(perturbed[None, :, :], dtype=torch.float32).unsqueeze(0)
        if type_model == 'vae_gbz':
            _, log_prob_y, _, _ = model(perturbed_tensor)
            confidence = torch.exp(log_prob_y)[0, label].item()
        elif type_model == 'convnet':
            with torch.no_grad():
                confidence = torch.softmax(model(perturbed_tensor), dim=1)[0, label].item()
        return -confidence

    bounds = []
    for _ in range(num_pixels):
        bounds.extend([
            (0, image_np.shape[0] - 1),
            (0, image_np.shape[1] - 1),
            (0, 1)
        ])

    result = differential_evolution(objective_function, bounds, maxiter=max_iter, disp=True, workers=1)

    perturbed_image_np = apply_perturbation(result.x)
    perturbed_image = torch.tensor(perturbed_image_np[None, :, :], dtype=torch.float32)

    return perturbed_image.unsqueeze(0)


# Function to evaluate model accuracy on a perturbed image
def perturb_and_classify(model, image, pixels, device, model_type):
    perturbed_image = image.clone()
    for pixel in pixels:
        x, y, gray_value = pixel
        perturbed_image[0, int(y), int(x)] = gray_value

    with torch.no_grad():
        if model_type == 'vae_gbz':
            _, y_pred, _, _ = model(perturbed_image.unsqueeze(0).to(device))
            confidence = torch.exp(y_pred)[0]
        elif model_type == 'convnet':
            outputs = model(perturbed_image.unsqueeze(0).to(device))
            confidence = torch.softmax(outputs, dim=1)[0]
    return confidence


def attack_loss(pixel_params, model, image, label, device, model_type):
    pixels = pixel_params.reshape(-1, 3)
    confidence = perturb_and_classify(model, image, pixels, device, model_type)
    return -confidence[label].item()


def one_pixel_attack(model, image, label, num_pixels, bounds, device, model_type):
    bounds = [(0, image.shape[2] - 1),
              (0, image.shape[1] - 1),
              (0, 1)] * num_pixels

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

    optimal_pixels = result.x.reshape(-1, 3)
    return optimal_pixels, result.fun


# Visualize attack
def visualize_attack(original_image, perturbed_image, class_names, original_label,
                     attacked_label):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image.squeeze(0).cpu().numpy(), cmap='gray')
    plt.title(f"Original Image\nLabel: {class_names[original_label]}")
    plt.axis('off')

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
    for image, label in tqdm(data_loader):
        image, label = image[0].to(device), label[0].item()
        optimal_pixels, _ = one_pixel_attack(
            model, image, label, num_pixels, bounds=None, device=device, model_type=model_type
        )
        perturbed_image = image.clone()
        for pixel in optimal_pixels:
            x, y, gray_value = pixel
            perturbed_image[0, int(y), int(x)] = gray_value

        perturbed_image.unsqueeze(0)
        perturbed_images.append(perturbed_image)
    return perturbed_images


# attack with brightness
def change_brightness(image, factor=1.3):
    bright_image = torch.clamp(image * factor, 0, 1)


# Function to adjust contrast
def adjust_contrast(image, factor=4):
    mean = image.mean(dim=(-1, -2, -3), keepdim=True)
    contrast_image = torch.clamp((image - mean) * factor + mean, 0, 1)
    return contrast_image


# CW attack
def cw_attack(image, label, model, c=100, lr=0.001, max_iter=100, model_type='vae_gbz', device='cuda'):
    model.to(device)
    image = image.to(device)
    perturbed_image = image.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([perturbed_image], lr=lr)

    for _ in range(max_iter):
        if model_type == 'vae_gbz':
            recon_x, log_prob_y, mu, logvar = model(perturbed_image)
            logits = log_prob_y
        else:
            logits = model(perturbed_image)

        real = logits[0, label]
        other = torch.max(logits[0, torch.arange(len(logits[0])) != label])
        loss = c * (other - real).clamp(min=0) + torch.norm(perturbed_image - image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        perturbed_image.data = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image.detach().cpu()
