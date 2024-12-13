import os
import json
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm

from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from attacks import fgsm_attack, more_pixel_attack


def plot_confusion_matrix(true_labels, pred_labels):
    '''plot the confusion matrix.'''
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.figure(figsize=(8,6))
    disp.plot(cmap=plt.colormaps['Blues'])
    # plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.savefig('./Inference/confusion_matrix.pdf')

    return cm


def test_evaluation(fit_model,
                    dataset,
                    device='cpu',
                    type_model='vae',
                    attack=False,
                    adv_images=None):
    """
    model.eval() mode, returns information to print for evaluate performances
    on a test set for example.
    """

    was_training = fit_model.training
    fit_model.eval()
    fit_model.to(device)

    true_labels = []
    pred_labels = []
    probs_positive = []

    if not attack:
        for image, label in tqdm(dataset):
            with torch.no_grad():
                if type_model == 'vae':
                    recon_x, log_prob_y, mu, logvar = fit_model(image.unsqueeze(0))
                    output = log_prob_y
                else:
                    output = fit_model(image.unsqueeze(0))
                _, preds = torch.max(output, 1)
                probs_positive.append(F.softmax(output.cpu(), dim=1).flatten()[1].numpy())

                pred_labels.append(preds.cpu().numpy())
                true_labels.append(label)

            fit_model.train(mode=was_training)

    if attack:
        labels = [label for _, label in dataset]
        for image, label in tqdm(zip(adv_images, labels)):
            with torch.no_grad():
                if type_model == 'vae':
                    recon_x, log_prob_y, mu, logvar = fit_model(image)
                    output = log_prob_y
                else:
                    output = fit_model(image)
                _, preds = torch.max(output, 1)
                probs_positive.append(F.softmax(output.cpu(), dim=1).flatten()[1].numpy())

                pred_labels.append(preds.cpu().numpy())
                true_labels.append(label)

            fit_model.train(mode=was_training)

    return true_labels, pred_labels, probs_positive


def plot_acc_train(train_losses,
                   train_accs,
                   val_losses,
                   val_accs):

    idx = []
    for count in range(len(val_accs)):
        if val_accs[count] == np.max(val_accs):
            idx.append(count)

    plt.figure(figsize=(10, 3))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, marker='o', linestyle='-', label='train loss')
    plt.legend()

    plt.subplot(1, 2, 1)
    plt.plot(val_losses, marker='o', linestyle='-', label='val loss')
    plt.ylim(0, 1)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, marker='o', linestyle='-', label='train acc')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accs, marker='o', linestyle='-', label='val acc')
    plt.axvline(x=int(idx[-1]), color='orange', linestyle='--', label='Max Val Accuracy')
    plt.legend()

    plt.savefig('./Inference/train_val_loss_acc.pdf')


def perform_attack(fit_model,
                   dataset,
                   model_type='vae',
                   attack='fgsm') -> list:
    """
    model.eval() mode, returns information to print for evaluate performances
    on a test set for example.
    """
    adv_images = []
    for image, label in tqdm(dataset):
        if attack == 'fsgm':
            perturbed_image = fgsm_attack(image.unsqueeze(0),
                                          label,
                                          fit_model,
                                          model_type=model_type,
                                          epsilon=0.1
                                          )
            adv_images.append(perturbed_image)
        else:
            perturbed_image = more_pixel_attack(image.unsqueeze(0),
                                                label,
                                                fit_model,
                                                num_pixels=3,
                                                max_iter=100
                                                )
            adv_images.append(perturbed_image)
    return adv_images
