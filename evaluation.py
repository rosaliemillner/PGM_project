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
                    train_losses,
                    train_accs,
                    val_losses,
                    val_accs,
                    device='cpu',
                    type_model='vae'):
    """
    model.eval() mode, returns information to print for evaluate performances
    on a test set for example.
    """

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

    was_training = fit_model.training
    fit_model.eval()
    fit_model.cpu()

    true_labels = []
    pred_labels = []
    probs_positive = []

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

    return true_labels, pred_labels, probs_positive
