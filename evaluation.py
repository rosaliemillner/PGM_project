import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from attacks import fgsm_attack, pixel_attack, change_brightness, adjust_contrast
from attacks import pgd_attack, cw_attack


def plot_confusion_matrix(true_labels, pred_labels, attack_name=None):
    '''plot the confusion matrix.'''
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.colormaps['Blues'])
    # plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.savefig(f'./Inference/confusion_matrix_{attack_name}.pdf' if attack_name else './Inference/confusion_matrix.pdf')

    return cm


def test_evaluation(fit_model,
                    dataset,
                    device='cpu',
                    type_model='vae_gbz',
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
    logits = []

    if not attack:
        for image, label in tqdm(dataset):
            with torch.no_grad():
                if type_model == 'vae_gbz':
                    recon_x, log_prob_y, mu, logvar = fit_model(image.unsqueeze(0))
                    output = log_prob_y
                else:
                    output = fit_model(image.unsqueeze(0))
                logits.append(output.cpu().numpy())

                _, preds = torch.max(output, 1)
                probs_positive.append(F.softmax(output.cpu(), dim=1).flatten()[1].numpy())

                pred_labels.append(preds.cpu().numpy())
                true_labels.append(label)

            fit_model.train(mode=was_training)

    if attack:
        labels = [label for _, label in dataset]
        labels = labels[:len(adv_images)]
        for image, label in tqdm(zip(adv_images, labels)):
            with torch.no_grad():
                if type_model == 'vae_gbz':
                    recon_x, log_prob_y, mu, logvar = fit_model(image)
                    output = log_prob_y
                else:
                    output = fit_model(image.to(device))
                logits.append(output.cpu().numpy())
                _, preds = torch.max(output, 1)
                probs_positive.append(F.softmax(output.cpu(), dim=1).flatten()[1].numpy())

                pred_labels.append(preds.cpu().numpy())
                true_labels.append(label)

            fit_model.train(mode=was_training)

    return true_labels, pred_labels, probs_positive, logits


def plot_acc_train(train_losses,
                   train_accs,
                   val_losses,
                   val_accs,
                   type_model):

    idx = []
    for count in range(len(val_accs)):
        if val_accs[count] == np.max(val_accs):
            idx.append(count)

    plt.figure(figsize=(10, 3))

    if type_model == 'vae_gbz':
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, marker='o', linestyle='-', label='train loss')
        plt.legend()

        plt.subplot(1, 2, 1)
        plt.plot(val_losses, marker='o', linestyle='-', label='val loss')
        plt.legend()

    else:
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

    plt.savefig(f'./Inference/train_val_loss_acc_{type_model}.pdf')


def perform_attack(fit_model,
                   dataset,
                   model_type,
                   attack,
                   epsilon=0.05) -> list:
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
                                          epsilon=epsilon,
                                          model_type=model_type
                                          )
            adv_images.append(perturbed_image)
        elif attack == 'pixel':
            perturbed_image = pixel_attack(image.unsqueeze(0),
                                           label,
                                           fit_model,
                                           num_pixels=10,
                                           max_iter=100,
                                           type_model=model_type
                                           )
            adv_images.append(perturbed_image)
        elif attack == 'brightness':
            perturbed_image = change_brightness(image.unsqueeze(0), factor=1.3)
            adv_images.append(perturbed_image)
        elif attack == 'contrast':
            perturbed_image = adjust_contrast(image.unsqueeze(0))
            adv_images.append(perturbed_image)
        elif attack == 'pgd':
            perturbed_image = pgd_attack(image.unsqueeze(0),
                                         label,
                                         fit_model,
                                         model_type=model_type,
                                         epsilon=epsilon,
                                         )
            adv_images.append(perturbed_image)
        elif attack == 'cw':
            perturbed_image = cw_attack(image.unsqueeze(0),
                                        label,
                                        fit_model,
                                        model_type=model_type,
                                        device='cuda' if torch.cuda.is_available() else 'cpu'
                                        )
            adv_images.append(perturbed_image)
    return adv_images
