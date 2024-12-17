'''training and val module.'''
import os
import time
from tempfile import TemporaryDirectory

import torch
from torch.backends import cudnn
from tqdm import tqdm

from utils import elbo_loss
cudnn.benchmark = True


# le dataloader dans cette fonction est un dictionnaire comportant 2 dataloaders
def evalutrain_model(model,
                     model_type: str,
                     dataloaders,
                     image_datasets,
                     criterion,
                     optimizer,
                     scheduler,
                     device,
                     num_epochs):
    '''train and val function.'''

    since = time.time()

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(dataset_sizes, image_datasets['train'][2][0].size())
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, f'best_model_params_{model_type}.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        train_losses, train_accs = [], []
        val_losses, val_accs = [], []

        for epoch in tqdm(range(num_epochs)):
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        if model_type == 'convnet':
                            outputs = model(inputs)
                            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                            targets_one_hot = torch.zeros(labels.size(0), 6)  # Shape: (batch_size, 6)
                            targets_one_hot[torch.arange(labels.size(0)), labels] = 1
                            loss = criterion(logits, targets_one_hot.to(device))

                        elif model_type == 'vae_gbz':
                            # Forward pass
                            recon_x, log_probs_y, mu, logvar = model(inputs)
                            # Compute loss
                            loss, _, _, _ = elbo_loss(inputs, recon_x, labels, log_probs_y, mu, logvar)

                            logits = log_probs_y

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(logits, 1)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                # for train AND val:
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if phase == 'train':
                    train_losses.append(epoch_loss)
                    train_accs.append(epoch_acc.cpu())

                if phase == 'val':
                    val_losses.append(epoch_loss)
                    val_accs.append(epoch_acc.cpu())

                # copy the model
                if phase == 'val' and epoch_acc >= best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    torch.save(model.state_dict(), best_model_params_path)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f} during the {best_epoch}th epoch.')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))

    return model, train_losses, train_accs, val_losses, val_accs
