'''training and val module.'''
import os
import time
from typing import Dict, Any
from tempfile import TemporaryDirectory

import torch
from torch.backends import cudnn
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm

cudnn.benchmark = True


# le dataloader de cette fonction est un dictionnaire comportant 2 dataloaders
def evalutrain_model(model,
                model_type: str,
                dataloaders,
                image_datasets,
                criterion,
                optimizer,
                scheduler,
                device,
                num_epochs=25):
    '''train and val function.'''

    since = time.time()

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(dataset_sizes, image_datasets['train'][2][0].size())
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        train_losses, train_accs = [], []
        val_losses, val_accs = [], []

        for epoch in tqdm(range(num_epochs)):
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)

                        if model_type == 'convnet':
                            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                            targets_one_hot = torch.zeros(labels.size(0), 6)  # Shape: (batch_size, 6)
                            targets_one_hot[torch.arange(labels.size(0)), labels] = 1
                            loss = criterion(logits, targets_one_hot.to(device))
                        else:
                            recon_x, mu, var = outputs
                            reconstruction_loss = criterion(recon_x, inputs)
                            kl_div = -torch.sum(1 + torch.log(var**2) - mu**2 - var**2)

                            loss = reconstruction_loss + kl_div

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