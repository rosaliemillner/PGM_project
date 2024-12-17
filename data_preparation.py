import os
import json

import torch
from torchvision import transforms
from torch.utils.data import Subset
from PIL import Image
from sklearn.model_selection import train_test_split


def generate_json_annotations(img_path, annotation_file):
    '''generates a json file for annotation'''
    try:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"folder {img_path} does not exists")
        annotations = []
        for idx, folder in enumerate(os.listdir(img_path)):
            print(f"folder name: {str(folder)}")
            label = idx
            for image_file in os.listdir(os.path.join(img_path, folder)):
                annotations.append({'filename': image_file, 'label': label})

        with open(annotation_file, mode='w') as f:
            json.dump(annotations, f, indent=4)

        print("Annotation process: annotations have been generated successfully and stored in 'annotations_classif.json'.")

    except FileNotFoundError as e:
        print(f" - Error: {e}")


def transform_data(resize_shape):
    '''
    Transform the data by resizing and normalizing it.
    '''
    transform = transforms.Compose([
        transforms.Resize(resize_shape),
        transforms.ToTensor()])
    return transform


class ImageDataset(torch.utils.data.Dataset):
    '''
    Class Dataset. Generate the dataset dictionnary
    for loaders in the expected format.
    '''
    def __init__(self, annotations_file, img_dir, transform=None):
        with open(annotations_file, mode='r', encoding='utf8') as f:
            annos = json.load(f)

        self.img_labels = annos
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        '''len of the dataset'''
        return len(self.img_labels)

    def __getitem__(self, idx):
        '''get item'''
        id = self.img_labels[idx]['filename']
        label = self.img_labels[idx]['label']

        for idx, name in enumerate(os.listdir(self.img_dir)):
            if idx != label:
                continue
            else:
                pil_image = Image.open(str(self.img_dir + name + '/' + id))  # pil format
                break

        if self.transform:
            image = self.transform(pil_image)  # tensor image
        else:
            image = transforms.ToTensor()(pil_image)

        return image, label


def train_val_dataset(dataset, val_split=0.2):
    '''
    Create a dictionnary of 2 datasets:
    One for training, and another for validation.
    '''
    train_idx, val_idx = train_test_split(list(range(len(dataset))),
                                          test_size=val_split)
    datasets = {}

    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)

    return datasets


def train_val_dataloader(datasets, batch_size):
    '''
    Create a dictionnary of 2 dataloaders:
    One for training, and another for validation.
    '''
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'],
                                                       batch_size=batch_size,
                                                       shuffle=True)
    dataloaders['val'] = torch.utils.data.DataLoader(datasets['val'],
                                                     batch_size=batch_size,
                                                     shuffle=True)
    return dataloaders
