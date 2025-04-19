import os
import zipfile

'''
def setup_data():
    dataset_path = "C:/Users/Sonu/Desktop/nature_12k"
    zip_path = os.path.join(dataset_path, "inaturalist_12K.zip")
    extract_path = os.path.join(dataset_path, "inaturalist_12K")

    if not os.path.exists(extract_path):
        print("Unzipping dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        print(f"Dataset unzipped to {dataset_path}")
    else:
        print("Data source already available. Skipping unzip.")

'''

import torch
import torch.nn as nn

def img_size(in_size, kernel_size, padding, stride):
    return (in_size - kernel_size + 2 * padding) // stride + 1

class CNN_Scratch(nn.Module):
    def __init__(self,
                 in_channels=3,
                 num_filters=[32, 64, 128, 256, 512],
                 filter_size=[3, 3, 5, 5, 7],
                 activation=nn.ReLU(),
                 stride=1,
                 padding=1,
                 pool_size=(2, 2),
                 fc_dims=[512],
                 num_classes=10,
                 dropout=0.5,
                 batch_norm='Yes'):
        super(CNN_Scratch, self).__init__()

        self.activation = activation
        self.batch_norm = batch_norm
        self.pool = nn.MaxPool2d(pool_size, stride=2)

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.drop_layers = nn.ModuleList()

        in_ch = in_channels
        current_size = 224  # assuming input image size is 224x224

        # Convolution layers....................................
        for i in range(len(num_filters)):
            out_ch = num_filters[i]
            k = filter_size[i]

            self.conv_layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=padding)
            )
            self.drop_layers.append(nn.Dropout2d(dropout))

            if batch_norm == 'Yes':
                self.bn_layers.append(nn.BatchNorm2d(out_ch))
            else:
                self.bn_layers.append(nn.Identity())

            # update spatial size.....................................
            current_size = img_size(current_size, k, padding, stride)
            current_size = img_size(current_size, pool_size[0], 0, 2)
            in_ch = out_ch

        flat_size = num_filters[-1] * current_size * current_size

        # Fully connected layers............................
        self.fc_layers = nn.ModuleList()
        self.fc_bn_layers = nn.ModuleList()

        prev_dim = flat_size
        for dim in fc_dims:
            self.fc_layers.append(nn.Linear(prev_dim, dim))
            if batch_norm == 'Yes':
                self.fc_bn_layers.append(nn.BatchNorm1d(dim))
            else:
                self.fc_bn_layers.append(nn.Identity())
            prev_dim = dim

        self.fc_drop = nn.Dropout1d(dropout)
        self.out = nn.Linear(prev_dim, num_classes)

    def forward_prop(self, x):
        # Convolutional layers.............................
        for conv, bn, drop in zip(self.conv_layers, self.bn_layers, self.drop_layers):
            x = conv(x)
            x = bn(x)
            x = self.activation(x)
            x = self.pool(x)
            x = drop(x)

        # Flattening for dense layers.........................
        x = x.view(x.size(0), -1)

        # Fully connected ...................................
        for fc, bn in zip(self.fc_layers, self.fc_bn_layers):
            x = fc(x)
            x = bn(x)
            x = self.activation(x)
            x = self.fc_drop(x)

        # Output layer..........................
        x = self.out(x)
        return x

import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
import numpy as np

def load_dataset(train_data_dir, data_aug='Yes', batch_size=64, val_split=0.2):
    # Define image transformations
    basic_resize = transforms.Resize((224, 224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    crop_resize = transforms.RandomResizedCrop(224)
    flip = transforms.RandomHorizontalFlip()
    color_shift = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
    rotate = transforms.RandomRotation(20)

    # Apply augmentation or not
    if data_aug == 'Yes':
        transform_pipeline = transforms.Compose([
            crop_resize,
            color_shift,
            flip,
            rotate,
            to_tensor,
            normalize
        ])
    else:
        transform_pipeline = transforms.Compose([
            basic_resize,
            to_tensor,
            normalize
        ])

    # Loading dataset.................................................
    full_dataset = ImageFolder(root=train_data_dir, transform=transform_pipeline)
    total_len = len(full_dataset)
    val_len = int(val_split * total_len)
    train_len = total_len - val_len

    # Splitting the dataset using torch's random_split..........................
    train_set, val_set = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    # Creating dataloaders...............................
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return {
        'train': train_loader,
        'val': val_loader,
        'classes': full_dataset.classes
    }

