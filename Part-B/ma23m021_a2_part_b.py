
import torch
import torch.nn as nn
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
import numpy as np

def pretrained_model(model_name='resnet', freeze_percent=0.5, freeze_all_except_last_layer='No'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the models.........................
    if model_name == 'resnet':
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        classifier_layer_name = 'fc'

    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        classifier_layer_name = 'fc'

    elif model_name == 'inception':
        model = models.inception_v3(pretrained=True, aux_logits=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        classifier_layer_name = 'fc'

    elif model_name == 'vgg':
        model = models.vgg16(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 10)
        classifier_layer_name = 'classifier.6'

    else:
        raise ValueError("Unsupported model. Choose from ['resnet', 'googlenet', 'inception', 'vgg']")

    # Freezing stratagies...........................
    if freeze_all_except_last_layer == 'Yes':
        for name, param in model.named_parameters():
            if classifier_layer_name not in name:
                param.requires_grad = False
    else:
        # Partial freezing: Freeze % of the model
        children = list(model.children())
        num_layers = len(children)
        num_freeze = int(freeze_percent * num_layers)

        for idx, layer in enumerate(children):
            requires_grad = False if idx < num_freeze else True
            for param in layer.parameters():
                param.requires_grad = requires_grad

    return model.to(device)


def loading_data(train_data_dir, data_aug='Yes', batch_size=64, val_split=0.2):
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

    # Load dataset
    full_dataset = ImageFolder(root=train_data_dir, transform=transform_pipeline)
    total_len = len(full_dataset)
    val_len = int(val_split * total_len)
    train_len = total_len - val_len

    # Split the dataset using torch's random_split
    train_set, val_set = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return {
        'train': train_loader,
        'val': val_loader,
        'classes': full_dataset.classes
    }
















