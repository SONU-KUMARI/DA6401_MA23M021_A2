# Assignment Part-B: Fine-Tuning Pretrained Models on iNaturalist Dataset

##  Overview

In this part of the assignment, we fine-tune various **pretrained deep learning models** (ResNet50, GoogLeNet, InceptionV3, VGG16) on the iNaturalist-12K dataset using PyTorch and track experiment metrics via Weights & Biases (WandB). 

---

## Dataset

The dataset used is inaturalist with this structure:

- **Train set**: Used for training and validation (split inside code)
- **Val set**: Randomly split from train (default 80/20)
- **Test set**: Not used in Part-B (could be added for final evaluation)

---

## Model Architecture & Freezing Strategy

The script allows loading and fine-tuning any of these pretrained models:
- **ResNet50**
- **GoogLeNet**
- **InceptionV3**
- **VGG16**

### Final Layers Replaced:
- **ResNet, GoogLeNet, InceptionV3**: Replace `fc` with `nn.Linear(..., 10)`
- **VGG16**: Replace `classifier[6]` with `nn.Linear(..., 10)`

### Layer Freezing Options:
- Freeze a **percentage of base layers** using `freeze_percent`
- OR freeze **all layers except final classifier** using `freeze_all_except_last_layer = 'Yes'`

---

##  Main Components Explained

### `pretrain_model(...)`
- Loads pretrained model
- Modifies classifier head to output 10 classes
- Freezes layers either by percentage or keeps only final layer trainable

### `load_dataloaders(...)`
- Applies data augmentation (optional)
- Resizes all images to 224x224
- Returns `train` and `val` `DataLoader`s

### `train_with_wandb()`
- Initializes WandB run and config
- Loads model and data
- Trains and evaluates the model over `config.epochs`
- Logs training & validation metrics to WandB

### Validation Metrics Logged:
- Accuracy
- Loss (for both train and val)

---

##  Sweep Configuration

A **Bayesian hyperparameter sweep** is used to explore the optimal combination of training strategies and model types.

### Sweep Parameters:
| Parameter               | Values                        |
|-------------------------|-------------------------------|
| `model_name`            | `['resnet', 'googlenet']`     |
| `epochs`                | `[5, 10]`                     |
| `freeze_percent`        | `[0.8, 0.9]`                  |
| `lr` (learning rate)    | `[0.001, 0.0001]`             |
| `data_aug`              | `['Yes', 'No']`              |


## Running One Custom Strategy

```python
config = {
    'model_name': 'resnet',
    'freeze_percent': 0.8,
    'epochs': 10,
    'lr': 0.0001,
    'data_aug': 'No'
}

