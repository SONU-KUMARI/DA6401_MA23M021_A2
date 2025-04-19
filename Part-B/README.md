# DA6401 Deep Learning Assignment-2 Part-B (Fine-Tuning Pretrained Models on iNaturalist Dataset)

##  Overview

In this part of the assignment, we fine-tune various **pretrained deep learning models** (ResNet50, GoogLeNet, InceptionV3, VGG16) on the iNaturalist-12K dataset using PyTorch and track experiment metrics via Weights & Biases (WandB). 

---

## Dataset

The dataset used is inaturalist with this structure:

- **Train set**: Used for training and validation (split inside code)
- **Val set**: Randomly split from train (default 80/20)

---

## Model Architecture & Freezing Strategy

I took these models to load and fine-tune of these pretrained models:
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

##  Main Components:

### `pretrained_model(...)`
- Loads pretrained model
- Modifies classifier head to output 10 classes
- Freezes layers either by percentage or keeps only final layer trainable

### `loading_data(...)`
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

best_model:
- 'model_name': 'resnet',
- 'freeze_percent': 0.8,
- 'epochs': 10,
- 'lr': 0.0001,
- 'data_aug': 'No'


##  How to Run

I have given ```ma23m021_a2_part_b.py``` and ```train_part-B.py``` files. These files have to be in same directory after downloading as I am importing the functions from this file in train_part-A.py file. You just have to change the address of data in line 20 of ```train_part-A.py``` And use this command to run the code finally:
```
python train_part-B.py 
```
