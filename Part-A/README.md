# DA6401 Deep Learning Assignment-2 Part-A

## Overview

This part focuses on building a custom convolutional neural network (CNN) from scratch using PyTorch. The model is trained and evaluated on the iNaturalist12K dataset. All training experiments are tracked using Weights & Biases (WandB).

---

## Dataset

The dataset used is the `inaturalist_12K`, which consists of natural images across 10 categories. It is split as follows:

- **Training**: `/inaturalist_12K/train`
- **Validation** (20% split from training)
- **Test**: `/inaturalist_12K/val`

---

##  Model Architecture
### 1. `ConvNet` (Custom CNN Class)
A fully customizable CNN model is implemented with the following features:

- Variable convolutional filters and sizes
- Activation function (`ReLU`, `Mish`, `SiLU`, etc.)
- Optional Batch Normalization
- Dropout in both convolutional and dense layers
- Customizable fully connected layers
- Final linear layer for 10-class classification


### 2. `load_dataloaders()`
- Loads training and validation sets
- Supports optional data augmentation (random crops, flips, color jitter, etc.)

### 3. `load_test_loader()`
- Loads the test set with optional augmentation (disabled by default)

### 4. `train_with_wandb()`
- Trains the CNN using Adam optimizer and CrossEntropyLoss
- Tracks training accuracy
- Runs for configurable number of epochs

### 5. `evaluate_test()`
- Evaluates final model on test set
- Computes and logs final accuracy

### 6. `plot_predictions_grid()`
- Collects 3 sample images per class
- Shows predictions vs ground truth in a 10x3 image grid
- Logs the grid to WandB

### 7. Main Block
- Initializes WandB
- Instantiates model with best config from sweeps
- Trains model, evaluates on test data, plots predictions

---

##  Training Pipeline

- Uses `Adam` optimizer and `CrossEntropyLoss`
- Includes validation accuracy logging
- Configurable epochs and learning rate
- Batch size: 64
- Image input size: 224×224

###  Sweep Hyperparameters (WandB)

WandB Sweeps were used to search for the best hyperparameters. Here's a sample of what was optimized:

- `activation`: ReLU, Mish, SiLU, GELU
- `dropout`: [0.2, 0.3]
- `batch_norm`: [Yes,No]
- `fc_dims`: e.g., [128], [256, 128]
- `kernel_size`: [3, 3, 3, 3, 3], [3, 5, 5, 7, 7], [3, 5, 3, 5, 7], [5, 5, 5, 5, 5]
- `num_filters`: [32, 32, 32, 32, 32], [128, 128, 64, 64, 32], [32, 64, 128, 256, 512], [32, 64, 64, 128, 128]
- `learning_rate`: [0.001, 0.0005]
- `data_aug`: [Yes, No]

**Best configuration:**

best_model:
- `activation` = Mish,
- `dropout` = 0.2,
- `batch_norm`= 'Yes',
- `fc_dims`= [128],
- `num_filters`= [32, 64, 128, 256, 512],
- `kernel_size` =[3, 3, 3, 3, 3],
- `data_aug` = 'No'


---

##  Evaluation

After training:
- Final model is evaluated on the test set
- Test accuracy is logged
- A 10x3 image prediction grid is generated for visual inspection

---

##  Logging with WandB

The training and evaluation metrics are tracked using WandB. The following logs are saved:
- Training accuracy per epoch
- Final test accuracy
- Prediction grid image

---

##  How to Run

I have given ```ma23m021_a2_part_a.py``` and ```train_part-A.py``` files. These files have to be in same directory after downloading as I am importing the functions from this file in train_part-A.py file. You just change the address of dataset in line 25 of ```train_part-A.py```.And use this command to run the code finally:
```
python train_part-A.py 
```
