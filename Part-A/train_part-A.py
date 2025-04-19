import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from ma23m021_a2_part_a import CNN_Scratch,load_dataset# setup_data # Replace with your actual model file



#setup_data()
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--kernel_size', type=int, nargs='+', default=[3, 3, 5, 5, 7])
    parser.add_argument('--filter_org', type=int, nargs='+', default=[32, 64, 128, 256, 512])
    parser.add_argument('--fc_dim', type=int, nargs='+', default=[128])
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'mish', 'silu', 'gelu'])
    parser.add_argument('--batch_norm', type=str, default='Yes', choices=['Yes', 'No'])
    parser.add_argument('--data_augmentation', type=str, default='Yes', choices=['Yes', 'No'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_dir', type=str, default="C:/Users/Sonu/Desktop/nature_12k/inaturalist_12K/train")

    return parser.parse_args()

def train_with_wandb(args):
    wandb.init(project="train-py-part-A", config=vars(args), name="run-" + wandb.util.generate_id())
    config = wandb.config
    print(f"Configuration: {config}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    activation_map = {
        'relu': nn.ReLU(),
        'mish': nn.Mish(),
        'silu': nn.SiLU(),
        'gelu': nn.GELU()
    }

    model = CNN_Scratch(
        in_channels=3,
        num_filters=config.filter_org,
        filter_size=config.kernel_size,
        activation=activation_map[config.activation],
        dropout=config.dropout,
        batch_norm=config.batch_norm == 'Yes',
        fc_dims=config.fc_dim,
        num_classes=10
    ).to(device)

    dataloaders = load_dataset(
        train_data_dir=config.data_dir,
        data_aug=config.data_augmentation,
        batch_size=config.batch_size,
        val_split=0.2
    )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    print("Dataloaders ready!")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        model.train()
        train_loss, train_correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            train_correct += (preds == labels).sum().item()

        train_acc = 100 * train_correct / total
        train_loss /= total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                loss = criterion(val_outputs, val_labels)

                val_loss += loss.item() * val_images.size(0)
                _, val_preds = val_outputs.max(1)
                val_total += val_labels.size(0)
                val_correct += (val_preds == val_labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss /= val_total

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")


if __name__ == "__main__":
    args = parse_args()
    train_with_wandb(args)

#python train_a2_part-A.py
    










