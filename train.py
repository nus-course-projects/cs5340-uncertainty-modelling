import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
import torch.nn as nn
import torch.optim as optim
from model.ResNet3D import ResNet3D
from model.I3D import InceptionI3d
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    Div255,
    Normalize,
    # The modules above are only compatible with torchvision 0.19.
    # If you use torchvision 0.21 or above, change functional_tensor to functional in line 9 of:
    # $ENV_INSTALLATION/lib/python3.12/site-packages/pytorchvideo/transforms/augmentations.py
)
from torchvision.transforms import Compose, Lambda, CenterCrop, Resize, RandomAffine
from tqdm import tqdm
import argparse
from datetime import datetime
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from utils.dataset import load_msasl
from utils.optical_flow import OpticalFlowTransform

class RandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            return x.flip(3)
        return x

# A helper dataset wrapper that applies a transform to each sample.
class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        """
        Args:
            dataset (Dataset): Original dataset returning (video, label, metadata)
            transform (callable): Transformation to apply on a sample dict.
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        video, label, metadata = self.dataset[idx]
        sample = {"video": video, "label": label, "metadata": metadata}
        if self.transform is not None:
            sample = self.transform(sample)
        # Return transformed video, label, and optionally metadata
        return sample["video"], sample["label"], sample.get("metadata", {})

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for videos, labels, _ in tqdm(dataloader, desc="Training batches", leave=False):
        videos = videos.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        accelerator.backward(loss)
        optimizer.step()

        running_loss += loss.item() * videos.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    epoch_loss = running_loss / total if total > 0 else float('inf')
    epoch_acc = correct / total if total > 0 else 0
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for videos, labels, _ in tqdm(dataloader, desc="Validation batches", leave=False):
            videos = videos.to(device)
            labels = labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * videos.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    epoch_loss = running_loss / total if total > 0 else float('inf')
    epoch_acc = correct / total if total > 0 else 0
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description="Train Video Classification Model using MS-ASL dataset with Accelerate support")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of frames to sample from each video")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--frozen_layers", type=int, default=None, help="Number of frozen layers for the model")
    parser.add_argument("--model", type=str, default="resnet", choices=["resnet", "i3d"], help="Model to use for training")
    parser.add_argument("--input_type", type=str, default="rgb", choices=["rgb", "optical_flow"], help="Input image of the model. Only for I3D model.")
    parser.add_argument("--top_k_labels", type=int, default=100, help="Number of top k labels to use for training")
    args = parser.parse_args()

    # Define experiment variables.
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_frames = args.num_frames
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    num_classes = args.top_k_labels

    # Create a ProjectConfiguration object and initialize Accelerator with TensorBoard logging.
    config = ProjectConfiguration(project_dir=".", logging_dir="runs")
    global accelerator
    accelerator = Accelerator(log_with="tensorboard", project_config=config)

    # Log hyperparameters.
    hparams = {
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_frames": num_frames,
        "num_classes": num_classes,
        "num_workers": num_workers,
        "model_class": args.model,
        "input_type": args.input_type,
        "frozen_layers": args.frozen_layers
    }
    accelerator.init_trackers(f'3DCNN_{args.model}_{args.frozen_layers}')
    tb_tracker = accelerator.get_tracker("tensorboard")

    # Load datasets.
    test_dataset, train_dataset, validation_dataset = load_msasl("bin", top_k_labels=num_classes)

    train_transform = ApplyTransformToKey(
        key="video",
        transform=Compose([
            OpticalFlowTransform() if args.input_type  == "optical_flow" else Lambda(lambda x: x),
            Lambda(lambda x: x.permute(1, 0, 2, 3)),  # Convert (T,H,W,C) -> (C,T,H,W)
            UniformTemporalSubsample(num_frames),
            Resize(112) if args.model == "resnet" else Resize(224),
            RandomHorizontalFlip(),
            Div255(),
            Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]) if args.model == "resnet" \
                else Lambda(lambda x: (x-0.5)*2.0),
        ])
    )
    test_transform = ApplyTransformToKey(
        key="video",
        transform=Compose([
            OpticalFlowTransform() if args.input_type  == "optical_flow" else Lambda(lambda x: x),
            Lambda(lambda x: x.permute(1, 0, 2, 3)),  # Convert (T,H,W,C) -> (C,T,H,W)
            UniformTemporalSubsample(num_frames),
            Resize(112) if args.model == "resnet" else Resize(224),
            Div255(),
            Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]) if args.model == "resnet" \
                else Lambda(lambda x: (x-0.5)*2.0),
        ])
    )
    # Wrap each dataset with the transformation.
    train_dataset = TransformDataset(train_dataset, train_transform)
    validation_dataset = TransformDataset(validation_dataset, test_transform)
    test_dataset = TransformDataset(test_dataset, test_transform)

    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = accelerator.device
    print("Using device:", device)

    if args.model == "resnet":
        model = ResNet3D(num_classes=num_classes, frozen_layers=args.frozen_layers)
    elif args.model == "i3d":
        if args.input_type == "rgb":
            model = InceptionI3d(num_classes=num_classes, frozen_layers=args.frozen_layers)
        elif args.input_type == "optical_flow":
            model = InceptionI3d(num_classes=num_classes, frozen_layers=args.frozen_layers, in_channels=2, input_type="optical_flow")
        else:
            raise ValueError(f"Invalid input type for I3D: {args.input_type}")
    else:
        raise ValueError(f"Invalid model: {args.model}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # Prepare with accelerator.
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    best_val_acc = 0.0
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        accelerator.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}, step=epoch)
        accelerator.print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model based on validation accuracy.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            accelerator.save_model(model, "best_video_classification_model")
            accelerator.print(f"New best model saved with Val Acc: {best_val_acc:.4f}")
        metrics = {"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "best_val_acc": best_val_acc}
        tb_tracker.writer.add_hparams(hparams, metrics, run_name=f'3DCNN_{args.model}_{args.frozen_layers}', global_step=epoch)
    accelerator.wait_for_everyone()
    torch.save(model.state_dict(), "video_classification_model_resnet.pth")
    accelerator.print("Training complete. Model saved as video_classification_model_resnet.pth")

if __name__ == "__main__":
    main()
