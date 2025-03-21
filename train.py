import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
import torch.nn as nn
import torch.optim as optim
from torchvision.models.video import r3d_18
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    ShortSideScale,
    UniformCropVideo,
    Normalize,
    Div255
)
from torchvision.transforms import Compose, Lambda, CenterCrop
from tqdm import tqdm
import gc
import argparse

from accelerate import Accelerator
from utils.dataset import load_msasl

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
        loss.backward()
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
    args = parser.parse_args()

    batch_size = args.batch_size
    num_workers = args.num_workers
    num_frames = args.num_frames

    # Initialize Accelerator (mixed precision is set via accelerate launch)
    accelerator = Accelerator()

    # Settings
    num_epochs = 100
    learning_rate = 0.001
    num_classes = 1000
    test_dataset, train_dataset, validation_dataset = load_msasl("bin", num_classes)

    video_transform = ApplyTransformToKey(
        key="video",
        transform=Compose([
            Lambda(lambda x: x.permute(1, 0, 2, 3)),  # Convert (T,C,H,W) -> (C,T,H,W)
            UniformTemporalSubsample(num_frames),
            Div255(),
            CenterCrop(224),
            Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])
    )
    # Wrap each dataset with the transformation.
    train_dataset = TransformDataset(train_dataset, video_transform)
    validation_dataset = TransformDataset(validation_dataset, video_transform)
    test_dataset = TransformDataset(test_dataset, video_transform)

    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = accelerator.device
    print("Using device:", device)

    model = r3d_18(pretrained=False, progress=True, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare with accelerator.
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        accelerator.print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    accelerator.wait_for_everyone()
    torch.save(model.state_dict(), "video_classification_model_resnet.pth")
    accelerator.print("Training complete. Model saved as video_classification_model_resnet.pth")

if __name__ == "__main__":
    main()
