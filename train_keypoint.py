from datetime import datetime
import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
import torch.nn as nn
import torch.optim as optim
from model.KeyPointsNet import KeyPointsLSTM, KeyPointsTransformer

from torchvision.transforms import Compose, Lambda, CenterCrop
from tqdm import tqdm
import argparse

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from utils.dataset import load_msasl, load_keypoints
class KeypointTransformDataset(Dataset):
    def __init__(self, dataset, shift_range=0.05, rotation_max=30, noise_scale=0.01, 
                 score_noise_scale=0.05, apply_transforms=True, prob=0.5):
        """
        Dataset wrapper that applies random transformations to keypoint data.
        
        Args:
            dataset (Dataset): Original keypoint dataset
            shift_range (float): Maximum shift as a fraction of frame size (default: 0.05)
            rotation_max (float): Maximum rotation in degrees (default: 30)
            noise_scale (float): Scale of Gaussian noise to add to coordinates (default: 0.01)
            score_noise_scale (float): Scale of Gaussian noise to add to scores (default: 0.05)
            apply_transforms (bool): Whether to apply transformations (set False for validation)
            prob (float): Probability of applying transformations (default: 0.5)
        """
        self.dataset = dataset
        self.shift_range = shift_range
        self.rotation_max = rotation_max
        self.noise_scale = noise_scale
        self.score_noise_scale = score_noise_scale
        self.apply_transforms = apply_transforms
        self.prob = 1.0 - prob
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        keypoints, label = self.dataset[idx]
        
        if self.apply_transforms:
            # Apply random transformations
            keypoints = self._apply_transformations(keypoints, prob=self.prob)
        
        return keypoints, label
    
    def _apply_transformations(self, keypoints, prob=0.5):
        """
        Apply random transformations to keypoints.
        
        Args:
            keypoints (torch.Tensor): Keypoints tensor of shape (T, K, 3) where 3 is (x, y, score)
            
        Returns:
            torch.Tensor: Transformed keypoints
        """
        # Create a copy to avoid modifying the original
        transformed = keypoints.clone()
        
        # Extract coordinates and scores
        coords = transformed[:, :, :2]  # (T, K, 2)
        scores = transformed[:, :, 2:3]  # (T, K, 1)
        
        # 1. Random shift
        if random.random() > prob:
            shift_x = random.uniform(-self.shift_range, self.shift_range)
            shift_y = random.uniform(-self.shift_range, self.shift_range)
            coords[:, :, 0] += shift_x  # Add to x coordinates
            coords[:, :, 1] += shift_y  # Add to y coordinates
        
        # 2. Random rotation around the center of keypoints
        if random.random() > prob:
            angle = random.uniform(-self.rotation_max, self.rotation_max)
            angle_rad = torch.tensor(angle * torch.pi / 180.0)
            
            # For each frame, calculate the center of valid keypoints
            for t in range(coords.size(0)):  # For each time step
                # Use only keypoints with valid scores
                valid_mask = scores[t, :, 0] > 0.1
                if valid_mask.sum() > 0:
                    # Calculate center of valid keypoints
                    valid_coords = coords[t, valid_mask]
                    center_x = valid_coords[:, 0].mean()
                    center_y = valid_coords[:, 1].mean()
                    
                    # Translate to keypoint center
                    coords[t, :, 0] -= center_x
                    coords[t, :, 1] -= center_y
                    
                    # Rotate
                    cos_angle = torch.cos(angle_rad)
                    sin_angle = torch.sin(angle_rad)
                    
                    x_rotated = coords[t, :, 0] * cos_angle - coords[t, :, 1] * sin_angle
                    y_rotated = coords[t, :, 0] * sin_angle + coords[t, :, 1] * cos_angle
                    
                    coords[t, :, 0] = x_rotated
                    coords[t, :, 1] = y_rotated
                    
                    # Translate back
                    coords[t, :, 0] += center_x
                    coords[t, :, 1] += center_y
        
        # 3. Add random noise to coordinates
        if random.random() > prob:
            noise = torch.randn_like(coords) * self.noise_scale
            # Only add noise to keypoints with high confidence
            high_confidence = (scores > 0.5).expand_as(coords)
            coords = torch.where(high_confidence, coords + noise, coords)
        
        # 4. Add random noise to scores
        if random.random() > prob:
            score_noise = torch.randn_like(scores) * self.score_noise_scale
            # Add noise but ensure scores stay in valid range [0, 1]
            scores = scores + score_noise
            scores = torch.clamp(scores, 0.0, 1.0)
        
        # 5. Random horizontal flip with respect to center keypoint
        if random.random() > prob:
            for t in range(coords.size(0)):  # For each time step
                valid_mask = scores[t, :, 0] > 0.1
                if valid_mask.sum() > 0:
                    # Calculate center of valid keypoints
                    valid_coords = coords[t, valid_mask]
                    center_x = valid_coords[:, 0].mean()
                    
                    # Flip x-coordinates around the center
                    coords[t, :, 0] = 2 * center_x - coords[t, :, 0]
        
        # Ensure coordinates and scores stay in valid range [0, 1]
        coords = torch.clamp(coords, 0.0, 1.0)
        scores = torch.clamp(scores, 0.0, 1.0)
        
        # Combine coordinates and scores back
        transformed = torch.cat([coords, scores], dim=2)
        
        return transformed



def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for keypoints, labels in tqdm(dataloader, desc="Training batches", leave=False):
        keypoints = keypoints.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(keypoints)
        loss = criterion(outputs, labels)
        accelerator.backward(loss)
        optimizer.step()

        running_loss += loss.item() * keypoints.size(0)
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
        for keypoints, labels in tqdm(dataloader, desc="Validation batches", leave=False):
            keypoints = keypoints.to(device)
            labels = labels.to(device)
            outputs = model(keypoints)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * keypoints.size(0)
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
    args = parser.parse_args()

    # Define experiment variables.
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_frames = args.num_frames
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    num_classes = 100

    # Create a ProjectConfiguration object and initialize Accelerator with TensorBoard logging.
    config = ProjectConfiguration(project_dir=".", logging_dir="runs")
    global accelerator
    accelerator = Accelerator(log_with="tensorboard", project_config=config)

    # Log hyperparameters.
    hparams = {
        "model_class": "KeypointsTransformer",
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_frames": num_frames,
        "num_classes": num_classes,
        "num_workers": num_workers
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    accelerator.init_trackers(f'video_classification_{timestamp}', config=hparams)

    # Load datasets.
    test_dataset, train_dataset, validation_dataset = load_keypoints("MS-ASL-Key-Points", top_k_labels=num_classes)
    
    # Apply transformations to training dataset only
    train_dataset = KeypointTransformDataset(train_dataset, shift_range=0.3, rotation_max=45, 
                                             noise_scale=0.10, score_noise_scale=0.20, apply_transforms=True, 
                                             prob=0.6)
    validation_dataset = KeypointTransformDataset(validation_dataset, apply_transforms=False)
    test_dataset = KeypointTransformDataset(test_dataset, apply_transforms=False)

    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = accelerator.device
    print("Using device:", device)

    # model = KeyPointsLSTM(num_classes=num_classes, hidden_size=128, mlp_hidden_size=128, num_layers=4, dropout=0.0)
    model = KeyPointsTransformer(num_classes=num_classes, hidden_size=32, num_layers=4, dropout=0.0, num_heads=4)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

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
            accelerator.save_model(model, "best_keypoints_model")
            accelerator.print(f"New best model saved with Val Acc: {best_val_acc:.4f}")

    accelerator.wait_for_everyone()
    torch.save(model.state_dict(), "keypoints_model.pth")
    accelerator.print("Training complete. Model saved as keypoints_model.pth")

if __name__ == "__main__":
    main()
