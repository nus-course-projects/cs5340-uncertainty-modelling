import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model.ResNet3D import ResNet3D
from model.I3D import InceptionI3d
from accelerate.utils.modeling import load_checkpoint_in_model
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    Div255,
    Normalize,
)
from torchvision.transforms import Compose, Lambda, Resize
from tqdm import tqdm
import argparse
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from utils.dataset import load_msasl, load_asl_citizen
from utils.optical_flow import OpticalFlowTransform

class TransformDataset:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        video, label, metadata = self.dataset[idx]
        sample = {"video": video, "label": label, "metadata": metadata}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample["video"], sample["label"], sample.get("metadata", {})

def compute_calibration_metrics(confidence, accuracy, num_bins=10):
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)"""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    mce = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
        if np.any(in_bin):
            bin_confidence = np.mean(confidence[in_bin])
            bin_accuracy = np.mean(accuracy[in_bin])
            bin_weight = np.mean(in_bin)
            ece += np.abs(bin_confidence - bin_accuracy) * bin_weight
            mce = max(mce, np.abs(bin_confidence - bin_accuracy))
    
    return ece, mce

def plot_reliability_diagram(confidence, accuracy, num_bins=10, title="Reliability Diagram", filename="reliability_diagram.png"):
    """Plot reliability diagram"""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = []
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
        if np.any(in_bin):
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_confidences.append(np.mean(confidence[in_bin]))
            bin_accuracies.append(np.mean(accuracy[in_bin]))
            bin_counts.append(np.sum(in_bin))
    
    bin_centers = np.array(bin_centers)
    bin_confidences = np.array(bin_confidences)
    bin_accuracies = np.array(bin_accuracies)
    bin_counts = np.array(bin_counts)
    
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(bin_confidences, bin_accuracies, 's-', label='Model Calibration')
    plt.bar(bin_centers, bin_counts / np.sum(bin_counts), alpha=0.3, label='Sample Count', width=0.1)
    
    # Add sample count information
    total_samples = np.sum(bin_counts)
    plt.text(1.02, 0.95, f'Total Samples: {total_samples}', transform=plt.gca().transAxes)
    for i, (center, count) in enumerate(zip(bin_centers, bin_counts)):
        plt.text(1.02, 0.85 - i*0.05, f'Bin {i+1}: {count}', transform=plt.gca().transAxes)
    
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

@torch.inference_mode()
def evaluate_epoch(model, dataloader, criterion, device, num_monte_carlo=10):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_confidences = []
    all_accuracies = []
    
    with torch.no_grad():
        for videos, labels, _ in tqdm(dataloader, desc="Evaluation batches", leave=False):
            output_mc = []
            videos = videos.to(device)
            labels = labels.to(device)
            for _ in range(num_monte_carlo):
                outputs = model(videos)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * videos.size(0)
                output_mc.append(outputs)
            total += labels.size(0)
            output_mc = torch.stack(output_mc, dim=0).mean(dim=0)
            probs = torch.softmax(output_mc, dim=1)
            confidences, preds = torch.max(probs, 1)
            correct += (preds == labels).sum().item()
            
            # Store confidences and accuracies for calibration metrics
            all_confidences.extend(confidences.cpu().numpy())
            all_accuracies.extend((preds == labels).cpu().numpy())
    
    epoch_loss = running_loss / total if total > 0 else float('inf')
    epoch_acc = correct / total if total > 0 else 0
    
    # Compute calibration metrics
    all_confidences = np.array(all_confidences)
    all_accuracies = np.array(all_accuracies)
    ece, mce = compute_calibration_metrics(all_confidences, all_accuracies)
    
    return epoch_loss, epoch_acc, ece, mce, all_confidences, all_accuracies

def main():
    parser = argparse.ArgumentParser(description="Test Video Classification Model using MS-ASL dataset with Accelerate support")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of frames to sample from each video")
    parser.add_argument("--model", type=str, default="resnet", choices=["resnet", "i3d"], help="Model to use for evaluation")
    parser.add_argument("--input_type", type=str, default="rgb", choices=["rgb", "optical_flow"], help="Input image of the model. Only for I3D model.")
    parser.add_argument("--top_k_labels", type=int, default=100, help="Number of top k labels to use for evaluation")
    parser.add_argument("--num_monte_carlo", type=int, default=10, help="Number of Monte Carlo samples for evaluation")
    parser.add_argument("--dataset", type=str, default="msasl", choices=["msasl", "asl_citizen"], help="Dataset to use for evaluation")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing the model checkpoint")
    parser.add_argument("--bayesian_layers", type=int, default=None, help="Number of Bayesian layers for the model, this layer to end layer")
    args = parser.parse_args()

    # Initialize accelerator
    config = ProjectConfiguration(project_dir=".")
    accelerator = Accelerator(project_config=config)

    # Create output directory if it doesn't exist
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Adjust num_monte_carlo based on bayesian_layers
    if args.bayesian_layers is None:
        args.num_monte_carlo = 1
        print("No Bayesian layers specified, setting num_monte_carlo to 1")

    # Load datasets
    test_dataset, train_dataset, validation_dataset = load_msasl("bin", top_k_labels=args.top_k_labels) if args.dataset == "msasl" else load_asl_citizen("ASL_Citizen", top_k_labels=args.top_k_labels)

    # Define transforms
    test_transform = ApplyTransformToKey(
        key="video",
        transform=Compose([
            OpticalFlowTransform() if args.input_type == "optical_flow" else Lambda(lambda x: x),
            Lambda(lambda x: x.permute(1, 0, 2, 3)),  # Convert (T,H,W,C) -> (C,T,H,W)
            UniformTemporalSubsample(args.num_frames),
            Resize((112, 112)) if args.model == "resnet" else Resize(224),
            Div255(),
            Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]) if args.model == "resnet" \
                else Lambda(lambda x: (x-0.5)*2.0),
        ])
    )

    # Wrap datasets with transforms
    test_dataset = TransformDataset(test_dataset, test_transform)
    validation_dataset = TransformDataset(validation_dataset, test_transform)

    # Create DataLoaders
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = accelerator.device
    print("Using device:", device)

    # Load model
    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",
        "moped_enable": True,
        "moped_delta": 0.5,
    }

    if args.model == "resnet":
        model = ResNet3D(num_classes=args.top_k_labels, bayesian_layers=args.bayesian_layers, bayesian_options=const_bnn_prior_parameters)
    elif args.model == "i3d":
        assert args.bayesian_layers is not None, "Bayesian layers not supported for I3D model yet!"
        if args.input_type == "rgb":
            model = InceptionI3d(num_classes=args.top_k_labels)
        elif args.input_type == "optical_flow":
            model = InceptionI3d(num_classes=args.top_k_labels, in_channels=2, input_type="optical_flow")
        else:
            raise ValueError(f"Invalid input type for I3D: {args.input_type}")
    else:
        raise ValueError(f"Invalid model: {args.model}")

    # Load checkpoint
    load_checkpoint_in_model(model, args.ckpt_dir)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    
    # Prepare with accelerator
    model, test_loader, val_loader = accelerator.prepare(model, test_loader, val_loader)

    # Evaluate on validation set
    val_loss, val_acc, val_ece, val_mce, val_confidences, val_accuracies = evaluate_epoch(model, val_loader, criterion, device, args.num_monte_carlo)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    print(f"Validation ECE: {val_ece:.4f}, Validation MCE: {val_mce:.4f}")
    plot_reliability_diagram(val_confidences, val_accuracies, 
                           title="Validation Set Reliability Diagram", 
                           filename=os.path.join(args.ckpt_dir, "validation_reliability_diagram.png"))

    # Evaluate on test set
    test_loss, test_acc, test_ece, test_mce, test_confidences, test_accuracies = evaluate_epoch(model, test_loader, criterion, device, args.num_monte_carlo)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f"Test ECE: {test_ece:.4f}, Test MCE: {test_mce:.4f}")
    plot_reliability_diagram(test_confidences, test_accuracies, 
                           title="Test Set Reliability Diagram", 
                           filename=os.path.join(args.ckpt_dir, "test_reliability_diagram.png"))

if __name__ == "__main__":
    main()
