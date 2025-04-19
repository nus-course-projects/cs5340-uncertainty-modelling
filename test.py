import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
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
    correct_top1 = 0
    correct_top5 = 0
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
            
            # Calculate top-1 accuracy
            correct_top1 += (preds == labels).sum().item()
            
            # Calculate top-5 accuracy
            _, top5_preds = torch.topk(probs, 5, dim=1)
            correct_top5 += sum([1 for i in range(len(labels)) if labels[i] in top5_preds[i]])
            
            # Store confidences and accuracies for calibration metrics
            all_confidences.extend(confidences.cpu().numpy())
            all_accuracies.extend((preds == labels).cpu().numpy())
    
    epoch_loss = running_loss / total if total > 0 else float('inf')
    epoch_acc_top1 = correct_top1 / total if total > 0 else 0
    epoch_acc_top5 = correct_top5 / total if total > 0 else 0
    
    # Compute calibration metrics
    all_confidences = np.array(all_confidences)
    all_accuracies = np.array(all_accuracies)
    ece, mce = compute_calibration_metrics(all_confidences, all_accuracies)
    
    return epoch_loss, epoch_acc_top1, epoch_acc_top5, ece, mce, all_confidences, all_accuracies

def main():
    parser = argparse.ArgumentParser(description="Test Video Classification Model using MS-ASL dataset with Accelerate support")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of frames to sample from each video")
    parser.add_argument("--model", type=str, default="resnet", choices=["resnet", "i3d"], help="Model to use for evaluation")
    parser.add_argument("--input_type", type=str, default="rgb", choices=["rgb", "optical_flow"], help="Input image of the model. Only for I3D model.")
    parser.add_argument("--top_k_labels", type=int, default=100, help="Number of top k labels to use for evaluation")
    parser.add_argument("--dataset", type=str, default="msasl", choices=["msasl", "asl_citizen"], help="Dataset to use for evaluation")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing the model checkpoint")
    parser.add_argument("--bayesian_layers", type=int, default=None, help="Number of Bayesian layers for the model, this layer to end layer")
    parser.add_argument("--mc_iterations", type=str, default="1,5,10,20", help="Comma-separated list of Monte Carlo iterations to run (only for Bayesian models)")
    args = parser.parse_args()

    # Initialize accelerator
    config = ProjectConfiguration(project_dir=".")
    accelerator = Accelerator(project_config=config)

    # Create output directory if it doesn't exist
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Parse Monte Carlo iterations
    mc_iterations = [int(x.strip()) for x in args.mc_iterations.split(',')]
    
    # If not Bayesian model, use only the first iteration
    if args.bayesian_layers is None:
        mc_iterations = [1]
        print("No Bayesian layers specified, using single forward pass")

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

    # Initialize results dictionary
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "validation_metrics": {},
        "test_metrics": {}
    }

    # Run evaluation for each Monte Carlo iteration
    for mc_iter in mc_iterations:
        print(f"\nRunning evaluation with {mc_iter} Monte Carlo iterations")
        
        # Evaluate on validation set
        val_loss, val_acc_top1, val_acc_top5, val_ece, val_mce, val_confidences, val_accuracies = evaluate_epoch(model, val_loader, criterion, device, mc_iter)
        print(f"Validation (MC={mc_iter}) - Loss: {val_loss:.4f}, Top-1: {val_acc_top1:.4f}, Top-5: {val_acc_top5:.4f}")
        print(f"Validation (MC={mc_iter}) - ECE: {val_ece:.4f}, MCE: {val_mce:.4f}")
        
        # Save validation plots
        plot_reliability_diagram(val_confidences, val_accuracies, 
                               title=f"Validation Set Reliability Diagram (MC={mc_iter})", 
                               filename=os.path.join(args.ckpt_dir, f"validation_reliability_diagram_mc{mc_iter}.png"))

        # Evaluate on test set
        test_loss, test_acc_top1, test_acc_top5, test_ece, test_mce, test_confidences, test_accuracies = evaluate_epoch(model, test_loader, criterion, device, mc_iter)
        print(f"Test (MC={mc_iter}) - Loss: {test_loss:.4f}, Top-1: {test_acc_top1:.4f}, Top-5: {test_acc_top5:.4f}")
        print(f"Test (MC={mc_iter}) - ECE: {test_ece:.4f}, MCE: {test_mce:.4f}")
        
        # Save test plots
        plot_reliability_diagram(test_confidences, test_accuracies, 
                               title=f"Test Set Reliability Diagram (MC={mc_iter})", 
                               filename=os.path.join(args.ckpt_dir, f"test_reliability_diagram_mc{mc_iter}.png"))

        # Store results for this iteration
        results["validation_metrics"][f"mc_{mc_iter}"] = {
            "loss": float(val_loss),
            "top1_accuracy": float(val_acc_top1),
            "top5_accuracy": float(val_acc_top5),
            "ece": float(val_ece),
            "mce": float(val_mce)
        }
        
        results["test_metrics"][f"mc_{mc_iter}"] = {
            "loss": float(test_loss),
            "top1_accuracy": float(test_acc_top1),
            "top5_accuracy": float(test_acc_top5),
            "ece": float(test_ece),
            "mce": float(test_mce)
        }

    # Save all results to JSON file
    results_file = os.path.join(args.ckpt_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nAll results saved to {results_file}")

if __name__ == "__main__":
    main()
