# /// script
# dependencies = [
# "torch>=1.9.0",
# "torchvision>=0.10.0",
# "matplotlib>=3.3.0",
# ]
# [[tool.uv.index]]
# name = "pytorch-cuda"
# url = "https://download.pytorch.org/whl/cu124"
# explicit = true
# ///

"""
Head-to-head comparison: PyTorch SGD vs ClassicSGD with ReduceLROnPlateau

This script demonstrates how the decoupled learning rate and momentum in ClassicSGD
behaves differently from PyTorch's standard SGD, especially when learning rate
scheduling is applied.

Key insight:
- PyTorch SGD:  v = μ*v + g,     p = p - lr*v  (LR scales velocity at update time)
- ClassicSGD:   v = μ*v + lr*g,  p = p - v     (LR scales gradient into velocity)

When ReduceLROnPlateau triggers:
- PyTorch SGD: Accumulated momentum is suddenly scaled down
- ClassicSGD: Momentum continues at current scale, only new gradients are scaled

This leads to smoother transitions and more predictable behavior with ClassicSGD.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import copy
import argparse
from pathlib import Path

from classic_sgd import ClassicSGD


class SimpleCNN(nn.Module):
    """Small CNN for FashionMNIST - complex enough to show optimizer differences."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_data_loaders(batch_size=128, num_workers=2):
    """Load FashionMNIST dataset."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
    )

    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    val_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Use generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(train_loader.dataset)


def evaluate(model, val_loader, criterion, device):
    """Evaluate model, return loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(val_loader.dataset)
    accuracy = 100.0 * correct / len(val_loader.dataset)

    return avg_loss, accuracy


def train_with_optimizer(
    model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, name
):
    """
    Train model and track metrics.

    Returns dict with training history.
    """
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "lr_reductions": [],  # epochs where LR was reduced
    }

    prev_lr = optimizer.param_groups[0]["lr"]

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Step scheduler (ReduceLROnPlateau uses val_loss)
        scheduler.step(val_loss)

        # Track current LR
        current_lr = optimizer.param_groups[0]["lr"]

        # Detect LR reduction
        if current_lr < prev_lr:
            history["lr_reductions"].append(epoch)
            print(
                f"  [{name}] Epoch {epoch}: LR reduced {prev_lr:.6f} -> {current_lr:.6f}"
            )
        prev_lr = current_lr

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(
                f"  [{name}] Epoch {epoch:3d}: "
                f"Train Loss={train_loss:.4f}, "
                f"Val Loss={val_loss:.4f}, "
                f"Val Acc={val_acc:.2f}%, "
                f"LR={current_lr:.6f}"
            )

    return history


def plot_comparison(pytorch_history, classic_history, save_path=None):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = range(len(pytorch_history["train_loss"]))

    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    ax1.plot(
        epochs, pytorch_history["train_loss"], "b-", label="PyTorch SGD", alpha=0.8
    )
    ax1.plot(epochs, classic_history["train_loss"], "r-", label="ClassicSGD", alpha=0.8)
    # Mark LR reductions
    for e in pytorch_history["lr_reductions"]:
        ax1.axvline(x=e, color="b", linestyle="--", alpha=0.3)
    for e in classic_history["lr_reductions"]:
        ax1.axvline(x=e, color="r", linestyle=":", alpha=0.3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    ax2 = axes[0, 1]
    ax2.plot(epochs, pytorch_history["val_loss"], "b-", label="PyTorch SGD", alpha=0.8)
    ax2.plot(epochs, classic_history["val_loss"], "r-", label="ClassicSGD", alpha=0.8)
    for e in pytorch_history["lr_reductions"]:
        ax2.axvline(x=e, color="b", linestyle="--", alpha=0.3)
    for e in classic_history["lr_reductions"]:
        ax2.axvline(x=e, color="r", linestyle=":", alpha=0.3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Validation Accuracy
    ax3 = axes[1, 0]
    ax3.plot(epochs, pytorch_history["val_acc"], "b-", label="PyTorch SGD", alpha=0.8)
    ax3.plot(epochs, classic_history["val_acc"], "r-", label="ClassicSGD", alpha=0.8)
    for e in pytorch_history["lr_reductions"]:
        ax3.axvline(x=e, color="b", linestyle="--", alpha=0.3)
    for e in classic_history["lr_reductions"]:
        ax3.axvline(x=e, color="r", linestyle=":", alpha=0.3)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Validation Accuracy (%)")
    ax3.set_title("Validation Accuracy Comparison")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Learning Rate Schedule
    ax4 = axes[1, 1]
    ax4.plot(epochs, pytorch_history["lr"], "b-", label="PyTorch SGD", alpha=0.8)
    ax4.plot(epochs, classic_history["lr"], "r--", label="ClassicSGD", alpha=0.8)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Learning Rate")
    ax4.set_title("Learning Rate Schedule (ReduceLROnPlateau)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch SGD vs ClassicSGD with LR scheduling"
    )
    parser.add_argument(
        "--epochs", type=int, default=40, help="Number of training epochs (default: 40)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.05, help="Initial learning rate (default: 0.05)"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum coefficient (default: 0.9)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size (default: 128)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="ReduceLROnPlateau patience (default: 3)",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau factor (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (useful for headless environments)",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="comparison_results.png",
        help="Path to save plot (default: comparison_results.png)",
    )

    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loaders
    print("\nLoading FashionMNIST dataset...")
    train_loader, val_loader = get_data_loaders(batch_size=args.batch_size)

    # Create base model with fixed seed
    torch.manual_seed(args.seed)
    base_model = SimpleCNN()

    # Clone model for fair comparison (identical starting weights)
    pytorch_model = copy.deepcopy(base_model).to(device)
    classic_model = copy.deepcopy(base_model).to(device)

    # Setup optimizers with identical hyperparameters
    pytorch_optimizer = torch.optim.SGD(
        pytorch_model.parameters(), lr=args.lr, momentum=args.momentum
    )
    classic_optimizer = ClassicSGD(
        classic_model.parameters(), lr=args.lr, momentum=args.momentum
    )

    # Setup schedulers with identical parameters
    pytorch_scheduler = ReduceLROnPlateau(
        pytorch_optimizer,
        mode="min",
        factor=args.factor,
        patience=args.patience,
    )
    classic_scheduler = ReduceLROnPlateau(
        classic_optimizer,
        mode="min",
        factor=args.factor,
        patience=args.patience,
    )

    # Print configuration
    print(f"\n{'=' * 60}")
    print("EXPERIMENT CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Initial LR:      {args.lr}")
    print(f"  Momentum:        {args.momentum}")
    print(f"  Batch Size:      {args.batch_size}")
    print(
        f"  Scheduler:       ReduceLROnPlateau (patience={args.patience}, factor={args.factor})"
    )
    print(f"  Random Seed:     {args.seed}")
    print(f"{'=' * 60}\n")

    # Train PyTorch SGD
    print("Training with PyTorch SGD...")
    print("-" * 40)
    # Reset data loader randomness for fair comparison
    torch.manual_seed(args.seed)
    pytorch_history = train_with_optimizer(
        pytorch_model,
        train_loader,
        val_loader,
        pytorch_optimizer,
        pytorch_scheduler,
        args.epochs,
        device,
        "PyTorch SGD",
    )

    # Train ClassicSGD
    print("\nTraining with ClassicSGD...")
    print("-" * 40)
    # Reset data loader randomness for fair comparison
    torch.manual_seed(args.seed)
    classic_history = train_with_optimizer(
        classic_model,
        train_loader,
        val_loader,
        classic_optimizer,
        classic_scheduler,
        args.epochs,
        device,
        "ClassicSGD",
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  PyTorch SGD:")
    print(f"    Final Val Acc:     {pytorch_history['val_acc'][-1]:.2f}%")
    print(f"    Best Val Acc:      {max(pytorch_history['val_acc']):.2f}%")
    print(f"    Final Val Loss:    {pytorch_history['val_loss'][-1]:.4f}")
    print(f"    LR Reductions:     {len(pytorch_history['lr_reductions'])} times")
    print()
    print(f"  ClassicSGD:")
    print(f"    Final Val Acc:     {classic_history['val_acc'][-1]:.2f}%")
    print(f"    Best Val Acc:      {max(classic_history['val_acc']):.2f}%")
    print(f"    Final Val Loss:    {classic_history['val_loss'][-1]:.4f}")
    print(f"    LR Reductions:     {len(classic_history['lr_reductions'])} times")
    print(f"{'=' * 60}")

    # Plot results
    if not args.no_plot:
        print("\nGenerating comparison plots...")
        plot_comparison(pytorch_history, classic_history, save_path=args.save_plot)


if __name__ == "__main__":
    main()
