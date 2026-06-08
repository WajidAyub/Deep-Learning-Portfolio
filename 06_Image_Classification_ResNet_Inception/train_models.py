"""
Image Classification Using Pretrained CNN Architectures
Comparing ResNet50 and InceptionV3 on CIFAR-10 Dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time
import os
import json
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Configuration
# ============================================================
NUM_CLASSES = 10
BATCH_SIZE = 64
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
OUTPUT_DIR = "./outputs"

CIFAR10_CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")

# ============================================================
# Step 1: Download and Preprocess CIFAR-10 Dataset
# ============================================================
print("\n[Step 1] Downloading and preprocessing CIFAR-10 dataset...")

# ResNet50 expects 224x224 images
transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_resnet_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# InceptionV3 expects 299x299 images
transform_inception = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_inception_test = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Download datasets
train_dataset_resnet = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=True, download=True, transform=transform_resnet)
test_dataset_resnet = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=False, download=True, transform=transform_resnet_test)

train_dataset_inception = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=True, download=True, transform=transform_inception)
test_dataset_inception = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=False, download=True, transform=transform_inception_test)

train_loader_resnet = DataLoader(train_dataset_resnet, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=2)
test_loader_resnet = DataLoader(test_dataset_resnet, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=2)

train_loader_inception = DataLoader(train_dataset_inception, batch_size=BATCH_SIZE,
                                     shuffle=True, num_workers=2)
test_loader_inception = DataLoader(test_dataset_inception, batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=2)

print(f"Training samples: {len(train_dataset_resnet)}")
print(f"Test samples: {len(test_dataset_resnet)}")
print(f"Number of classes: {NUM_CLASSES}")


# ============================================================
# Step 2 & 3: Build Models with Transfer Learning
# ============================================================
def build_resnet50(num_classes):
    """Load pretrained ResNet50 and replace the final layer."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, num_classes)
    )
    return model


def build_inceptionv3(num_classes):
    """Load pretrained InceptionV3 and replace the final layer."""
    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, num_classes)
    )
    # Replace AuxLogits classifier too
    num_features_aux = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features_aux, num_classes)
    )
    return model


# ============================================================
# Step 4: Training Function
# ============================================================
def train_model(model, train_loader, test_loader, model_name, is_inception=False):
    """Train a model and record metrics per epoch."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "epoch_times": []
    }

    total_start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            if is_inception:
                outputs, aux_outputs = model(inputs)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # Evaluation phase
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = test_loss / len(test_loader)
        test_acc = 100.0 * correct / total

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["epoch_times"].append(epoch_time)

        print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")

        scheduler.step()

    total_time = time.time() - total_start_time
    history["total_training_time"] = total_time

    # Calculate model size
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    history["total_params"] = param_count
    history["trainable_params"] = trainable_params
    history["model_size_mb"] = model_size_mb

    print(f"\n{model_name} Training Complete!")
    print(f"  Total Training Time: {total_time:.1f}s")
    print(f"  Best Test Accuracy: {max(history['test_acc']):.2f}%")
    print(f"  Total Parameters: {param_count:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Model Size: {model_size_mb:.2f} MB")

    return model, history


# ============================================================
# Step 5: Generate Confusion Matrix
# ============================================================
def get_confusion_matrix(model, test_loader):
    """Generate confusion matrix from test data."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return confusion_matrix(all_labels, all_preds)


# ============================================================
# Step 6: Plotting Functions
# ============================================================
def plot_accuracy_comparison(resnet_history, inception_history):
    """Plot accuracy vs epoch for both models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, NUM_EPOCHS + 1)

    ax.plot(epochs, resnet_history["test_acc"], 'b-o', linewidth=2,
            markersize=6, label="ResNet50 Test Accuracy")
    ax.plot(epochs, inception_history["test_acc"], 'r-s', linewidth=2,
            markersize=6, label="InceptionV3 Test Accuracy")
    ax.plot(epochs, resnet_history["train_acc"], 'b--', linewidth=1.5,
            alpha=0.6, label="ResNet50 Train Accuracy")
    ax.plot(epochs, inception_history["train_acc"], 'r--', linewidth=1.5,
            alpha=0.6, label="InceptionV3 Train Accuracy")

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Accuracy vs Epoch: ResNet50 vs InceptionV3", fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_comparison.png"), dpi=150)
    plt.close()
    print("Saved: accuracy_comparison.png")


def plot_loss_comparison(resnet_history, inception_history):
    """Plot loss vs epoch for both models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, NUM_EPOCHS + 1)

    ax.plot(epochs, resnet_history["test_loss"], 'b-o', linewidth=2,
            markersize=6, label="ResNet50 Test Loss")
    ax.plot(epochs, inception_history["test_loss"], 'r-s', linewidth=2,
            markersize=6, label="InceptionV3 Test Loss")
    ax.plot(epochs, resnet_history["train_loss"], 'b--', linewidth=1.5,
            alpha=0.6, label="ResNet50 Train Loss")
    ax.plot(epochs, inception_history["train_loss"], 'r--', linewidth=1.5,
            alpha=0.6, label="InceptionV3 Train Loss")

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title("Loss vs Epoch: ResNet50 vs InceptionV3", fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_comparison.png"), dpi=150)
    plt.close()
    print("Saved: loss_comparison.png")


def plot_confusion_matrices(cm_resnet, cm_inception):
    """Plot confusion matrices for both models side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(cm_resnet, annot=True, fmt='d', cmap='Blues',
                xticklabels=CIFAR10_CLASSES, yticklabels=CIFAR10_CLASSES,
                ax=axes[0], cbar_kws={'shrink': 0.8})
    axes[0].set_title("ResNet50 Confusion Matrix", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Label", fontsize=11)
    axes[0].tick_params(axis='both', labelsize=9)

    sns.heatmap(cm_inception, annot=True, fmt='d', cmap='Reds',
                xticklabels=CIFAR10_CLASSES, yticklabels=CIFAR10_CLASSES,
                ax=axes[1], cbar_kws={'shrink': 0.8})
    axes[1].set_title("InceptionV3 Confusion Matrix", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("True Label", fontsize=11)
    axes[1].tick_params(axis='both', labelsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"), dpi=150)
    plt.close()
    print("Saved: confusion_matrices.png")


def plot_training_time_comparison(resnet_history, inception_history):
    """Bar chart comparing training times."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models_names = ["ResNet50", "InceptionV3"]
    times = [resnet_history["total_training_time"],
             inception_history["total_training_time"]]
    colors = ['#3498db', '#e74c3c']

    bars = ax.bar(models_names, times, color=colors, width=0.5, edgecolor='black')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{t:.1f}s', ha='center', fontsize=12, fontweight='bold')

    ax.set_ylabel("Training Time (seconds)", fontsize=13)
    ax.set_title("Total Training Time Comparison", fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_time_comparison.png"), dpi=150)
    plt.close()
    print("Saved: training_time_comparison.png")


def plot_model_size_comparison(resnet_history, inception_history):
    """Bar chart comparing model sizes."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models_names = ["ResNet50", "InceptionV3"]
    sizes = [resnet_history["model_size_mb"],
             inception_history["model_size_mb"]]
    colors = ['#3498db', '#e74c3c']

    bars = ax.bar(models_names, sizes, color=colors, width=0.5, edgecolor='black')
    for bar, s in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{s:.1f} MB', ha='center', fontsize=12, fontweight='bold')

    ax.set_ylabel("Model Size (MB)", fontsize=13)
    ax.set_title("Model Size Comparison", fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_size_comparison.png"), dpi=150)
    plt.close()
    print("Saved: model_size_comparison.png")


# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    # Build models
    print("\n[Step 2] Building pretrained models...")
    resnet_model = build_resnet50(NUM_CLASSES)
    inception_model = build_inceptionv3(NUM_CLASSES)

    # Train ResNet50
    resnet_model, resnet_history = train_model(
        resnet_model, train_loader_resnet, test_loader_resnet,
        "ResNet50", is_inception=False
    )

    # Train InceptionV3
    inception_model, inception_history = train_model(
        inception_model, train_loader_inception, test_loader_inception,
        "InceptionV3", is_inception=True
    )

    torch.save(resnet_model.state_dict(), os.path.join(OUTPUT_DIR, "resnet50.pth"))
    print(f"Saved: resnet50.pth")
    torch.save(inception_model.state_dict(), os.path.join(OUTPUT_DIR, "inceptionv3.pth"))
    print(f"Saved: inceptionv3.pth")

    # Generate confusion matrices
    print("\n[Step 5] Generating confusion matrices...")
    cm_resnet = get_confusion_matrix(resnet_model, test_loader_resnet)
    cm_inception = get_confusion_matrix(inception_model, test_loader_inception)

    # Generate all plots
    print("\n[Step 6] Generating plots...")
    plot_accuracy_comparison(resnet_history, inception_history)
    plot_loss_comparison(resnet_history, inception_history)
    plot_confusion_matrices(cm_resnet, cm_inception)
    plot_training_time_comparison(resnet_history, inception_history)
    plot_model_size_comparison(resnet_history, inception_history)

    # Save results to JSON for report generation
    results = {
        "resnet50": {
            "train_loss": resnet_history["train_loss"],
            "train_acc": resnet_history["train_acc"],
            "test_loss": resnet_history["test_loss"],
            "test_acc": resnet_history["test_acc"],
            "epoch_times": resnet_history["epoch_times"],
            "total_training_time": resnet_history["total_training_time"],
            "total_params": resnet_history["total_params"],
            "trainable_params": resnet_history["trainable_params"],
            "model_size_mb": resnet_history["model_size_mb"],
            "best_test_acc": max(resnet_history["test_acc"]),
            "final_test_loss": resnet_history["test_loss"][-1],
        },
        "inceptionv3": {
            "train_loss": inception_history["train_loss"],
            "train_acc": inception_history["train_acc"],
            "test_loss": inception_history["test_loss"],
            "test_acc": inception_history["test_acc"],
            "epoch_times": inception_history["epoch_times"],
            "total_training_time": inception_history["total_training_time"],
            "total_params": inception_history["total_params"],
            "trainable_params": inception_history["trainable_params"],
            "model_size_mb": inception_history["model_size_mb"],
            "best_test_acc": max(inception_history["test_acc"]),
            "final_test_loss": inception_history["test_loss"][-1],
        },
        "confusion_matrix_resnet": cm_resnet.tolist(),
        "confusion_matrix_inception": cm_inception.tolist(),
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "dataset": "CIFAR-10",
        "num_classes": NUM_CLASSES,
    }

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results.json")

    # Print summary comparison table
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Metric':<30} {'ResNet50':<20} {'InceptionV3':<20}")
    print("-"*70)
    print(f"{'Best Test Accuracy':<30} {max(resnet_history['test_acc']):<20.2f} {max(inception_history['test_acc']):<20.2f}")
    print(f"{'Final Test Loss':<30} {resnet_history['test_loss'][-1]:<20.4f} {inception_history['test_loss'][-1]:<20.4f}")
    print(f"{'Total Training Time (s)':<30} {resnet_history['total_training_time']:<20.1f} {inception_history['total_training_time']:<20.1f}")
    print(f"{'Model Size (MB)':<30} {resnet_history['model_size_mb']:<20.2f} {inception_history['model_size_mb']:<20.2f}")
    print(f"{'Total Parameters':<30} {resnet_history['total_params']:<20,} {inception_history['total_params']:<20,}")
    print(f"{'Trainable Parameters':<30} {resnet_history['trainable_params']:<20,} {inception_history['trainable_params']:<20,}")
    print("="*70)

    print("\nAll outputs saved to ./outputs/ directory.")
    print("Run 'python generate_report.py' to create the PDF report.")
