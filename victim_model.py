"""Baseline medical image classifier for Chest X-ray pneumonia detection.

This module provides VictimClassifier, a fine-tuned ResNet-18 model wrapping a pretrained
backbone with a 2-class classification head and feature extraction capability for Captum.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models


class VictimClassifier(nn.Module):
    """ResNet-18 classifier for chest X-ray pneumonia detection.

    Wraps a torchvision ResNet-18 backbone with a custom binary classification head.
    Exposes `forward_features` for intermediate representation extraction and Captum attributions.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """Initialize VictimClassifier with ResNet-18 backbone and binary classification head.

        Args:
            num_classes: Number of target output classes (default 2 for NORMAL vs PNEUMONIA).
            pretrained: Whether to load ImageNet-pretrained weights.
        """
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features  # 512 for ResNet-18
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled 512-dimensional feature vector before the final classifier layer.

        Args:
            x: Input image tensor of shape [B, C, H, W] (typically [B, 3, 224, 224]).

        Returns:
            torch.Tensor: Pooled feature vector tensor of shape [B, 512].
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing class logits for pneumonia classification.

        Args:
            x: Input image tensor of shape [B, C, H, W] (typically [B, 3, 224, 224]).

        Returns:
            torch.Tensor: Class logits tensor of shape [B, 2].
        """
        features = self.forward_features(x)
        return self.backbone.fc(features)


def train_victim(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-4,
    device: Optional[torch.device] = None,
) -> List[float]:
    """Train victim classifier on training DataLoader using Adam and CrossEntropyLoss.

    Args:
        model: Target PyTorch neural network module (nn.Module).
        train_loader: DataLoader yielding image tensors of shape [B, 3, 224, 224] and labels [B].
        epochs: Number of training epochs (default 5).
        lr: Learning rate for Adam optimizer (default 1e-4).
        device: Target torch device. Defaults to CUDA if available else CPU.

    Returns:
        List[float]: Epoch-wise average training loss history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loss_history = []

    for epoch in range(epochs):
        running_loss, total_samples = 0.0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        epoch_loss = running_loss / max(1, total_samples)
        loss_history.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.4f}")

    return loss_history


if __name__ == "__main__":
    device = torch.device("cpu")
    print("Initializing VictimClassifier (pretrained ResNet-18)...")
    victim = VictimClassifier(num_classes=2, pretrained=True)

    dummy_images = torch.randn(4, 3, 224, 224, device=device)
    dummy_labels = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device)

    # Validate feature extraction shape: [B, 512]
    features = victim.forward_features(dummy_images)
    print(f"Extracted features shape: {features.shape}")
    assert features.shape == (4, 512), f"Expected shape [4, 512], got {features.shape}"

    # Validate classification logits shape: [B, 2]
    logits = victim(dummy_images)
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (4, 2), f"Expected shape [4, 2], got {logits.shape}"

    # Train on dummy batch of shape [4, 3, 224, 224]
    print("Training on dummy batch of shape [4, 3, 224, 224]...")
    dummy_dataset = TensorDataset(dummy_images, dummy_labels)
    dummy_loader = DataLoader(dummy_dataset, batch_size=4)
    train_victim(victim, dummy_loader, epochs=5, device=device)
    print("Victim model self-test passed successfully.")
