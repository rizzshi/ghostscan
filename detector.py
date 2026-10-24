"""Dual-stream adversarial detector for GhostScan.

This module implements a dual-stream architecture classifying images into
CLEAN (0), FGSM (1), or PGD (2) adversarial categories by combining:
1. SpatialStream: A frozen pretrained ResNet-18 feature extractor (512-d).
2. ForensicNoiseStream: A CNN operating on high-pass noise residuals (64-d).
"""

from typing import Any, Callable, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models

from contracts import AttackType


class ForensicNoiseStream(nn.Module):
    """Forensic noise residual feature extractor.

    Operates on a high-pass spatial residual computed via average pooling subtraction:
        residual = image - F.avg_pool2d(image, kernel_size=3, stride=1, padding=1)
    to isolate high-frequency adversarial noise patterns, followed by a 3-layer CNN
    yielding a 64-dimensional forensic feature representation.
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 64):
        """Initialize the forensic noise stream CNN layers.

        Args:
            in_channels: Number of input channels (typically 3 for RGB).
            out_dim: Output feature dimension (64).
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=False)

        self.conv3 = nn.Conv2d(32, out_dim, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_dim)
        self.relu3 = nn.ReLU(inplace=False)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def compute_residual(self, image: torch.Tensor) -> torch.Tensor:
        """Compute high-pass noise residual using non-in-place average pool subtraction.

        Args:
            image: Input image tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: High-pass noise residual tensor of shape [B, C, H, W].
        """
        low_pass = F.avg_pool2d(image, kernel_size=3, stride=1, padding=1)
        residual = image - low_pass
        return residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 64-dimensional forensic noise features from input image tensor.

        Args:
            x: Input image tensor of shape [B, C, H, W] (typically [B, 3, 224, 224]).

        Returns:
            torch.Tensor: Forensic feature vector of shape [B, 64].
        """
        residual = self.compute_residual(x)
        out = self.relu1(self.bn1(self.conv1(residual)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.relu3(self.bn3(self.conv3(out)))
        out = self.global_pool(out)
        return torch.flatten(out, 1)


class SpatialStream(nn.Module):
    """Frozen spatial feature extractor using pretrained ResNet-18 up to layer4.

    All parameters are frozen (requires_grad = False) to act as a fixed semantic representation.
    Outputs a 512-dimensional feature vector.
    """

    def __init__(self, pretrained: bool = True):
        """Initialize frozen ResNet-18 spatial feature extractor.

        Args:
            pretrained: Whether to load ImageNet-pretrained weights (ResNet18_Weights.DEFAULT).
        """
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )

        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> "SpatialStream":
        """Keep backbone in eval mode to prevent running statistics from updating."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dimensional spatial features from input image tensor.

        Args:
            x: Input image tensor of shape [B, C, H, W] (typically [B, 3, 224, 224]).

        Returns:
            torch.Tensor: Frozen spatial feature vector of shape [B, 512].
        """
        out = self.backbone(x)
        return torch.flatten(out, 1)


class DualStreamDetector(nn.Module):
    """Dual-stream detector combining frozen spatial semantics and forensic noise residuals.

    Architecture:
        - SpatialStream: Pretrained frozen ResNet-18 (512-d)
        - ForensicNoiseStream: 3-layer CNN on high-pass noise residual (64-d)
        - Fusion Classifier: Linear(576, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 3)

    Target classes:
        0: CLEAN, 1: FGSM, 2: PGD
    """

    def __init__(self, pretrained_backbone: bool = True, dropout_rate: float = 0.3):
        """Initialize DualStreamDetector with spatial and forensic streams and classifier head.

        Args:
            pretrained_backbone: Whether to initialize SpatialStream with pretrained ResNet-18 weights.
            dropout_rate: Dropout probability in the classification head (default 0.3).
        """
        super().__init__()
        self.spatial_stream = SpatialStream(pretrained=pretrained_backbone)
        self.forensic_stream = ForensicNoiseStream(in_channels=3, out_dim=64)

        self.classifier = nn.Sequential(
            nn.Linear(512 + 64, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing attack classification logits [CLEAN, FGSM, PGD].

        Args:
            x: Input image tensor of shape [B, C, H, W] (typically [B, 3, 224, 224]).

        Returns:
            torch.Tensor: Logits tensor of shape [B, 3].
        """
        spatial_feats = self.spatial_stream(x)    # [B, 512]
        forensic_feats = self.forensic_stream(x)  # [B, 64]
        combined = torch.cat([spatial_feats, forensic_feats], dim=1)  # [B, 576]
        logits = self.classifier(combined)        # [B, 3]
        return logits


def train_detector(
    detector: nn.Module,
    clean_loader: DataLoader,
    attack_generator: Union[Callable, nn.Module],
    epochs: int = 3,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    victim_model: Optional[nn.Module] = None,
) -> List[float]:
    """Train the dual-stream detector on clean and generated adversarial samples.

    For each batch, clean images are augmented with dynamically generated FGSM and PGD
    counterparts, with targets 0 (CLEAN), 1 (FGSM), and 2 (PGD), trained with Adam
    and CrossEntropyLoss.

    Args:
        detector: DualStreamDetector instance to train.
        clean_loader: DataLoader yielding clean batches of images [B, 3, 224, 224] and labels [B].
        attack_generator: Attack generation function or victim model used to create perturbations.
        epochs: Number of training epochs (default 3).
        lr: Learning rate for Adam optimizer (default 1e-3).
        device: PyTorch device (CPU or CUDA). Defaults to CUDA if available else CPU.
        victim_model: Optional victim model to supply if attack_generator expects (model, images, labels, type).

    Returns:
        List[float]: Epoch-wise average training loss history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector.to(device)
    detector.train()
    optimizer = optim.Adam(
        [p for p in detector.parameters() if p.requires_grad],
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss()
    loss_history: List[float] = []

    def _generate(imgs: torch.Tensor, lbls: torch.Tensor, atk_type: AttackType) -> torch.Tensor:
        if callable(attack_generator):
            try:
                out = attack_generator(imgs, lbls, atk_type)
                return out[0] if isinstance(out, tuple) else out
            except TypeError:
                if victim_model is not None:
                    out = attack_generator(victim_model, imgs, lbls, atk_type)
                    return out[0] if isinstance(out, tuple) else out
        if isinstance(attack_generator, nn.Module):
            from attacks import generate_adversarial_batch
            out = generate_adversarial_batch(attack_generator, imgs, lbls, atk_type)
            return out[0] if isinstance(out, tuple) else out

        from attacks import generate_adversarial_batch
        vm = victim_model if victim_model is not None else attack_generator
        out = generate_adversarial_batch(vm, imgs, lbls, atk_type)
        return out[0] if isinstance(out, tuple) else out

    for epoch in range(epochs):
        running_loss, total_samples = 0.0, 0
        for clean_images, targets in clean_loader:
            clean_images = clean_images.to(device)
            targets = targets.to(device)
            b_size = clean_images.size(0)

            # Generate adversarial counterparts
            fgsm_images = _generate(clean_images, targets, AttackType.FGSM).to(device)
            pgd_images = _generate(clean_images, targets, AttackType.PGD).to(device)

            # Labels: CLEAN=0, FGSM=1, PGD=2
            labels_clean = torch.zeros(b_size, dtype=torch.long, device=device)
            labels_fgsm = torch.ones(b_size, dtype=torch.long, device=device)
            labels_pgd = torch.full((b_size,), 2, dtype=torch.long, device=device)

            # Concatenate batches
            all_images = torch.cat([clean_images, fgsm_images, pgd_images], dim=0)
            all_labels = torch.cat([labels_clean, labels_fgsm, labels_pgd], dim=0)

            optimizer.zero_grad()
            outputs = detector(all_images)
            loss = criterion(outputs, all_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * all_images.size(0)
            total_samples += all_images.size(0)

        epoch_loss = running_loss / max(1, total_samples)
        loss_history.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{epochs}] - Detector Loss: {epoch_loss:.4f}")

    return loss_history


if __name__ == "__main__":
    from attacks import generate_adversarial_batch
    from victim_model import VictimClassifier
    from torch.utils.data import TensorDataset

    print("--- Testing GhostScan DualStreamDetector ---")
    device = torch.device("cpu")

    # 1. Initialize detector
    detector = DualStreamDetector(pretrained_backbone=True)
    detector.to(device)

    # 2. Verify SpatialStream has zero trainable parameters
    spatial = detector.spatial_stream
    trainable_spatial_params = sum(p.numel() for p in spatial.parameters() if p.requires_grad)
    print(f"Spatial stream trainable parameters: {trainable_spatial_params}")
    assert trainable_spatial_params == 0, f"Expected 0 trainable parameters, got {trainable_spatial_params}"

    # 3. Verify forward pass on batch of shape [4, 3, 224, 224] returns logits [4, 3]
    dummy_input = torch.randn(4, 3, 224, 224, device=device)
    logits = detector(dummy_input)
    print(f"Detector output logits shape: {logits.shape}")
    assert logits.shape == (4, 3), f"Expected shape [4, 3], got {logits.shape}"

    # 4. Verify non-in-place residual gradient computation in ForensicNoiseStream
    forensic = detector.forensic_stream
    dummy_x = torch.randn(2, 3, 224, 224, device=device, requires_grad=True)
    forensic_out = forensic(dummy_x)
    assert forensic_out.shape == (2, 64), f"Expected [2, 64], got {forensic_out.shape}"
    loss_test = forensic_out.sum()
    loss_test.backward()
    assert dummy_x.grad is not None, "Gradient failed to backpropagate through residual computation"
    print(f"Forensic residual backward gradient norm: {dummy_x.grad.norm().item():.4f}")

    # 5. Test train_detector with a dummy clean loader and victim model
    print("\nTesting train_detector loop...")
    victim = VictimClassifier(num_classes=2, pretrained=True).to(device)
    dummy_clean = torch.rand(2, 3, 224, 224, device=device)
    dummy_y = torch.tensor([0, 1], dtype=torch.long, device=device)
    dummy_loader = DataLoader(TensorDataset(dummy_clean, dummy_y), batch_size=2)

    history = train_detector(
        detector=detector,
        clean_loader=dummy_loader,
        attack_generator=generate_adversarial_batch,
        epochs=2,
        victim_model=victim,
        device=device,
    )
    print("Detector training self-test completed successfully!")
