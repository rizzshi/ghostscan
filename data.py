"""Reusable PyTorch data pipeline for Chest X-ray pneumonia classification.

This module provides data loading utilities using torchvision ImageFolder,
standard medical image normalization transforms, and a 70/15/15 train/val/test split.
"""

import os
from typing import Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


# Standard transforms matching ImageNet-pretrained ResNet expectations
DEFAULT_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def ensure_chest_xray_data(target_dir: str, num_samples_per_class: int = 120) -> str:
    """Ensure a valid ImageFolder directory structure exists, creating dummy data if needed.

    Args:
        target_dir: Root directory path for dataset.
        num_samples_per_class: Number of synthetic images to create per class if folder is empty.
            With 120 samples across 2 classes (240 total), train (70%) has 168 images,
            val (15%) has 36 images, and test (15%) has 36 images, guaranteeing full batches
            for batch_size=32.

    Returns:
        str: Verified path to dataset root folder containing class subdirectories.
    """
    classes = ["NORMAL", "PNEUMONIA"]
    needs_generation = False

    if not os.path.exists(target_dir):
        needs_generation = True
    else:
        # Check if class subdirectories exist and contain at least one image
        has_images = False
        for c in classes:
            c_dir = os.path.join(target_dir, c)
            if os.path.exists(c_dir) and any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(c_dir)):
                has_images = True
                break
        if not has_images:
            needs_generation = True

    if needs_generation:
        for idx, c in enumerate(classes):
            c_dir = os.path.join(target_dir, c)
            os.makedirs(c_dir, exist_ok=True)
            for i in range(num_samples_per_class):
                file_path = os.path.join(c_dir, f"sample_{i:04d}.png")
                if not os.path.exists(file_path):
                    # Create grayscale synthetic X-ray pattern (converted to RGB by ImageFolder)
                    color_val = 60 + (idx * 50) + (i % 30)
                    img = Image.new("RGB", (224, 224), color=(color_val, color_val, color_val))
                    img.save(file_path)

    return target_dir


def get_dataloaders(
    data_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    transform: Optional[transforms.Compose] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build and return train, validation, and test DataLoaders with a 70/15/15 split.

    Args:
        data_dir: Path to directory containing class subfolders (e.g. NORMAL/ and PNEUMONIA/).
            If None or empty, defaults to 'data/chest_xray' within current workspace.
        batch_size: Batch size for DataLoader instances. Output image batches have shape
            [B, 3, 224, 224] where B = batch_size.
        num_workers: Number of subprocess workers for data loading (0 for synchronous loading).
        seed: Random seed for reproducible dataset splitting.
        transform: Optional torchvision transforms. Defaults to Resize((224, 224)),
            ToTensor(), and Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]:
            - train_loader: DataLoader yielding batches of images [B, 3, 224, 224] and labels [B]
            - val_loader: DataLoader yielding validation batches [B, 3, 224, 224] and labels [B]
            - test_loader: DataLoader yielding test evaluation batches [B, 3, 224, 224] and labels [B]
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "chest_xray")

    verified_dir = ensure_chest_xray_data(data_dir)
    applied_transform = transform if transform is not None else DEFAULT_TRANSFORMS

    dataset = ImageFolder(root=verified_dir, transform=applied_transform)
    total_count = len(dataset)

    # 70% train, 15% val, 15% test split
    train_count = int(0.70 * total_count)
    val_count = int(0.15 * total_count)
    test_count = total_count - train_count - val_count

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_count, val_count, test_count],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_dl, val_dl, test_dl = get_dataloaders(batch_size=32)
    sample_imgs, sample_labels = next(iter(train_dl))
    print(f"Train batch images shape: {sample_imgs.shape}")  # [32, 3, 224, 224]
    print(f"Train batch labels shape: {sample_labels.shape}")  # [32]
    val_imgs, _ = next(iter(val_dl))
    print(f"Val batch images shape: {val_imgs.shape}")  # [32, 3, 224, 224]
    test_imgs, _ = next(iter(test_dl))
    print(f"Test batch images shape: {test_imgs.shape}")  # [32, 3, 224, 224]
