"""Adversarial perturbation engine for GhostScan.

This module provides routines to generate Fast Gradient Sign Method (FGSM) and
Projected Gradient Descent (PGD) adversarial examples using torchattacks.
"""

from typing import Tuple, Union
import torch
import torch.nn as nn
import torchattacks

from contracts import AttackType


def generate_adversarial_batch(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    attack_type: Union[AttackType, str],
    eps: float = 8 / 255,
) -> Tuple[torch.Tensor, AttackType]:
    """Generate adversarial perturbations for a batch of clean images.

    Args:
        model: Target PyTorch classification model (nn.Module) used to compute adversarial gradients.
            Expected to accept image tensors of shape [B, C, H, W] and output class logits.
        images: Input clean image tensor of shape [B, C, H, W] with pixel values in range [0.0, 1.0].
            where:
                - B: Batch size
                - C: Number of channels (3 for RGB or 1 for grayscale)
                - H: Image height
                - W: Image width
        labels: Ground truth or target class index tensor of shape [B].
        attack_type: Type of perturbation to generate, chosen from AttackType (CLEAN, FGSM, PGD).
        eps: Maximum L-infinity perturbation budget epsilon (default: 8/255 ≈ 0.0314).

    Returns:
        Tuple[torch.Tensor, AttackType]:
            - Perturbed image tensor of shape [B, C, H, W] bounded in [0.0, 1.0].
            - Selected AttackType enum instance.

    Raises:
        ValueError: If attack_type is not supported.
    """
    if isinstance(attack_type, str):
        attack_type = AttackType(attack_type.upper())

    if attack_type == AttackType.CLEAN:
        return images.clone(), attack_type

    # Ensure model is in eval mode for attack generation
    was_training = model.training
    model.eval()

    if attack_type == AttackType.FGSM:
        attacker = torchattacks.FGSM(model, eps=eps)
    elif attack_type == AttackType.PGD:
        attacker = torchattacks.PGD(model, eps=eps, alpha=2 / 255, steps=10)
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")

    # Generate adversarial images
    adv_images = attacker(images, labels)

    # Restore prior training mode
    if was_training:
        model.train()

    return adv_images, attack_type


if __name__ == "__main__":
    from victim_model import VictimClassifier

    print("--- Testing GhostScan Attack Engine ---")
    device = torch.device("cpu")
    eps = 8 / 255  # ~0.03137

    # Initialize victim model
    model = VictimClassifier(num_classes=2, pretrained=True)
    model.to(device)
    model.eval()

    # Create dummy batch of normalized images [B, C, H, W] in [0, 1]
    torch.manual_seed(42)
    dummy_images = torch.rand(4, 3, 224, 224, device=device)
    dummy_labels = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device)

    print(f"Base image batch shape: {dummy_images.shape}")
    print(f"Target perturbation budget eps: {eps:.6f} ({eps*255:.1f}/255)")

    # 1. FGSM Attack
    print("\nRunning FGSM attack...")
    adv_fgsm, atk_type_fgsm = generate_adversarial_batch(
        model=model,
        images=dummy_images,
        labels=dummy_labels,
        attack_type=AttackType.FGSM,
        eps=eps,
    )
    linf_fgsm = (adv_fgsm - dummy_images).abs().max().item()
    print(f"FGSM Attack Type: {atk_type_fgsm.value}")
    print(f"FGSM Output shape: {adv_fgsm.shape}")
    print(f"FGSM L-infinity perturbation: {linf_fgsm:.6f}")
    assert adv_fgsm.shape == dummy_images.shape, "Shape mismatch for FGSM output"
    assert linf_fgsm <= eps + 1e-5, f"FGSM perturbation {linf_fgsm} exceeded eps {eps}"

    # 2. PGD Attack (10 steps)
    print("\nRunning PGD attack (10 steps)...")
    adv_pgd, atk_type_pgd = generate_adversarial_batch(
        model=model,
        images=dummy_images,
        labels=dummy_labels,
        attack_type=AttackType.PGD,
        eps=eps,
    )
    linf_pgd = (adv_pgd - dummy_images).abs().max().item()
    print(f"PGD Attack Type: {atk_type_pgd.value}")
    print(f"PGD Output shape: {adv_pgd.shape}")
    print(f"PGD L-infinity perturbation: {linf_pgd:.6f}")
    assert adv_pgd.shape == dummy_images.shape, "Shape mismatch for PGD output"
    assert linf_pgd <= eps + 1e-5, f"PGD perturbation {linf_pgd} exceeded eps {eps}"

    # 3. Clean Baseline
    print("\nRunning Clean baseline...")
    clean_out, atk_type_clean = generate_adversarial_batch(
        model=model,
        images=dummy_images,
        labels=dummy_labels,
        attack_type=AttackType.CLEAN,
        eps=eps,
    )
    linf_clean = (clean_out - dummy_images).abs().max().item()
    print(f"Clean Attack Type: {atk_type_clean.value}")
    print(f"Clean L-infinity perturbation: {linf_clean:.6f}")
    assert linf_clean == 0.0, "Clean images were modified"

    print("\nAll attack engine self-tests passed successfully!")
