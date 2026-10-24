"""Explainability and forensic attribution engine for GhostScan.

This module provides attribution routines using Captum's LayerGradCam to visualize
adversarial artifacts detected by the ForensicNoiseStream, and matplotlib utilities
to blend heatmaps over original scan images.
"""

from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from captum.attr import LayerGradCam


def get_forensic_attribution(
    detector: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int = 1,
) -> torch.Tensor:
    """Compute LayerGradCam forensic attribution heatmap on the ForensicNoiseStream.

    Targets the final convolutional layer of the detector's forensic noise stream
    (`detector.forensic_stream.conv3`), isolating high-frequency noise and boundary artifacts.

    Args:
        detector: DualStreamDetector neural network instance (nn.Module).
        image_tensor: Input image tensor of expected shape [1, C, H, W] or [C, H, W],
            where C is channel count (1 or 3), H is image height, and W is image width.
        target_class: Target class index to explain (0: CLEAN, 1: FGSM, 2: PGD). Default is 1.

    Returns:
        torch.Tensor: 2D attribution heatmap tensor of shape [H, W] normalized to [0.0, 1.0].
    """
    # Ensure input tensor is batched [1, C, H, W]
    if image_tensor.dim() == 3:
        input_tensor = image_tensor.unsqueeze(0)
    elif image_tensor.dim() == 4:
        input_tensor = image_tensor
    else:
        raise ValueError(f"Expected 3D or 4D image tensor, got shape {image_tensor.shape}")

    # Match device and ensure gradient tracking
    device = next(detector.parameters()).device
    input_tensor = input_tensor.to(device).clone().detach()
    input_tensor.requires_grad = True

    # Identify the target layer: last conv layer of ForensicNoiseStream
    if hasattr(detector, "forensic_stream") and hasattr(detector.forensic_stream, "conv3"):
        target_layer = detector.forensic_stream.conv3
    else:
        # Fallback to search for final Conv2d in forensic stream
        forensic_stream = getattr(detector, "forensic_stream", detector)
        conv_layers = [m for m in forensic_stream.modules() if isinstance(m, nn.Conv2d)]
        if not conv_layers:
            raise AttributeError("Could not find a Conv2d layer in detector.forensic_stream")
        target_layer = conv_layers[-1]

    detector.eval()
    layer_grad_cam = LayerGradCam(detector, target_layer)

    # Compute attribution: shape [1, 1, H, W]
    attributions = layer_grad_cam.attribute(input_tensor, target=target_class)

    # Remove batch and channel dims -> [H, W]
    heatmap = attributions.squeeze()
    if heatmap.dim() > 2:
        heatmap = heatmap[0]

    # Normalize heatmap to [0.0, 1.0]
    h_min = heatmap.min()
    h_max = heatmap.max()
    if h_max > h_min:
        heatmap = (heatmap - h_min) / (h_max - h_min)
    else:
        heatmap = torch.zeros_like(heatmap)

    heatmap = torch.clamp(heatmap, 0.0, 1.0)
    return heatmap.detach().cpu()


def overlay_heatmap(
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    heatmap: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    """Blend a forensic attribution heatmap over the original image using Matplotlib colormaps.

    Args:
        image: Original input image as a PyTorch Tensor [C, H, W] or [1, C, H, W],
            NumPy array [H, W, 3], or PIL Image.
        heatmap: 2D attribution heatmap tensor or NumPy array of shape [H, W] in range [0, 1].
        alpha: Blending ratio for the heatmap overlay (0.0 = original image, 1.0 = heatmap).
        colormap: Matplotlib colormap name (default: 'jet').

    Returns:
        Image.Image: Alpha-blended RGB PIL Image highlighting localized forensic artifacts.
    """
    # 1. Standardize image to float NumPy array [H, W, 3] in range [0.0, 1.0]
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    elif isinstance(image, torch.Tensor):
        img_t = image.detach().cpu()
        if img_t.dim() == 4:
            img_t = img_t[0]
        if img_t.shape[0] in (1, 3):
            # [C, H, W] -> [H, W, C]
            img_t = img_t.permute(1, 2, 0)
        img_np = img_t.numpy().astype(np.float32)
        if img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)

        # Rescale if normalized with negative values
        i_min, i_max = img_np.min(), img_np.max()
        if i_max > i_min:
            img_np = (img_np - i_min) / (i_max - i_min)
        else:
            img_np = np.zeros_like(img_np)
    elif isinstance(image, np.ndarray):
        img_np = image.astype(np.float32)
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # 2. Standardize heatmap to NumPy array [H, W] in [0.0, 1.0]
    if isinstance(heatmap, torch.Tensor):
        hm_np = heatmap.detach().cpu().squeeze().numpy().astype(np.float32)
    elif isinstance(heatmap, np.ndarray):
        hm_np = heatmap.squeeze().astype(np.float32)
    else:
        raise TypeError(f"Unsupported heatmap type: {type(heatmap)}")

    hm_min, hm_max = hm_np.min(), hm_np.max()
    if hm_max > hm_min:
        hm_np = (hm_np - hm_min) / (hm_max - hm_min)
    else:
        hm_np = np.zeros_like(hm_np)
    hm_np = np.clip(hm_np, 0.0, 1.0)

    # 3. Resize heatmap if resolution differs from base image
    target_h, target_w = img_np.shape[0], img_np.shape[1]
    if hm_np.shape != (target_h, target_w):
        hm_pil = Image.fromarray((hm_np * 255).astype(np.uint8)).resize(
            (target_w, target_h), resample=Image.Resampling.BILINEAR
        )
        hm_np = np.array(hm_pil, dtype=np.float32) / 255.0

    # 4. Apply Matplotlib colormap and blend
    cmap = plt.get_cmap(colormap)
    colored_heatmap = cmap(hm_np)[:, :, :3]  # Drop alpha channel -> [H, W, 3] in [0, 1]

    blended = (1.0 - alpha) * img_np + alpha * colored_heatmap
    blended_uint8 = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(blended_uint8)


if __name__ == "__main__":
    from attacks import generate_adversarial_batch
    from contracts import AttackType
    from detector import DualStreamDetector
    from victim_model import VictimClassifier

    print("--- Testing GhostScan Explainability (Captum LayerGradCam) ---")
    device = torch.device("cpu")

    # 1. Initialize victim and detector
    victim = VictimClassifier(num_classes=2, pretrained=True).to(device)
    detector = DualStreamDetector(pretrained_backbone=True).to(device)

    # 2. Generate a synthetic chest X-ray image with smooth lung fields and an edge
    torch.manual_seed(42)
    clean_sample = torch.zeros(1, 3, 224, 224, device=device)
    # Add smooth background / lung field
    clean_sample[:, :, 50:180, 50:180] = 0.5
    # Add high-contrast sharp edge
    clean_sample[:, :, 100:120, 100:120] = 0.95

    label = torch.tensor([1], dtype=torch.long, device=device)

    # 3. Perturb with FGSM
    adv_fgsm, _ = generate_adversarial_batch(
        model=victim,
        images=clean_sample,
        labels=label,
        attack_type=AttackType.FGSM,
        eps=8 / 255,
    )

    # 4. Compute forensic attribution heatmap
    print("Computing LayerGradCam attribution on ForensicNoiseStream.conv3...")
    heatmap = get_forensic_attribution(
        detector=detector,
        image_tensor=adv_fgsm,
        target_class=1,  # Explaining FGSM class
    )

    print(f"Attribution heatmap shape: {heatmap.shape}")
    print(f"Heatmap min value: {heatmap.min().item():.4f}")
    print(f"Heatmap max value: {heatmap.max().item():.4f}")

    assert heatmap.shape == (224, 224), f"Expected shape (224, 224), got {heatmap.shape}"
    assert 0.0 <= heatmap.min().item() <= heatmap.max().item() <= 1.0, "Heatmap not in [0, 1]"

    # 5. Blend heatmap over original image
    print("Overlaying heatmap on image using Matplotlib...")
    overlay_img = overlay_heatmap(adv_fgsm, heatmap, alpha=0.5)
    print(f"Overlay PIL Image dimensions: {overlay_img.size}, mode: {overlay_img.mode}")
    assert isinstance(overlay_img, Image.Image), "Expected PIL Image return type"
    assert overlay_img.size == (224, 224), f"Expected size (224, 224), got {overlay_img.size}"

    print("\nExplainability self-test completed successfully!")
