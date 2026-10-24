"""GhostScan interface contracts and core data definitions.

This module locks the interface contract for GhostScan, defining the data structures,
attack types, and detector function signatures before model implementation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import torch
import torch.nn as nn


class AttackType(str, Enum):
    """Enumeration of supported adversarial attack types and perturbation evaluation modes.

    Expected input tensor shape for perturbed images across all attack types is
    [B, C, H, W], where:
        - B: Batch size
        - C: Number of channels (e.g., 1 for grayscale/medical imaging, 3 for RGB)
        - H: Spatial height
        - W: Spatial width

    Attributes:
        CLEAN: Clean, unperturbed baseline image tensor [B, C, H, W].
        FGSM: Fast Gradient Sign Method single-step adversarial attack [B, C, H, W].
        PGD: Projected Gradient Descent iterative adversarial attack [B, C, H, W].
    """

    CLEAN = "CLEAN"
    FGSM = "FGSM"
    PGD = "PGD"


@dataclass
class ScanResult:
    """Dataclass capturing diagnostic output and adversarial forensic analysis.

    Expected tensor shape for associated input image:
        [B, C, H, W] (batch) or [C, H, W] (single sample).

    Attributes:
        diagnosis_label: Predicted diagnostic class label (e.g., condition or pathology name).
        diagnosis_confidence: Model confidence score for the diagnosis in the range [0.0, 1.0].
        is_adversarial: Boolean flag indicating if adversarial perturbation was detected.
        forensic_score: Anomaly or adversarial score in the range [0.0, 1.0].
        artifact_map: Optional localization saliency/attribution map indicating regions
            with anomalous or adversarial artifacts.
            Expected tensor shape: [B, 1, H, W] or [1, H, W] (single-channel spatial attribution).
    """

    diagnosis_label: str
    diagnosis_confidence: float
    is_adversarial: bool
    forensic_score: float
    artifact_map: Optional[torch.Tensor] = None


def detect(image: torch.Tensor, model: nn.Module) -> ScanResult:
    """Execute dual-stream forensic inspection and classification on an input image tensor.

    Args:
        image: Input image tensor of expected shape [B, C, H, W] (or single instance [C, H, W]),
            where:
                - B: Batch size (default 1 for single scan inference)
                - C: Channel dimension (1 for grayscale/radiographs, 3 for RGB/dermoscopy)
                - H: Image height
                - W: Image width
            Values are expected to be float tensors normalized to [0.0, 1.0].
        model: Target neural network model (nn.Module) for primary diagnostic classification.
            Accepts image tensor of shape [B, C, H, W] and outputs class logits or predictions.

    Returns:
        ScanResult: Result object containing the diagnostic classification, adversarial detection
            status, forensic anomaly score, and optional spatial artifact heatmap tensor of shape [B, 1, H, W].

    Raises:
        NotImplementedError: Implementation logic has not yet been loaded.
    """
    raise NotImplementedError("GhostScan detector pipeline has not been implemented yet.")
