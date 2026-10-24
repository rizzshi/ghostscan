---
title: GhostScan
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
---

# GhostScan

GhostScan is a dual-stream forensic auditing and adversarial robustness framework designed for medical imaging and deep learning diagnostic pipelines.

---

## Motivation

Deep neural networks in clinical imaging diagnostics are susceptible to imperceptible adversarial perturbations (such as FGSM and PGD) and distribution shifts. These perturbations can induce catastrophic misclassifications while appearing entirely benign to human practitioners.

GhostScan addresses this vulnerability by decoupling clinical diagnostic classification from adversarial anomaly detection, providing clinicians and researchers with:
- Simultaneous diagnostic inference and adversarial threat detection.
- Localized spatial artifact attribution maps pinpointing anomalous perturbations.
- A standardized, locked interface contract for rapid evaluation across models and modalities.

---

## Architecture (dual-stream)

GhostScan employs a synchronized **dual-stream** architecture:

1. **Diagnostic Stream**:
   - Ingests the normalized image tensor of shape `[B, C, H, W]`.
   - Propagates features through the primary diagnostic backbone to generate clinical predictions and associated class confidence scores.

2. **Forensic / Artifact Stream**:
   - Inspects high-frequency perturbations and gradient saliency across the input tensor `[B, C, H, W]`.
   - Flags adversarial manipulation (`is_adversarial: bool`) and assigns a continuous forensic score (`forensic_score: float`).
   - Produces single-channel spatial artifact heatmaps of shape `[B, 1, H, W]` highlighting localized adversarial perturbations.

Both streams converge into the unified `ScanResult` interface contract.

---

## Quickstart

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-org/ghostscan.git
cd ghostscan
pip install -r requirements.txt
```

### 2. Using the Interface Contract

```python
import torch
import torch.nn as nn
from contracts import detect, AttackType, ScanResult

# Example placeholder image tensor: [B, C, H, W]
image = torch.rand(1, 3, 224, 224)

# Example placeholder model
model = nn.Identity()

# Detector interface invocation
try:
    result: ScanResult = detect(image=image, model=model)
except NotImplementedError:
    print("Interface contract locked. Pipeline implementation in progress.")
```

---

## Demo

An interactive Gradio web application will be provided for real-time forensic scanning, attack simulation (Clean, FGSM, PGD), and artifact heatmap visualization.

Launch the demo interface (available in future milestone):

```bash
python app.py
```

---

## Citation

If you use GhostScan in your research or clinical audit pipelines, please cite:

```bibtex
@software{ghostscan2026,
  author = {GhostScan Contributors},
  title = {GhostScan: Dual-Stream Adversarial Forensics and Diagnostics},
  year = {2026},
  url = {https://github.com/your-org/ghostscan}
}
```
