# GhostScan: Dual-Stream Adversarial Forensics and Robustness Framework for Medical Vision

GhostScan is a dual-stream forensic auditing and adversarial robustness framework designed for clinical medical imaging pipelines. It decouples diagnostic classification from adversarial anomaly detection, providing simultaneous clinical inference, perturbation detection, and localized forensic attribution heatmaps.

---

## 1. Problem Statement & Clinical Motivation

Deep neural networks deployed in clinical radiology and diagnostic imaging are susceptible to imperceptible adversarial perturbations (such as FGSM and PGD). Under an L-infinity perturbation budget as small as epsilon = 8/255 (~0.031), standard vision models can be induced to flip their clinical diagnoses from Normal to Pneumonia with over 98% confidence.

Crucially, standard explainability techniques (such as vanilla Grad-CAM on the diagnostic backbone) fail during adversarial attacks: they project plausible-looking heatmaps directly over healthy anatomical tissue, misleading practitioners into believing an adversarial hallucination is clinically grounded.

GhostScan eliminates this vulnerability by running an orthogonal, synchronized forensic stream that strips anatomical low frequencies, isolates high-frequency noise residuals, and produces forensic attribution heatmaps pinpointing the injected perturbation artifacts.

---

## 2. System Architecture

GhostScan implements a synchronized dual-stream architecture converging into a standardized evaluation contract:

![Dual-Stream Architecture](https://raw.githubusercontent.com/rizzshi/ghostscan/main/visuals/dual_stream_architecture.png)

### Architectural Components

1. **Spatial Diagnostic Stream**:
   - Backbone: Pretrained ResNet-18 (frozen during detector training to prevent catastrophic forgetting).
   - Ingests normalized image tensor of shape `[B, 3, H, W]`.
   - Extracts a 512-dimensional semantic spatial representation.

2. **Forensic Noise Stream**:
   - Computes an out-of-place high-pass spatial noise residual via average-pooling subtraction:
     $$\text{Residual}(x) = x - \text{AvgPool}_{3\times 3, s=1, p=1}(x)$$
   - A dedicated 3-layer convolutional network (Conv2D -> BatchNorm -> ReLU) processes the high-pass residual into a 64-dimensional forensic noise signature.

3. **Multi-Head Fusion & Classification**:
   - Concatenates spatial features (512-d) and forensic features (64-d) into a 576-dimensional joint representation.
   - Fully connected projection head: `Linear(576, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 3)` classifying into Clean (0), FGSM (1), or PGD (2).
   - Computes a continuous forensic score:
     $$\text{ForensicScore} = 1.0 - P(\text{Clean})$$
   - Anomaly flag: `is_adversarial = True` if predicted class != Clean or ForensicScore >= 0.50.

4. **Forensic Attribution Engine**:
   - Hooks Captum's `LayerGradCam` to the final convolutional layer of the forensic stream (`detector.forensic_stream.conv3`).
   - Generates single-channel attribution heatmaps highlighting localized perturbation patterns without highlighting healthy anatomy.

---

## 3. Experimental Benchmarks & Results

Below is the comparative evaluation of a fine-tuned ResNet-18 diagnostic victim model evaluated with and without GhostScan under Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks:

![Adversarial Flip and Attribution](https://raw.githubusercontent.com/rizzshi/ghostscan/main/visuals/adversarial_flip_and_attribution.png)

### Quantitative Evaluation Summary

| Input Condition | Perturbation Budget | Diagnostic Model Prediction | Diagnostic Confidence | GhostScan Forensic Score | Adversarial Detected | Attribution Target |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Clean Radiograph** | None ($\epsilon = 0$) | NORMAL | 93.7% | 0.01 | False | None (Nominal) |
| **FGSM Attack** | $\epsilon = 8/255$ | PNEUMONIA | 98.7% | 0.99 | True | Boundary & high-pass pixel noise |
| **PGD Attack** | $\epsilon = 8/255, \alpha = 2/255, 10\text{ steps}$ | PNEUMONIA | 99.4% | 0.99 | True | Multi-step iterative high-frequency noise |

### Forensic Metrics
- **Detector Adversarial Classification Accuracy**: > 98.5% across validation test sets.
- **Forensic Area Under ROC Curve (AUROC)**: > 0.99 for both one-step (FGSM) and iterative (PGD) perturbation regimes.
- **Diagnostic Invariance**: The primary clinical diagnostic backbone remains unchanged, preserving 100% of baseline clinical diagnostic performance on unperturbed inputs.

---

## 4. Interactive Demonstration Interface

GhostScan includes an interactive Gradio web application for real-time audit inspection:

![Gradio Dashboard Preview](https://raw.githubusercontent.com/rizzshi/ghostscan/main/visuals/gradio_dashboard_preview.png)

### Dashboard Capabilities
- Upload arbitrary clinical radiographs or select from included pre-loaded benchmarks.
- Select perturbation mode: Clean, FGSM (epsilon = 8/255), or PGD (iterative).
- View side-by-side diagnostic classification, forensic anomaly scores, and overlaid LayerGradCam attribution heatmaps.

To launch the dashboard locally:
```bash
python app.py
```

---

## 5. Repository Structure

```
ghostscan/
├── app.py                  # Gradio interactive web dashboard
├── contracts.py            # Locked dataclass interface contract (ScanResult, AttackType)
├── victim_model.py         # ResNet-18 diagnostic chest X-ray classifier
├── detector.py             # Dual-stream detector architecture and training logic
├── attacks.py              # Adversarial attack generation pipeline (FGSM, PGD via torchattacks)
├── explain.py              # Captum LayerGradCam forensic attribution and heatmap overlay
├── data.py                 # Dataset loading and radiograph preprocessing
├── push_to_hub.py          # Automated Hugging Face Hub deployment utility
├── requirements.txt        # Python dependency manifest
├── Makefile                # Quality checks, linting, and test execution
├── CITATION.cff            # Formal citation metadata
├── LICENSE                 # MIT License specification
├── checkpoints/
│   ├── victim.pt           # Fine-tuned ResNet-18 diagnostic model weights (44.8 MB)
│   └── detector.pt         # Trained DualStreamDetector weights (45.2 MB)
├── notebooks/
│   ├── GhostScan_Demo.ipynb# Executable interactive reproduction notebook
│   └── kernel-metadata.json# Kaggle deployment configuration
├── samples/                # Sample clean and pathological chest radiographs
└── visuals/                # High-resolution benchmark figures and architectural diagrams
```

---

## 6. Quickstart

### Installation

Clone the repository and install required dependencies:
```bash
git clone https://github.com/rizzshi/ghostscan.git
cd ghostscan
pip install -r requirements.txt
```

### Programmatic Usage

```python
import torch
from torchvision import transforms
from PIL import Image

from contracts import ScanResult, AttackType
from victim_model import VictimClassifier
from detector import DualStreamDetector
from explain import get_forensic_attribution

# 1. Load fine-tuned weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

victim = VictimClassifier(num_classes=2, pretrained=False).to(device).eval()
victim.load_state_dict(torch.load("checkpoints/victim.pt", map_location=device))

detector = DualStreamDetector(pretrained_backbone=False).to(device).eval()
detector.load_state_dict(torch.load("checkpoints/detector.pt", map_location=device))

# 2. Preprocess input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_tensor = transform(Image.open("samples/normal.png").convert("RGB")).unsqueeze(0).to(device)

# 3. Diagnostic inference
diag_logits = victim(image_tensor)
diag_probs = torch.softmax(diag_logits, dim=-1)
pred_class = "PNEUMONIA" if diag_logits.argmax().item() == 1 else "NORMAL"
confidence = diag_probs[0, diag_logits.argmax()].item()

# 4. Forensic adversarial inspection
det_logits = detector(image_tensor)
det_probs = torch.softmax(det_logits, dim=-1)
forensic_score = 1.0 - det_probs[0, 0].item()
is_adversarial = bool(det_logits.argmax().item() != 0 or forensic_score >= 0.50)

# 5. Localized forensic attribution
attribution_heatmap = get_forensic_attribution(detector, image_tensor)

print(f"Diagnosis: {pred_class} (Confidence: {confidence:.2%})")
print(f"Forensic Score: {forensic_score:.4f} | Is Adversarial: {is_adversarial}")
```

---

## 7. Model Hub & Platform Deployments

- **GitHub Repository**: [https://github.com/rizzshi/ghostscan](https://github.com/rizzshi/ghostscan)
- **Hugging Face Hub**: [https://huggingface.co/v2xvi/ghostscan](https://huggingface.co/v2xvi/ghostscan)
- **Kaggle Notebook**: [https://www.kaggle.com/code/tfisthis/ghostscan-adversarial-forensics](https://www.kaggle.com/code/tfisthis/ghostscan-adversarial-forensics)

---

## 8. Citation

If you use GhostScan in your research, audits, or clinical robustness pipelines, please cite:

```bibtex
@software{ghostscan2026,
  author = {GhostScan Contributors},
  title = {GhostScan: Dual-Stream Adversarial Forensics and Diagnostics for Medical Vision},
  year = {2026},
  url = {https://github.com/rizzshi/ghostscan}
}
```

---

## 9. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
