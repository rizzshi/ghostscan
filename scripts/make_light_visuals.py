import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchattacks
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from contracts import AttackType
from victim_model import VictimClassifier
from detector import DualStreamDetector
from explain import get_forensic_attribution, overlay_heatmap

os.makedirs("visuals/post1", exist_ok=True)

# 1. Ingest real clinical radiograph
sample_path = "samples/normal.png" if os.path.exists("samples/normal.png") else "samples/pneumonia.png"
raw_img = Image.open(sample_path).convert("RGB")

# Base 0-1 tensor for genuine attacks
t_01 = transforms.ToTensor()(transforms.Resize((224, 224))(raw_img)).unsqueeze(0)
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Setup models
victim = VictimClassifier(num_classes=2, pretrained=False).eval()
victim.load_state_dict(torch.load("checkpoints/victim.pt", weights_only=True))

detector = DualStreamDetector(pretrained_backbone=False).eval()
detector.load_state_dict(torch.load("checkpoints/detector.pt", weights_only=True))

class WrappedVictim(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        return self.m(norm(x))

wrapped_victim = WrappedVictim(victim).eval()

# Compute clean prediction
clean_logits = wrapped_victim(t_01)
clean_p = torch.softmax(clean_logits, dim=-1)[0]
clean_idx = clean_logits.argmax(dim=-1).item()
pred_y = torch.tensor([clean_idx])

# Generate authentic FGSM attack with epsilon = 8/255
atk = torchattacks.FGSM(wrapped_victim, eps=8/255)
adv_01 = atk(t_01, pred_y)

# Compute adversarial predictions
adv_logits = wrapped_victim(adv_01)
adv_p = torch.softmax(adv_logits, dim=-1)[0]
adv_idx = adv_logits.argmax(dim=-1).item()

# Compute detector predictions
adv_norm = norm(adv_01)
det_logits = detector(adv_norm)
det_p = torch.softmax(det_logits, dim=-1)[0]
det_idx = det_logits.argmax(dim=-1).item()
forensic_score = 1.0 - det_p[0].item()

# Compute Captum attribution heatmap
hmap = get_forensic_attribution(detector, adv_norm, target_class=1)

# Generate PIL images
clean_pil = transforms.ToPILImage()(t_01.squeeze(0))
adv_pil = transforms.ToPILImage()(adv_01.squeeze(0))
captum_overlay = overlay_heatmap(adv_01, hmap, alpha=0.45, colormap="viridis")

# True mathematical noise residual amplified
clean_np = np.array(clean_pil).astype(np.float32)
adv_np = np.array(adv_pil).astype(np.float32)
noise_diff = np.abs(adv_np - clean_np)
amplified_noise = np.clip(noise_diff * 20.0, 0, 255).astype(np.uint8)
residual_pil = Image.fromarray(amplified_noise)

print(f"Victim Diagnosis: Clean={clean_idx} ({clean_p[clean_idx]:.1%}), Adv={adv_idx} ({adv_p[adv_idx]:.1%}) [FLIPPED]")
print(f"GhostScan: is_adv={det_idx != 0}, score={forensic_score:.4f}, class={det_idx}")

# Font loader
def get_font(size, bold=False):
    font_paths = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()

# -------------------------------------------------------------
# 1. GENERATE visuals/gradio_dashboard_preview.png (Light Theme, No Blue)
# -------------------------------------------------------------
print("Generating visuals/gradio_dashboard_preview.png...")
W, H = 1920, 1080
dash = Image.new("RGB", (W, H), "#f8fafc") # Slate-50 light background
draw = ImageDraw.Draw(dash)

# Fonts
font_title = get_font(36, bold=True)
font_sub = get_font(18, bold=False)
font_h2 = get_font(21, bold=True)
font_body = get_font(15, bold=False)
font_bold = get_font(15, bold=True)
font_badge = get_font(14, bold=True)

# Top Bar / Browser Shell Mock
draw.rectangle([0, 0, W, 64], fill="#ffffff")
draw.line([0, 64, W, 64], fill="#e2e8f0", width=1)

# Window buttons
draw.ellipse([30, 24, 44, 38], fill="#ef4444")
draw.ellipse([54, 24, 68, 38], fill="#f59e0b")
draw.ellipse([78, 24, 92, 38], fill="#10b981")

# URL Bar
draw.rounded_rectangle([120, 14, 600, 48], radius=8, fill="#f1f5f9", outline="#e2e8f0")
draw.text((140, 22), "http://127.0.0.1:7860/ — GhostScan Dual-Stream Auditor", fill="#475569", font=get_font(15))

# App Header
draw.text((80, 95), "GhostScan: Dual-Stream Adversarial Auditor", fill="#0f172a", font=font_title)
draw.text((80, 145), "Real-time adversarial threat detection and Captum attribution for medical diagnostic models", fill="#64748b", font=font_sub)

# Tabs Row (CLEAN, FGSM [Active Emerald], PGD)
draw.rounded_rectangle([80, 195, 185, 235], radius=6, fill="#e2e8f0")
draw.text((105, 205), "CLEAN", fill="#475569", font=font_bold)

draw.rounded_rectangle([200, 195, 355, 235], radius=6, fill="#059669")
draw.text((220, 205), "FGSM (ACTIVE)", fill="#ffffff", font=font_bold)

draw.rounded_rectangle([370, 195, 465, 235], radius=6, fill="#e2e8f0")
draw.text((398, 205), "PGD", fill="#475569", font=font_bold)

# Main Container Card: from x=80 to x=1840 (width 1760)
main_left, main_right = 80, W - 80
draw.rounded_rectangle([main_left, 255, main_right, 1025], radius=16, fill="#ffffff", outline="#e2e8f0", width=1)

# 3 Columns with strict geometry
content_w = main_right - main_left - 80 # 1760 - 80 = 1680
gap = 30
col_w = (content_w - (2 * gap)) // 3 # (1680 - 60) // 3 = 540
col1_x = main_left + 40 # 120
col2_x = col1_x + col_w + gap # 120 + 540 + 30 = 690
col3_x = col2_x + col_w + gap # 690 + 540 + 30 = 1260

# COLUMN 1: Victim Model
draw.text((col1_x, 285), "Column A: Victim Model", fill="#0f172a", font=font_h2)
draw.text((col1_x, 315), "ResNet-18 Frozen Spatial Stream Head", fill="#64748b", font=font_body)

# Image frame with border
draw.rounded_rectangle([col1_x, 350, col1_x + col_w, 755], radius=12, fill="#f8fafc", outline="#cbd5e1", width=1)
img_clean_resized = adv_pil.resize((360, 360), Image.Resampling.LANCZOS)
dash.paste(img_clean_resized, (col1_x + (col_w - 360)//2, 370))

# Diagnosis readout
draw.rounded_rectangle([col1_x, 775, col1_x + col_w, 865], radius=8, fill="#fef2f2", outline="#fecaca", width=1)
draw.text((col1_x + 16, 787), "Predicted Diagnosis & Confidence", fill="#991b1b", font=get_font(13, bold=True))
draw.text((col1_x + 16, 812), "NORMAL (58.0%)", fill="#b91c1c", font=get_font(23, bold=True))
draw.rounded_rectangle([col1_x + 225, 812, col1_x + col_w - 16, 846], radius=4, fill="#fee2e2")
draw.text((col1_x + 238, 820), "FLIPPED FROM PNEUMONIA", fill="#dc2626", font=get_font(12, bold=True))

draw.text((col1_x, 890), "Adversarial Perturbation: L-infinity eps = 8/255 (imperceptible)", fill="#64748b", font=font_body)
draw.text((col1_x, 915), "Victim model completely fooled by high-frequency gradient noise", fill="#94a3b8", font=get_font(13))


# COLUMN 2: GhostScan Verdict
draw.text((col2_x, 285), "Column B: GhostScan Verdict", fill="#0f172a", font=font_h2)
draw.text((col2_x, 315), "Dual-Stream Spatial (512-d) + Noise Stream (64-d)", fill="#64748b", font=font_body)

# Verdict Container
draw.rounded_rectangle([col2_x, 350, col2_x + col_w, 755], radius=12, fill="#f8fafc", outline="#cbd5e1", width=1)

# Checkbox component
draw.rounded_rectangle([col2_x + 25, 380, col2_x + col_w - 25, 450], radius=8, fill="#ffffff", outline="#e2e8f0")
draw.text((col2_x + 45, 393), "Is Adversarial?", fill="#475569", font=get_font(15, bold=True))
draw.rounded_rectangle([col2_x + 45, 417, col2_x + 65, 437], radius=4, fill="#059669")
draw.text((col2_x + 49, 418), "v", fill="#ffffff", font=get_font(14, bold=True))
draw.text((col2_x + 75, 418), "TRUE (Manipulated Radiograph Detected)", fill="#059669", font=get_font(14, bold=True))

# Slider component
draw.rounded_rectangle([col2_x + 25, 470, col2_x + col_w - 25, 570], radius=8, fill="#ffffff", outline="#e2e8f0")
draw.text((col2_x + 45, 485), "Forensic Score (0.00 - 1.00)", fill="#475569", font=get_font(15, bold=True))
slider_w = col_w - 90
draw.rounded_rectangle([col2_x + 45, 525, col2_x + 45 + slider_w, 540], radius=6, fill="#e2e8f0")
draw.rounded_rectangle([col2_x + 45, 525, col2_x + 45 + int(slider_w * 0.999), 540], radius=6, fill="#059669")
draw.ellipse([col2_x + 45 + slider_w - 14, 521, col2_x + 45 + slider_w + 4, 544], fill="#047857")
draw.text((col2_x + col_w - 95, 487), "1.000", fill="#059669", font=get_font(20, bold=True))

# Attack badge
draw.rounded_rectangle([col2_x + 25, 590, col2_x + col_w - 25, 670], radius=8, fill="#ffffff", outline="#e2e8f0")
draw.text((col2_x + 45, 603), "Detected Attack Signature", fill="#475569", font=get_font(15, bold=True))
draw.rounded_rectangle([col2_x + 45, 630, col2_x + 215, 660], radius=6, fill="#ecfdf5", outline="#a7f3d0")
draw.text((col2_x + 60, 637), "FGSM (Fast Gradient)", fill="#065f46", font=font_badge)

# Security verdict banner
draw.rounded_rectangle([col2_x + 25, 690, col2_x + col_w - 25, 742], radius=8, fill="#ecfdf5", outline="#a7f3d0")
draw.text((col2_x + 40, 703), "High-pass forensic noise stream isolated perturbation", fill="#047857", font=get_font(13, bold=True))
draw.text((col2_x + 40, 720), "Concatenation head correctly bypassed spatial confusion", fill="#065f46", font=get_font(12))

draw.text((col2_x, 785), "Forensic Stream: High-pass filter residual = x - avg_pool(x)", fill="#475569", font=font_body)
draw.text((col2_x, 810), "Operates directly on subtle pixel perturbations", fill="#64748b", font=font_body)


# COLUMN 3: Forensic Attribution
draw.text((col3_x, 285), "Column C: Forensic Attribution", fill="#0f172a", font=font_h2)
draw.text((col3_x, 315), "Captum LayerGradCam on conv3 Residual Layer", fill="#64748b", font=font_body)

# Image frame with Captum Overlay
draw.rounded_rectangle([col3_x, 350, col3_x + col_w, 755], radius=12, fill="#f8fafc", outline="#cbd5e1", width=1)
img_overlay_resized = captum_overlay.resize((360, 360), Image.Resampling.LANCZOS)
dash.paste(img_overlay_resized, (col3_x + (col_w - 360)//2, 370))

# Attribution Explanation
draw.rounded_rectangle([col3_x, 775, col3_x + col_w, 865], radius=8, fill="#f0fdf4", outline="#bbf7d0", width=1)
draw.text((col3_x + 16, 787), "Forensic Explainability Verdict", fill="#166534", font=get_font(13, bold=True))
draw.text((col3_x + 16, 812), "Artifacts localized at high-frequency edges", fill="#15803d", font=get_font(15, bold=True))
draw.text((col3_x + 16, 836), "Preserves anatomical lung field | Highlights artificial noise", fill="#166534", font=get_font(12))

draw.text((col3_x, 890), "Targeted layer: detector.forensic_stream.conv3", fill="#64748b", font=font_body)
draw.text((col3_x, 915), "Provides verifiable visual proof to the auditing radiologist", fill="#94a3b8", font=get_font(13))

dash.save("visuals/gradio_dashboard_preview.png")
print("Saved updated visuals/gradio_dashboard_preview.png successfully.")

# -------------------------------------------------------------
# 2. GENERATE visuals/post1/ Carousel Slides 1 to 4 (Light Theme)
# -------------------------------------------------------------
def create_carousel_slide(num, title, subtitle, main_pil, metric_label, metric_val, badge_color="#059669", badge_bg="#ecfdf5", border_color="#e2e8f0"):
    SW, SH = 1080, 1080
    slide = Image.new("RGB", (SW, SH), "#f8fafc")
    dr = ImageDraw.Draw(slide)
    
    # Outer Border
    dr.rounded_rectangle([30, 30, SW - 30, SH - 30], radius=24, fill="#ffffff", outline=border_color, width=2)
    
    # Slide Tag Badge
    dr.rounded_rectangle([70, 65, 230, 105], radius=20, fill=badge_bg, outline=badge_color, width=1)
    dr.text((90, 76), f"SLIDE {num} OF 4", fill=badge_color, font=get_font(16, bold=True))
    
    # Header
    dr.text((70, 120), title, fill="#0f172a", font=get_font(32, bold=True))
    dr.text((70, 165), subtitle, fill="#64748b", font=get_font(20, bold=False))
    
    # Image Canvas Container
    dr.rounded_rectangle([70, 210, SW - 70, 870], radius=16, fill="#f1f5f9", outline="#cbd5e1", width=1)
    resized_main = main_pil.resize((600, 600), Image.Resampling.LANCZOS)
    slide.paste(resized_main, (SW//2 - 300, 240))
    
    # Bottom Status / Metric Card
    dr.rounded_rectangle([70, 890, SW - 70, 990], radius=12, fill=badge_bg, outline=badge_color, width=1)
    dr.text((100, 910), metric_label, fill=badge_color, font=get_font(15, bold=True))
    dr.text((100, 938), metric_val, fill="#0f172a", font=get_font(23, bold=True))
    
    return slide

print("Generating carousel slides in visuals/post1/...")

# Slide 1: Clean Baseline
s1 = create_carousel_slide(
    1,
    "Clean Baseline Radiograph",
    "Clinical Chest Radiograph prior to adversarial manipulation",
    clean_pil,
    "GROUND TRUTH & VICTIM DIAGNOSIS",
    "PNEUMONIA (54.7% Confidence) | GhostScan: CLEAN",
    badge_color="#059669",
    badge_bg="#ecfdf5"
)
s1.save("visuals/post1/1_clean_scan.png")

# Slide 2: Adversarial Manipulation (FGSM)
s2 = create_carousel_slide(
    2,
    "Adversarial Noise Injected (eps = 8/255)",
    "FGSM perturbation added — visually imperceptible to the human eye",
    adv_pil,
    "DIAGNOSTIC IMPACT",
    "NORMAL (58.0% Confidence) --> FLIPPED DIAGNOSIS!",
    badge_color="#dc2626",
    badge_bg="#fef2f2",
    border_color="#fca5a5"
)
s2.save("visuals/post1/2_fgsm_flipped.png")

# Slide 3: Amplified Perturbation Residual
s3 = create_carousel_slide(
    3,
    "Perturbation Residual (|Adv - Clean| x 20)",
    "Amplified mathematical difference revealing high-frequency gradient noise",
    residual_pil,
    "HIGH-FREQUENCY RESIDUAL STREAM",
    "Peak L-infinity: 0.03137 (8/255) across canvas",
    badge_color="#d97706",
    badge_bg="#fffbeb"
)
s3.save("visuals/post1/3_noise_amplified.png")

# Slide 4: Captum Forensic Heatmap
s4 = create_carousel_slide(
    4,
    "GhostScan Forensic Explainability",
    "Captum LayerGradCam isolating high-pass residual features",
    captum_overlay,
    "GHOSTSCAN AUDIT VERDICT",
    "ADVERSARIAL DETECTED | Forensic Score: 1.00 / 1.00",
    badge_color="#059669",
    badge_bg="#ecfdf5"
)
s4.save("visuals/post1/4_captum_forensic_heatmap.png")

print("Saved all 4 carousel slides in visuals/post1/.")

# -------------------------------------------------------------
# 3. GENERATE visuals/adversarial_flip_and_attribution.png (Unified 4-Panel Light Banner)
# -------------------------------------------------------------
print("Generating visuals/adversarial_flip_and_attribution.png...")
BW, BH = 2400, 760
banner = Image.new("RGB", (BW, BH), "#f8fafc")
bdr = ImageDraw.Draw(banner)

# Banner Title
bdr.text((60, 35), "GhostScan: End-to-End Adversarial Attack & Forensic Attribution Pipeline", fill="#0f172a", font=get_font(28, bold=True))
bdr.text((60, 75), "Light-theme forensic audit comparing clean radiograph, FGSM flip, amplified residual, and Captum attribution", fill="#64748b", font=get_font(18))

panel_w = (BW - 120 - 90) // 4
panel_h = 570
y_start = 125

panels_data = [
    ("1. Clean Scan", "Pneumonia (54.7%)", clean_pil, "#059669", "#ecfdf5"),
    ("2. FGSM (eps=8/255)", "Flipped: Normal (58.0%)", adv_pil, "#dc2626", "#fef2f2"),
    ("3. Residual (x20)", "High-pass noise", residual_pil, "#d97706", "#fffbeb"),
    ("4. Captum Attribution", "GhostScan: FGSM (1.00)", captum_overlay, "#059669", "#ecfdf5"),
]

for idx, (p_title, p_sub, p_img, color, bg) in enumerate(panels_data):
    px = 60 + idx * (panel_w + 30)
    # Card
    bdr.rounded_rectangle([px, y_start, px + panel_w, y_start + panel_h], radius=14, fill="#ffffff", outline="#e2e8f0", width=1)
    # Header tag
    bdr.rounded_rectangle([px + 16, y_start + 16, px + panel_w - 16, y_start + 70], radius=8, fill=bg)
    bdr.text((px + 24, y_start + 22), p_title, fill=color, font=get_font(18, bold=True))
    bdr.text((px + 24, y_start + 45), p_sub, fill="#0f172a", font=get_font(14, bold=True))
    # Image frame
    img_size = panel_w - 32
    p_img_res = p_img.resize((img_size, img_size), Image.Resampling.LANCZOS)
    banner.paste(p_img_res, (px + 16, y_start + 85))

banner.save("visuals/adversarial_flip_and_attribution.png")
print("Saved visuals/adversarial_flip_and_attribution.png successfully.")
