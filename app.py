import os, torch, gradio as gr
from PIL import Image
from torchvision import transforms
from contracts import AttackType
from victim_model import VictimClassifier
from detector import DualStreamDetector
from attacks import generate_adversarial_batch
from explain import get_forensic_attribution, overlay_heatmap

CKPT_DIR = "checkpoints"
victim, detector = None, None
if os.path.exists(f"{CKPT_DIR}/victim.pt"):
    victim = VictimClassifier(num_classes=2, pretrained=False).eval()
    victim.load_state_dict(torch.load(f"{CKPT_DIR}/victim.pt", weights_only=True))
if os.path.exists(f"{CKPT_DIR}/detector.pt"):
    detector = DualStreamDetector(pretrained_backbone=False).eval()
    detector.load_state_dict(torch.load(f"{CKPT_DIR}/detector.pt", weights_only=True))

CLASSES, DET_CLASSES = ["NORMAL", "PNEUMONIA"], ["CLEAN", "FGSM", "PGD"]
trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def analyze(image):
    if victim is None or detector is None:
        gr.Warning("Model checkpoints not loaded from checkpoints/ directory!")
        return [None] * 18
    if image is None:
        return [None] * 18
    if image.mode != "RGB":
        gr.Warning("Uploaded image is not RGB mode. Converting to RGB.")
        image = image.convert("RGB")
    t = trans(image).unsqueeze(0)
    pred_y = victim(t).argmax(dim=-1)
    results = []
    for atk in [AttackType.CLEAN, AttackType.FGSM, AttackType.PGD]:
        adv_t, _ = generate_adversarial_batch(victim, t, pred_y, atk, eps=8 / 255)
        with torch.no_grad():
            v_logits = victim(adv_t)
            v_p = torch.softmax(v_logits, dim=-1)[0]
            v_idx = v_logits.argmax(dim=-1).item()
            d_logits = detector(adv_t)
            d_p = torch.softmax(d_logits, dim=-1)[0]
            d_idx = d_logits.argmax(dim=-1).item()
        is_adv = bool(d_idx != 0)
        f_score = float(1.0 - d_p[0].item())
        hmap = get_forensic_attribution(detector, adv_t, target_class=max(1, d_idx))
        overlay = overlay_heatmap(adv_t, hmap)
        clean_view = overlay_heatmap(adv_t, torch.zeros_like(hmap), alpha=0.0)
        results.extend([clean_view, f"{CLASSES[v_idx]} ({v_p[v_idx]:.1%})", is_adv, f_score, DET_CLASSES[d_idx], overlay])
    return results

def make_panel(name):
    with gr.Tab(name):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Victim Model")
                img = gr.Image(label=f"{name} Image", type="pil")
                diag = gr.Textbox(label="Predicted Diagnosis & Confidence")
            with gr.Column():
                gr.Markdown("### GhostScan Verdict")
                is_adv = gr.Checkbox(label="Is Adversarial?", interactive=False)
                score = gr.Slider(0.0, 1.0, label="Forensic Score (0-1)", interactive=False)
                atk = gr.Textbox(label="Detected Attack Type")
            with gr.Column():
                gr.Markdown("### Forensic Attribution")
                attr = gr.Image(label="Captum Artifact Heatmap", type="pil")
    return [img, diag, is_adv, score, atk, attr]

samples = [os.path.join("samples", f) for f in os.listdir("samples") if f.endswith(".png")] if os.path.exists("samples") else []
light_theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    button_primary_background_fill="*primary_600",
    button_primary_text_color="white",
)
light_css = ":root, body, html { color-scheme: light !important; } .dark { background-color: #f8fafc !important; color: #0f172a !important; }"
light_js = "() => { document.documentElement.classList.remove('dark'); document.body.classList.remove('dark'); localStorage.setItem('gradio-color-mode', 'light'); }"

with gr.Blocks(title="GhostScan Forensic Auditor") as demo:
    gr.Markdown("# GhostScan: Dual-Stream Adversarial Auditor")
    with gr.Row():
        inp = gr.Image(label="Upload Chest X-ray", type="pil", sources=["upload", "clipboard"])
    if samples:
        gr.Examples(examples=samples, inputs=inp)
    with gr.Tabs():
        out_comps = make_panel("CLEAN") + make_panel("FGSM") + make_panel("PGD")
    inp.change(analyze, inputs=inp, outputs=out_comps)

if __name__ == "__main__":
    demo.launch(theme=light_theme, css=light_css, js=light_js)

