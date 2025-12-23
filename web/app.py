import sys
import os

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
from torchvision import transforms
import timm
import torch.nn.functional as F

from gradcam_efficientnet import GradCAM

# -------------------------
# App & Paths
# -------------------------
app = Flask(__name__)
UPLOAD_DIR = "static/uploads"
RESULT_DIR = "static/results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224

# -------------------------
# Load Model
# -------------------------
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
model.load_state_dict(
    torch.load("../models/efficientnet_deepfake.pth", map_location=DEVICE)
)
model.eval().to(DEVICE)

gradcam = GradCAM(model, model.conv_head)

# -------------------------
# Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# -------------------------
# Video Analyzer
# -------------------------
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)

    fake_frames = 0
    real_frames = 0
    total = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)

        fake_prob = probs[0][1].item()

        if fake_prob >= 0.65:
            fake_frames += 1
        else:
            real_frames += 1

    cap.release()

    if total == 0:
        return "UNCERTAIN", 0

    fake_ratio = fake_frames / total

    if fake_ratio >= 0.6:
        return "FAKE", fake_ratio * 100
    elif fake_ratio >= 0.4:
        return "UNCERTAIN", fake_ratio * 100
    else:
        return "REAL", (1 - fake_ratio) * 100

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    original_img = None
    result_img = None

    if request.method == "POST":
        file = request.files["file"]
        filename = file.filename.lower()
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        file.save(file_path)

        # ---------- VIDEO ----------
        if filename.endswith((".mp4", ".avi", ".mov")):
            prediction, confidence = analyze_video(file_path)

        # ---------- IMAGE ----------
        else:
            img = Image.open(file_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)

            fake_prob = probs[0][1].item()
            real_prob = probs[0][0].item()

            if fake_prob >= 0.65:
                prediction = "FAKE"
                confidence = fake_prob * 100
            elif 0.45 < fake_prob < 0.65:
                prediction = "UNCERTAIN"
                confidence = fake_prob * 100
            else:
                prediction = "REAL"
                confidence = real_prob * 100

            cam = gradcam.generate(x, class_idx=1)[0].detach().cpu().numpy()
            cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

            img_np = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

            result_img = os.path.join(RESULT_DIR, "result.jpg")
            cv2.imwrite(result_img, overlay)
            original_img = file_path

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=round(confidence, 2) if confidence else None,
        original_img=original_img,
        result_img=result_img
    )

# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
