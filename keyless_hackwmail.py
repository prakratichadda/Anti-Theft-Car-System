#!/usr/bin/env python3
"""
keyless_pipeline.py
Extended with email alerts for attack detection.
"""

import argparse, os, shutil, random, json
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ---------- CONFIG ----------
DEFAULT_RAW_DIR = "KeFRA Images Key-fob RKE Replay Attack"
OUT_ROOT = Path("data")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Email config (⚠️ Replace with your details)
SENDER_EMAIL = "REDACTED_SENDER_EMAIL"
APP_PASSWORD = "REDACTED_APP_PASSWORD"   # Gmail App Password
ALERT_RECIPIENT = "REDACTED_RECIPIENT_EMAIL"
# ----------------------------

def send_email_alert(subject, body, to_email=ALERT_RECIPIENT):
    """Send email alert when attack is detected."""
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
        server.quit()

        print(f"📩 Email sent to {to_email} → {subject}")
    except Exception as e:
        print("❌ Failed to send email:", str(e))

# ----------------------------
# Label inference (same as before)
# ----------------------------
def infer_label_from_folder(folder_name: str):
    name = folder_name.lower()
    if "fake" in name or "high" in name or "low" in name:
        return "attack"
    if "real" in name:
        return "legit"
    return "attack"

# ----------------------------
# Preprocessing (same as before)
# ----------------------------
def preprocess(raw_dir: str):
    # (preprocessing code unchanged from your version)
    ...

# ----------------------------
# Training (same as before)
# ----------------------------
def train(data_dir: str, epochs: int = 15, batch_size: int = 32, lr: float = 1e-4, out_path: str = "rf_model.pth"):
    # (training code unchanged from your version)
    ...

# ----------------------------
# PREDICT + EMAIL ALERT
# ----------------------------
def predict(model_path: str, img_path: str):
    ckpt = torch.load(model_path, map_location="cpu")
    classes = ckpt.get("classes")

    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        prediction = classes[idx]
        confidence = probs[idx]

        print("Prediction:", prediction, f"({confidence:.3f})")
        for i, c in enumerate(classes):
            print(f"  {c}: {probs[i]:.3f}")

        # 🚨 Email alert if attack detected
        if prediction == "attack" and confidence > 0.70:
            subject = "🚨 ALERT: Keyless Car Attack Detected!"
            body = f"An attack was detected with confidence {confidence:.2f}.\n\nImage: {img_path}"
            send_email_alert(subject, body, ALERT_RECIPIENT)
        elif prediction == "legit":
            print("✅ Legitimate access detected, no alert sent.")

# ----------------------------
# CLI (unchanged)
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preprocess","train","predict"], required=True)
    parser.add_argument("--raw_dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="rf_model.pth")
    parser.add_argument("--model", default="rf_model.pth")
    parser.add_argument("--img", default=None)
    args = parser.parse_args()

    if args.mode == "preprocess":
        preprocess(args.raw_dir)
    elif args.mode == "train":
        train(args.data_dir, epochs=args.epochs, batch_size=args.batch, lr=args.lr, out_path=args.out)
    elif args.mode == "predict":
        if args.img is None:
            raise SystemExit("For predict mode you must supply --img path/to/image")
        predict(args.model, args.img)

if __name__ == "__main__":
    main()
