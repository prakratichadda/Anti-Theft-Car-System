#!/usr/bin/env python3
"""
keyless_hackwmail.py

Keyless entry relay-attack classifier (RF signal encodings -> ResNet18).
Alerting is delegated to alert_system.send_alert() rather than sending
email directly from here.

NOTE: preprocess() and train() are not implemented yet - this module
currently only supports predict() against a pre-trained checkpoint.
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

# ---------- CONFIG ----------
DEFAULT_RAW_DIR = "KeFRA Images Key-fob RKE Replay Attack"
OUT_ROOT = Path("data")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# Preprocessing - NOT YET IMPLEMENTED
# ----------------------------
def preprocess(raw_dir: str):
    raise NotImplementedError(
        "preprocess() has not been implemented yet - the training pipeline for "
        "keyless relay-attack detection is still in progress."
    )

# ----------------------------
# Training - NOT YET IMPLEMENTED
# ----------------------------
def train(data_dir: str, epochs: int = 15, batch_size: int = 32, lr: float = 1e-4, out_path: str = "rf_model.pth"):
    raise NotImplementedError(
        "train() has not been implemented yet - the training pipeline for "
        "keyless relay-attack detection is still in progress."
    )

# ----------------------------
# PREDICT
# ----------------------------
def predict(model_path: str, img_path: str):
    """Pure prediction function: returns a result dict. No side effects, so
    it's safe to call from a web UI as well as the CLI below."""
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
        confidence = float(probs[idx])

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {c: float(probs[i]) for i, c in enumerate(classes)},
    }

# ----------------------------
# CLI
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
        result = predict(args.model, args.img)
        print("Prediction:", result["prediction"], f"({result['confidence']:.3f})")
        for c, p in result["probabilities"].items():
            print(f"  {c}: {p:.3f}")

        if result["prediction"] == "attack" and result["confidence"] > 0.70:
            from alert_system import send_alert
            send_alert(
                title="Keyless Car Attack Detected!",
                messages=[f"Confidence: {result['confidence']:.2f}", f"Image: {args.img}"],
                level="high",
            )
        else:
            print("✅ Legitimate access detected, no alert sent.")

if __name__ == "__main__":
    main()
