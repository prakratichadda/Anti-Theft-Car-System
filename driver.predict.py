#!/usr/bin/env python3
"""
driver_fingerprint.py

- Train a driver identification model on features.csv
- Save the trained pipeline (model + scaler + label encoder)
- Predict a driver from a new fingerprint input
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# -------------------------------
# STEP 1: Train model on features.csv
# -------------------------------
def train_driver_fingerprint(filepath="features.csv", model_path="driver_fingerprint.joblib"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)

    if "Target" not in df.columns:
        raise ValueError("Dataset must include a 'Target' column with driver labels.")

    X = df.drop("Target", axis=1)
    y = df["Target"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Improved RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Driver Fingerprint Model Accuracy: {acc:.2f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix - Driver Fingerprint")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Save pipeline
    joblib.dump({"model": model, "scaler": scaler, "label_encoder": le, "features": X.columns.tolist()}, model_path)
    print(f"🔒 Saved driver fingerprint model to {model_path}")

# -------------------------------
# STEP 2: Predict driver
# -------------------------------
def predict_driver(model_path="driver_fingerprint.joblib", threshold=0.65):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    obj = joblib.load(model_path)
    model, scaler, le, feature_names = obj["model"], obj["scaler"], obj["label_encoder"], obj["features"]

    print(f"\nEnter values for {len(feature_names)} features in order: {feature_names}")
    user_input = input("Comma-separated values: ")
    features = [float(x.strip()) for x in user_input.split(",")]

    if len(features) != len(feature_names):
        print(f"⚠️ Expected {len(feature_names)} features but got {len(features)}")
        return

    sample = pd.DataFrame([features], columns=feature_names)
    sample_scaled = scaler.transform(sample)

    probas = model.predict_proba(sample_scaled)[0]
    max_prob = np.max(probas)
    pred_class = le.inverse_transform([np.argmax(probas)])[0]

    if max_prob < threshold:
        print(f"⚠️ Unknown driver detected (confidence={max_prob:.2f})")
    else:
        print(f"✅ Predicted driver: {pred_class} (confidence={max_prob:.2f})")

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    # Train on features.csv
    train_driver_fingerprint("features.csv")
    
    # Predict from new input
    predict_driver()
    
