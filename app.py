#!/usr/bin/env python3
"""
app.py

Single Streamlit shell over three independent ML pipelines:
  1. Suspicious Activity Detection  - CCTV footage (YOLOv8 person + pose)
  2. Driver Fingerprinting          - CAN bus / accelerometer telemetry (Random Forest)
  3. Keyless Relay-Attack Detection - RF signal encodings (ResNet18)

Alerting for all tabs goes through alert_system.send_alert(), which reads
ALERT_EMAIL / ALERT_EMAIL_PASS / ALERT_TO (and optional Twilio/Telegram vars)
from the environment - configure these as Streamlit Cloud "Secrets", never
commit them to the repo.
"""

import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Anti-Theft Car System", page_icon="🚗", layout="wide")

st.title("🚗 Anti-Theft Car System")
st.caption("Three independent ML pipelines, one demo shell.")

with st.sidebar:
    st.header("Alert channels")
    email_ok = bool(os.environ.get("ALERT_EMAIL") and os.environ.get("ALERT_EMAIL_PASS") and os.environ.get("ALERT_TO"))
    sms_ok = bool(os.environ.get("TWILIO_SID") and os.environ.get("TWILIO_TOKEN"))
    telegram_ok = bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))
    st.write("Email:", "✅ configured" if email_ok else "❌ not configured")
    st.write("SMS (Twilio):", "✅ configured" if sms_ok else "❌ not configured")
    st.write("Telegram:", "✅ configured" if telegram_ok else "❌ not configured")
    st.caption("Set these as Streamlit Cloud **Secrets** (Settings → Secrets), never commit them to the repo.")

tab1, tab2, tab3 = st.tabs([
    "Suspicious Activity Detection",
    "Driver Fingerprinting",
    "Keyless Relay-Attack Detection",
])

# =====================================================================
# TAB 1 - Suspicious Activity Detection
# =====================================================================
with tab1:
    st.subheader("Suspicious Activity Detection")
    st.markdown(
        "Trained/pretrained for **CCTV-style footage** (YOLOv8 person detection + pose "
        "estimation). Flags loitering near the vehicle, hand-near-face gestures, "
        "leaning/intrusion posture, and group-proximity threats."
    )

    source = st.radio("Input source", ["Upload image", "Webcam snapshot"], horizontal=True, key="sa_source")
    img_file = None
    if source == "Upload image":
        img_file = st.file_uploader("Upload a frame", type=["jpg", "jpeg", "png"], key="sa_upload")
    else:
        img_file = st.camera_input("Take a snapshot", key="sa_camera")
        st.caption(
            "Loitering detection needs continuous frames; a single snapshot mainly "
            "exercises the pose and group-proximity checks."
        )

    if img_file is not None:
        from suspicious_detection import analyze_frame  # deferred: pulls in torch/ultralytics

        pil_img = Image.open(img_file).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        with st.spinner("Running YOLO detection..."):
            annotated_frame, alerts = analyze_frame(frame, realtime=(source == "Webcam snapshot"))

        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="Detection output", use_container_width=True)

        if alerts:
            st.error("Suspicious activity detected:")
            for a in alerts:
                st.write(f"- {a}")

            snapshot_path = "sa_alert_snapshot.jpg"
            cv2.imwrite(snapshot_path, annotated_frame)
            from alert_system import send_alert
            send_alert(title="Suspicious Activity Detected", messages=alerts, level="high", image_path=snapshot_path)
            st.info("Alert dispatched via alert_system (subject to per-channel cooldowns).")
        else:
            st.success("No suspicious activity detected.")

# =====================================================================
# TAB 2 - Driver Fingerprinting
# =====================================================================
with tab2:
    st.subheader("Driver Fingerprinting")
    st.markdown(
        "Trained on **driver telemetry** (CAN bus / accelerometer feature vectors). A "
        "Random Forest builds a per-driver behavioral fingerprint and flags unrecognized drivers."
    )

    from driver_predict import predict_driver_from_features, get_feature_names, train_driver_fingerprint

    model_path = "driver_fingerprint.joblib"

    if not os.path.exists(model_path):
        st.warning(
            "No trained model found (`driver_fingerprint.joblib`). Upload a `features.csv` "
            "(must include a `Target` column with driver labels) to train one now."
        )
        csv_file = st.file_uploader("Upload features.csv to train", type=["csv"], key="df_train_csv")
        if csv_file is not None and st.button("Train model", key="df_train_btn"):
            with open("features.csv", "wb") as f:
                f.write(csv_file.getbuffer())
            with st.spinner("Training Random Forest..."):
                result = train_driver_fingerprint("features.csv", model_path=model_path, show_plot=False)
            st.success(f"Trained. Test accuracy: {result['accuracy']:.2f}")
            st.text(result["report"])
            st.pyplot(result["confusion_matrix_fig"])
            st.rerun()
    else:
        feature_names = get_feature_names(model_path)
        st.caption(f"Model expects {len(feature_names)} features: {', '.join(feature_names)}")

        input_mode = st.radio("Input method", ["Manual sliders", "Upload single-row CSV"], horizontal=True, key="df_input_mode")

        features = None
        if input_mode == "Manual sliders":
            features = []
            cols = st.columns(3)
            for i, name in enumerate(feature_names):
                with cols[i % 3]:
                    val = st.number_input(name, value=0.0, key=f"df_feat_{name}")
                    features.append(val)
        else:
            row_file = st.file_uploader("Upload a single-row CSV matching the feature columns", type=["csv"], key="df_row_csv")
            if row_file is not None:
                row_df = pd.read_csv(row_file)
                features = row_df.iloc[0][feature_names].tolist()

        if features is not None and st.button("Classify driver", key="df_predict_btn"):
            result = predict_driver_from_features(features, model_path=model_path)
            if result["unknown"]:
                st.warning(f"Unknown driver detected (confidence={result['confidence']:.2f})")
            else:
                st.success(f"Predicted driver: {result['prediction']} (confidence={result['confidence']:.2f})")
            st.bar_chart(pd.Series(result["probabilities"]))

# =====================================================================
# TAB 3 - Keyless Relay-Attack Detection
# =====================================================================
with tab3:
    st.subheader("Keyless Relay-Attack Detection")
    st.markdown(
        "Classifies **RF signal encodings** (key-fob / RKE replay captures rendered as "
        "images) with a ResNet18 classifier, distinguishing relay-attack signals from "
        "legitimate key-fob transmissions."
    )
    st.warning(
        "⚠️ Training pipeline in progress: `preprocess()` and `train()` for this module "
        "are not implemented yet. This tab only runs inference against a pre-trained "
        "checkpoint (`rf_model.pth`), which must be provided separately."
    )

    model_path = "rf_model.pth"

    if not os.path.exists(model_path):
        st.info(f"No checkpoint found at `{model_path}`. Place a trained checkpoint there to enable this tab.")
    else:
        img_file = st.file_uploader("Upload an RF-signal image", type=["jpg", "jpeg", "png", "bmp"], key="kl_upload")
        if img_file is not None:
            from keyless_hackwmail import predict as keyless_predict  # deferred: pulls in torch/torchvision

            tmp_path = "kl_uploaded_input.jpg"
            with open(tmp_path, "wb") as f:
                f.write(img_file.getbuffer())

            st.image(img_file, caption="Uploaded RF-signal image", width=300)

            with st.spinner("Running ResNet18 inference..."):
                result = keyless_predict(model_path, tmp_path)

            st.write(f"**Prediction: {result['prediction']}** (confidence={result['confidence']:.2f})")
            st.bar_chart(pd.Series(result["probabilities"]))

            if result["prediction"] == "attack" and result["confidence"] > 0.70:
                from alert_system import send_alert
                send_alert(
                    title="Keyless Car Attack Detected!",
                    messages=[f"Confidence: {result['confidence']:.2f}"],
                    level="high",
                    image_path=tmp_path,
                )
                st.error("🚨 Attack detected — alert dispatched via alert_system.")
            else:
                st.success("Legitimate access — no alert sent.")
