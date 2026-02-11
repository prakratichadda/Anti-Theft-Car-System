#!/usr/bin/env python3
"""
alert_system.py

Standalone alert system that can send:
 - Email alerts (required: ALERT_EMAIL, ALERT_EMAIL_PASS, ALERT_TO)
 - SMS alerts via Twilio (optional: TWILIO_* vars)
 - Telegram alerts (optional: TELEGRAM_* vars)

Usage example:
    from alert_system import send_alert

    send_alert(
        title="Suspicious Activity",
        messages=["Loitering near car", "Group proximity threat"],
        level="high",
        image_path="snapshot.jpg"   # optional
    )
"""

import os
import time
import threading
from pathlib import Path
from typing import Optional, List

# --- EMAIL ---
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# --- Optional extras ---
try:
    from twilio.rest import Client as TwilioClient
    _HAS_TWILIO = True
except Exception:
    _HAS_TWILIO = False

try:
    import requests
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False


# --- CONFIG ---
ALERT_CONFIG = {
    "email_sender": os.environ.get("ALERT_EMAIL"),
    "email_pass": os.environ.get("ALERT_EMAIL_PASS"),
    "email_to": os.environ.get("ALERT_TO"),
    "smtp_server": os.environ.get("ALERT_SMTP", "smtp.gmail.com"),
    "smtp_port": int(os.environ.get("ALERT_SMTP_PORT", 465)),

    "twilio_sid": os.environ.get("TWILIO_SID"),
    "twilio_token": os.environ.get("TWILIO_TOKEN"),
    "twilio_from": os.environ.get("TWILIO_FROM"),
    "twilio_to": os.environ.get("TWILIO_TO"),

    "telegram_bot_token": os.environ.get("TELEGRAM_BOT_TOKEN"),
    "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID"),

    "cooldown": {"email": 60, "sms": 60, "telegram": 60},  # seconds
}

# Channel availability
_email_enabled = bool(ALERT_CONFIG["email_sender"] and ALERT_CONFIG["email_pass"] and ALERT_CONFIG["email_to"])
_sms_enabled = bool(ALERT_CONFIG["twilio_sid"] and ALERT_CONFIG["twilio_token"] and ALERT_CONFIG["twilio_from"] and ALERT_CONFIG["twilio_to"] and _HAS_TWILIO)
_telegram_enabled = bool(ALERT_CONFIG["telegram_bot_token"] and ALERT_CONFIG["telegram_chat_id"] and _HAS_REQUESTS)

_twilio_client = None
if _sms_enabled:
    _twilio_client = TwilioClient(ALERT_CONFIG["twilio_sid"], ALERT_CONFIG["twilio_token"])

_last_sent = {"email": 0, "sms": 0, "telegram": 0}
_lock_alert = threading.Lock()


# --- INTERNAL HELPERS ---
def _can_send(channel: str) -> bool:
    with _lock_alert:
        now = time.time()
        last = _last_sent.get(channel, 0)
        cd = ALERT_CONFIG["cooldown"].get(channel, 60)
        return (now - last) >= cd

def _mark_sent(channel: str):
    with _lock_alert:
        _last_sent[channel] = time.time()


# --- EMAIL ---
def send_email_alert(subject: str, body: str, image_path: Optional[str] = None) -> bool:
    if not _email_enabled:
        print("[EMAIL] Disabled or missing credentials.")
        return False
    if not _can_send("email"):
        print("[EMAIL] Cooldown active, skipping.")
        return False
    try:
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = ALERT_CONFIG["email_sender"]
        msg["To"] = ALERT_CONFIG["email_to"]
        msg.attach(MIMEText(body, "plain"))

        if image_path and Path(image_path).exists():
            with open(image_path, "rb") as f:
                img_data = f.read()
            img = MIMEImage(img_data)
            img.add_header("Content-Disposition", "attachment", filename=Path(image_path).name)
            msg.attach(img)

        with smtplib.SMTP_SSL(ALERT_CONFIG["smtp_server"], ALERT_CONFIG["smtp_port"]) as server:
            server.login(ALERT_CONFIG["email_sender"], ALERT_CONFIG["email_pass"])
            server.send_message(msg)

        _mark_sent("email")
        print("[EMAIL] Sent successfully.")
        return True
    except Exception as e:
        print("[EMAIL] Error:", e)
        return False


# --- SMS ---
def send_sms_alert(body: str) -> bool:
    if not _sms_enabled:
        print("[SMS] Disabled or missing Twilio config.")
        return False
    if not _can_send("sms"):
        print("[SMS] Cooldown active, skipping.")
        return False
    try:
        _twilio_client.messages.create(
            body=body,
            from_=ALERT_CONFIG["twilio_from"],
            to=ALERT_CONFIG["twilio_to"]
        )
        _mark_sent("sms")
        print("[SMS] Sent successfully.")
        return True
    except Exception as e:
        print("[SMS] Error:", e)
        return False


# --- TELEGRAM ---
def send_telegram_alert(text: str, image_path: Optional[str] = None) -> bool:
    if not _telegram_enabled:
        print("[TELEGRAM] Disabled or missing config.")
        return False
    if not _can_send("telegram"):
        print("[TELEGRAM] Cooldown active, skipping.")
        return False
    try:
        token = ALERT_CONFIG["telegram_bot_token"]
        chat_id = ALERT_CONFIG["telegram_chat_id"]
        base_url = f"https://api.telegram.org/bot{token}"

        if image_path and Path(image_path).exists():
            url = f"{base_url}/sendPhoto"
            with open(image_path, "rb") as f:
                files = {"photo": f}
                data = {"chat_id": chat_id, "caption": text}
                resp = requests.post(url, data=data, files=files, timeout=15)
        else:
            url = f"{base_url}/sendMessage"
            data = {"chat_id": chat_id, "text": text}
            resp = requests.post(url, data=data, timeout=10)

        resp.raise_for_status()
        _mark_sent("telegram")
        print("[TELEGRAM] Sent successfully.")
        return True
    except Exception as e:
        print("[TELEGRAM] Error:", e)
        return False


# --- MASTER FUNCTION ---
def send_alert(
    title: str,
    messages: List[str],
    level: str = "medium",
    image_path: Optional[str] = None,
    channels: Optional[List[str]] = None,
    async_send: bool = True
):
    """Send alert across chosen channels"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    body = f"{title}\nLevel: {level}\nTime: {timestamp}\n\n" + "\n".join(messages)

    if channels is None:
        channels = []
        if _email_enabled: channels.append("email")
        if _sms_enabled: channels.append("sms")
        if _telegram_enabled: channels.append("telegram")

    def _worker():
        if "email" in channels:
            send_email_alert(f"[ALERT][{level.upper()}] {title}", body, image_path=image_path)
        if "sms" in channels:
            send_sms_alert(f"{title} | {level} | {messages[0] if messages else ''}")
        if "telegram" in channels:
            send_telegram_alert(body, image_path=image_path)

    if async_send:
        threading.Thread(target=_worker, daemon=True).start()
    else:
        _worker()


# --- DEMO ---
if __name__ == "__main__":
    print("Testing alert system...")

    send_alert(
        title="Test Alert",
        messages=["This is a test", "Everything is working"],
        level="low",
        image_path=None
    )
