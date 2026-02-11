🚗 Anti-Theft Car System (AI-Powered Vehicle Security)
📌 Overview

The Anti-Theft Car System is an AI-powered, multi-layered vehicle security solution designed to combat modern car theft techniques such as keyless entry relay attacks and unauthorized vehicle usage. Unlike traditional alarm systems, this project provides proactive threat detection by combining computer vision, deep learning, behavioral analytics, and embedded hardware.

🎯 Problem Statement

Modern vehicle theft has evolved beyond simple break-ins. Attackers exploit vulnerabilities in keyless entry systems and traditional alarms fail to differentiate real threats from false positives. Existing solutions are reactive and lack intelligence.

💡 Solution

This project introduces a smart, intelligent security platform that:

Detects suspicious activity around the vehicle

Prevents keyless entry hacking attempts

Verifies authorized drivers using behavioral profiling

The system runs on an on-board computer (e.g., Raspberry Pi) and processes data from multiple sensors in real time.

🔑 Core Features
1️⃣ Suspicious Activity Detection (Computer Vision)

Uses YOLO-based object and pose detection

Identifies human presence and suspicious behaviors such as loitering or break-in gestures

Reduces false alarms by analyzing posture and movement patterns

2️⃣ Keyless Entry Hacking Defense

Captures RF signals using multiple antennas

Uses a deep learning classifier (ResNet-based) to detect relay attacks

Analyzes RSSI and timing differences to identify anomalous signals

3️⃣ Driver Behavioral Profiling

Builds a unique driving fingerprint using Random Forest

Uses accelerometer and CAN bus data

Detects unauthorized drivers and triggers alerts or engine immobilization

🛠️ Hardware & System Architecture

On-board Computer: Raspberry Pi

Sensors: Camera, RF modules, accelerometer, GPS timestamps

Optional Inputs: CAN bus, door sensors

Processing: Real-time AI inference and sensor fusion

🧪 Tech Stack

Programming: Python
AI / ML: YOLO, CNNs, ResNet, Random Forest
Computer Vision: OpenCV
RF Analysis: RSSI-based anomaly detection
Hardware: Raspberry Pi, RF modules, cameras, sensors

🚀 Applications

Smart vehicle security systems

Automotive safety research

IoT and AI-based embedded systems

🔮 Future Enhancements

Mobile app integration for real-time alerts

Cloud-based monitoring dashboard

Enhanced MLOps and model optimization

Integration with OEM vehicle systems

🏆 Highlights

Multi-layered AI-driven security

Low false positives

Scalable and real-world deployable

Hackathon-ready innovation
