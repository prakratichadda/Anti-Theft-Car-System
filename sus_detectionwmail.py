import cv2
import time
import numpy as np
from ultralytics import YOLO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# ----------------------------
# Gmail Alert System
# ----------------------------
EMAIL_SENDER = "REDACTED_SENDER_EMAIL"       # <-- change this
EMAIL_PASS = "REDACTED_APP_PASSWORD"          # <-- your App Password
EMAIL_TO = "REDACTED_RECIPIENT_EMAIL"    # <-- change this

def send_snapshot_alert_from_frame(title, messages, frame, image_path="alert.jpg", level="high"):
    """Send an email with attached snapshot when suspicious activity is detected"""
    try:
        # Save the current frame as an image
        cv2.imwrite(image_path, frame)

        # Create the email
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_TO
        msg["Subject"] = f"[ALERT-{level.upper()}] {title}"

        # Body text
        body = "\n".join(messages)
        msg.attach(MIMEText(body, "plain"))

        # Attach snapshot
        with open(image_path, "rb") as f:
            img_data = f.read()
            
        image = MIMEImage(img_data, name=image_path)
        msg.attach(image)

        # Send via Gmail
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASS)
            server.sendmail(EMAIL_SENDER, EMAIL_TO, msg.as_string())

        print(f"📧 Alert email sent to {EMAIL_TO}")

    except Exception as e:
        print(f"❌ Failed to send email: {e}")


# ----------------------------
# Load Models
# ----------------------------
object_model = YOLO("yolov8n.pt")       # for person detection
pose_model = YOLO("yolov8n-pose.pt")    # for posture detection

# ----------------------------
# Config
# ----------------------------
LOITER_THRESHOLD = 10   # seconds near car = suspicious
PROXIMITY_THRESHOLD = 100  # pixels (distance between persons)
CAR_REGION = (200, 200, 400, 400)  # Simulated car bounding box (x1,y1,x2,y2)

person_tracker = {}  # {bbox: [time_entered, bbox]}


# ----------------------------
# Helper Functions
# ----------------------------
def calculate_center(bbox):
    """Convert YOLO bbox [x1,y1,x2,y2] to center point"""
    x1, y1, x2, y2 = bbox
    return ((x1+x2)/2, (y1+y2)/2)


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_angle(a, b, c):
    """Calculate angle (in degrees) at point b given three keypoints"""
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle


def detect_suspicious_pose(keypoints):
    """
    Suspicious behavior rules:
    - Hand near face (covering / smashing glass)
    - Leaning into car (intrusion posture)
    """
    nose = keypoints[0]
    left_wrist, right_wrist = keypoints[9], keypoints[10]
    left_shoulder, right_shoulder = keypoints[5], keypoints[6]
    left_hip, right_hip = keypoints[11], keypoints[12]

    suspicious_flags = []

    # Rule 1: Hand near face
    if euclidean_distance(left_wrist, nose) < 50 or euclidean_distance(right_wrist, nose) < 50:
        suspicious_flags.append("🚨 Hand near face → possible smash/covering")

    # Rule 2: Leaning posture
    torso_angle = calculate_angle(left_shoulder, left_hip, right_shoulder)
    if torso_angle < 100:  # small angle = bending forward
        suspicious_flags.append("🚨 Leaning posture → possible intrusion")

    return suspicious_flags


# ----------------------------
# Core Detection Logic
# ----------------------------
def analyze_frame(frame, realtime=True):
    alerts = []
    persons = []

    # Object Detection (persons)
    results = object_model(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                persons.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Pose Detection
    pose_results = pose_model(frame, verbose=False)
    for r in pose_results:
        if r.keypoints is not None:
            keypoints = r.keypoints.xy.cpu().numpy()
            for person in keypoints:
                warnings = detect_suspicious_pose(person)
                x, y = int(person[0][0]), int(person[0][1])
                for idx, warning in enumerate(warnings):
                    alerts.append(warning)
                    cv2.putText(frame, warning, (x, y - 10 - (idx*20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Loitering Detection (only realtime mode)
    current_time = time.time()
    if realtime:
        for bbox in persons:
            center = calculate_center(bbox)
            cx, cy = center
            x1, y1, x2, y2 = CAR_REGION
            inside_car_area = x1 <= cx <= x2 and y1 <= cy <= y2

            if inside_car_area:
                if bbox not in person_tracker:
                    person_tracker[bbox] = [current_time, bbox]
                else:
                    elapsed = current_time - person_tracker[bbox][0]
                    if elapsed > LOITER_THRESHOLD:
                        alerts.append("🚨 Loitering Detected!")
                        cv2.putText(frame, "🚨 Loitering Detected!", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                if bbox in person_tracker:
                    del person_tracker[bbox]

    # Proximity Detection
    for i in range(len(persons)):
        for j in range(i+1, len(persons)):
            c1 = calculate_center(persons[i])
            c2 = calculate_center(persons[j])
            if euclidean_distance(c1, c2) < PROXIMITY_THRESHOLD:
                alerts.append("🚨 Group Proximity Threat!")
                cv2.putText(frame, "🚨 Group Proximity Threat!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Draw car region
    cv2.rectangle(frame, (CAR_REGION[0], CAR_REGION[1]),
                  (CAR_REGION[2], CAR_REGION[3]), (255, 0, 0), 2)
    cv2.putText(frame, "Car Region", (CAR_REGION[0], CAR_REGION[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # ✅ Send alert email if suspicious activity detected
    if alerts:
        send_snapshot_alert_from_frame(
            title="Suspicious Behavior Detected!",
            messages=alerts,
            frame=frame,
            image_path="alert_snapshot.jpg",
            level="high"
        )

    return frame, alerts


# ----------------------------
# Modes: Webcam or Single Image
# ----------------------------
def run_webcam():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, alerts = analyze_frame(frame, realtime=True)

        cv2.imshow("Suspicious Behavior Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_on_image(image_path):
    frame = cv2.imread(image_path)
    frame, alerts = analyze_frame(frame, realtime=False)
    print("\n--- Analysis Report ---")
    if alerts:
        for a in alerts:
            print(a)
    else:
        print("✅ No suspicious activity detected")
    cv2.imshow("Image Analysis", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    choice = input("Select mode: (1) Webcam  (2) Image file → ")
    if choice == "1":
        run_webcam()
    else:
        path = input("Enter image path: ")
        run_on_image(path)
