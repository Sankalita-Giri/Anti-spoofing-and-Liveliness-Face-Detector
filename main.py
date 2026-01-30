import cv2
import cvzone
import math
import time
from ultralytics import YOLO

# === Webcam Setup ===
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # width
cap.set(4, 720)   # height

# === Load Trained YOLOv8 Model ===
model = YOLO("../models/best_300.pt")  # path to your model
classNames = ["fake", "real"]

# === Frame Timing ===
prev_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        print("⚠️ Failed to grab frame.")
        break

    # === Inference ===
    results = model(img, stream=True, verbose=False)

    detected = False  # flag to avoid flickering

    for r in results:
        for box in r.boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Confidence score
            conf = float(box.conf[0])

            # Filter out weak detections
            if conf < 0.2:
                continue

            cls = int(box.cls[0])
            class_name = classNames[cls]

            # Draw rectangle with color based on class
            color = (0, 255, 0) if class_name == "real" else (0, 0, 255)  # green for real, red for fake
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=4)
            cvzone.putTextRect(img, f'{class_name.upper()} ({round(conf, 2)})', (x1, y1 - 15),
                               scale=1.2, thickness=2, colorB=color, colorT=(255, 255, 255))

            detected = True

    # === FPS Calculation ===
    fps = 1 / (new_frame_time - prev_frame_time + 1e-6)
    prev_frame_time = new_frame_time
    cvzone.putTextRect(img, f'FPS: {int(fps)}', (20, 30), scale=1, thickness=1)

    # === No Detection Message ===
    if not detected:
        cvzone.putTextRect(img, "No face detected", (20, 70), scale=1, thickness=1, colorB=(0, 0, 0))

    # === Show Output ===
    cv2.imshow("Real vs Fake Face Detection", img)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Release resources ===
cap.release()
cv2.destroyAllWindows()
