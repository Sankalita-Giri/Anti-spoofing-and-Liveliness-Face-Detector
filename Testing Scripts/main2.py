import cv2
import cvzone
import time
from ultralytics import YOLO

# === Webcam Setup ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()

cap.set(3, 1280)
cap.set(4, 720)

# === Load YOLOv8 Model ===
model = YOLO("../models/best_300.pt")
classNames = ["fake", "real"]  # confirm order with dataset.yaml

prev_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    if not success:
        print("⚠️ Failed to grab frame.")
        break

    results = model(img, stream=True, verbose=False)

    best_conf = 0
    best_box = None
    best_cls = None

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > best_conf and conf > 0.2:
                best_conf = conf
                best_box = box
                best_cls = int(box.cls[0])

    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        class_name = classNames[best_cls]

        color = (0, 255, 0) if class_name == "real" else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)

        cvzone.putTextRect(
            img,
            f'{class_name.upper()} ({round(best_conf, 2)})',
            (x1, max(35, y1 - 10)),
            scale=1.2,
            thickness=2,
            colorB=color
        )
    else:
        cvzone.putTextRect(img, "No face detected", (20, 70), scale=1)

    # === FPS ===
    fps = 1 / (new_frame_time - prev_frame_time + 1e-6)
    prev_frame_time = new_frame_time
    cvzone.putTextRect(img, f'FPS: {int(fps)}', (20, 30), scale=1)

    cv2.imshow("Real vs Fake Face Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
