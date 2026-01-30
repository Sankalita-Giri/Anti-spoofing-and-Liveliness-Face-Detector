from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Load YOLOv8 model
model = YOLO("../models/yolov8s.pt")

# COCO Class Labels (Include your custom additions if needed)
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "calendar"
]

# Frame time tracking
prev_frame_time = 0

while True:
    success, img = cap.read()
    if not success:
        print("⚠️ Failed to grab frame")
        break

    curr_time = time.time()

    # YOLOv8 Inference
    try:
        results = model(img, stream=True)
    except Exception as e:
        print(f"❌ Model inference error: {e}")
        continue

    # Draw detections
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls] if cls < len(classNames) else "Unknown"
            cvzone.putTextRect(img, f'{class_name} {conf}', (x1, y1 - 10), scale=1, thickness=1)

    # FPS Calculation
    fps = 1 / (curr_time - prev_frame_time) if prev_frame_time else 0
    prev_frame_time = curr_time
    cvzone.putTextRect(img, f'FPS: {int(fps)}', (20, 30), scale=1, thickness=1)

    # Display
    cv2.imshow("YOLOv8 Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()