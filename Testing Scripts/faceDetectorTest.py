from cvzone.FaceDetectionModule import FaceDetector
import cvzone
import cv2
cap = cv2.VideoCapture(0)
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)
while True:
    success, img = cap.read()
    if not success or img is None:
        print("⚠️ Failed to capture image from camera.")
        continue

    img, bboxs = detector.findFaces(img, draw=False)
    if bboxs:
        for bbox in bboxs:
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = int(bbox['score'][0] * 100)
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
            cvzone.putTextRect(img, f'{score}%', (x, y - 10))
            cvzone.cornerRect(img, (x, y, w, h))

    cv2.imshow("Image", img)
    cv2.waitKey(1)

