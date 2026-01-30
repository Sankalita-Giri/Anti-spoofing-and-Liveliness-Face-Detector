from cvzone.FaceDetectionModule import FaceDetector
import cvzone
import cv2
import os
from time import time

# ==== Settings ====
classID = 0  # 0 is fake and 1 is real
outputFolderPath = 'datasets/DataCollect'
os.makedirs(outputFolderPath, exist_ok=True)
confidence = 0.8
save = True
blurThreshold = 35
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint = 64
debug = False

# ==== Camera Setup ====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camHeight)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

detector = FaceDetector()

# ==== Main Loop ====
while True:
    success, img = cap.read()
    if not success or img is None:
        print("⚠️ Failed to capture image from camera.")
        continue
    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = []
    listInfo = []

    if bboxs:
        print("✅ Face Detected: True")

        for bbox in bboxs:
            x, y, w, h = bbox['bbox']
            score = bbox['score'][0]
            print(f"🟡 Confidence Score: {round(score, 4)}")

            if score > confidence:
                # Apply offset
                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)

                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3.5)

            # Clamp values to stay in frame
            x = max(x, 0)
            y = max(y, 0)
            w = max(w, 0)
            h = max(h, 0)

            # ---- Blur Detection ----
            imgFace = img[y:y + h, x:x + w]
            if imgFace.size != 0:
                cv2.imshow('imgFace', imgFace)
                blurvalue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
            else:
                blurvalue = 0

            isClear = blurvalue > blurThreshold
            listBlur.append(isClear)

            # ---- Normalized Values ----
            ih, iw, _ = img.shape
            xc, yc = x + w / 2, y + h / 2
            xcn = round(xc / iw, floatingPoint)
            ycn = round(yc / ih, floatingPoint)
            wn = round(w / iw, floatingPoint)
            hn = round(h / ih, floatingPoint)

            xcn = min(xcn, 1)
            ycn = min(ycn, 1)
            wn = min(wn, 1)
            hn = min(hn, 1)

            print(f"🔹 Normalized Values - xcn: {xcn}, ycn: {ycn}, wn: {wn}, hn: {hn}")

            listInfo.append([xcn, ycn, wn, hn])

            listInfo.append(f'{classID} {xcn} {ycn} {wn} {hn}\n')


            # ---- Drawing ----
            cv2.rectangle(imgOut, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cvzone.putTextRect(
                imgOut,
                f'Score: {int(score * 100)}%  Blur: {blurvalue}',
                (x, y - 20),
                scale=2,
                thickness=3
            )
            if debug:
                cv2.rectangle(imgOut, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cvzone.putTextRect(
                   img,
                   f'Score: {int(score * 100)}%  Blur: {blurvalue}',
                   (x, y - 20),
                    scale=2,
                    thickness=3
                 )


        # ---- Summary Print ----
        if save:
            if all(listBlur)and len(listBlur) != []:
               timeNow = time()
               timeNow = str(timeNow).split('.')
               timeNow = timeNow[0]+timeNow[1]
               cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)

               #-----Save LabeL Text File----
               with open(f'{outputFolderPath}/{timeNow}.txt', 'a') as f:
                   for info in listInfo:
                       if isinstance(info, str):  # Only write string lines
                           f.write(info)

            print(f"📸 Blur Status: {listBlur}")
            print(f"📝 Normalized Info: {listInfo}")
        print("-" * 60)

    else:
        print("❌ Face Detected: False")

    # Show Full Camera Frame
    cv2.imshow("Image", imgOut)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==== Cleanup ====
cap.release()
cv2.destroyAllWindows()
