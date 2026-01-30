# 🛡️ Anti-Spoofing and Liveliness Face Detector

A real-time **Anti-Spoofing & Liveliness Face Detection system** built using **OpenCV, Dlib, NumPy, and YOLOv8 (Ultralytics)**. This project aims to distinguish **real human faces** from **spoof attacks** such as photos, videos, or faces displayed on mobile screens.

---

## 📌 Project Overview

Face recognition systems are vulnerable to spoofing attacks using printed photos, mobile screens, or videos. This project addresses that problem by combining:

* **YOLOv8** for face detection
* **Liveliness checks** using webcam-based cues
* **Anti-spoofing logic** to classify faces as **REAL** or **FAKE**

The model works in real-time and is suitable for security-focused applications.

---

## 🚀 Features

* 🎯 Real-time face detection using **YOLOv8n**
* 🧠 Anti-spoofing detection (real vs fake face)
* 📷 Webcam-based live face analysis
* 🖼️ Detects spoof attempts via:

  * Printed photos
  * Mobile screens
  * Video replays
* 🔊 Optional alert system (can be extended)
* 💻 Runs smoothly on CPU

---

## 🧰 Tech Stack

* **Python**
* **OpenCV (cv2)** – Image processing & webcam handling
* **Dlib** – Face landmarks & detection utilities
* **NumPy** – Numerical operations
* **Ultralytics YOLOv8** – Face detection model
* **YOLOv8n** – Lightweight and fast

---

## 📂 Project Structure

```
Anti-Spoofing-and-Liveliness-Face-Detector/
│
├── models/
│   └── yolov8n.pt
│
├── dataset/
│   ├── real/
│   └── fake/
│
├── utils/
│   ├── liveliness.py
│   └── spoof_detector.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/Anti-Spoofing-and-Liveliness-Face-Detector.git
cd Anti-Spoofing-and-Liveliness-Face-Detector
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

> Make sure Python 3.8+ is installed.

---

## ▶️ How to Run

```bash
python main.py
```

* The webcam will open automatically
* The system will detect faces and classify them as:

  * 🟢 **REAL FACE**
  * 🔴 **FAKE FACE**

---

## 🧠 How It Works

1. **YOLOv8** detects the face region
2. Face features are extracted using **OpenCV & Dlib**
3. Anti-spoofing logic checks for:

   * Image sharpness / blur
   * Screen artifacts
   * Live facial cues
4. The face is classified as **REAL** or **FAKE**

---

## 📊 Dataset

* Custom dataset with labeled **real** and **fake** faces
* Fake samples include:

  * Photos on mobile screens
  * Printed images

> You can improve accuracy by expanding the dataset.

---

## 🛠️ Future Improvements

* 🔍 Eye-blink detection
* 🎥 Head movement tracking
* 🔊 Sound-based liveliness checks
* 📱 Mobile deployment
* 🤖 Deep learning-based spoof classifier

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Open a Pull Request

---

## 👩‍💻 Author

**Sankalita Giri**
Final Year CSE Student
Passionate about Computer Vision, ML & AI 🚀

---

⭐ If you like this project, don’t forget to **star the repo**!
