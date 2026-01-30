from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')  # or 'yolov8n.yaml' to train from scratch
    model.train(
        data='datasets/SplitData/data.yaml',
        epochs=300,
        imgsz=640,
        batch=16,
        project='runs',
        name='antispoof_yolov8n',
    )

if __name__ == '__main__':
    main()
