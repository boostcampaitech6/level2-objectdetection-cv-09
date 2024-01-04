from ultralytics import YOLO

model = YOLO('yolov8x.pt')

results = model.train(data='../data/recycle.yaml', epochs=30, imgsz=1024)
