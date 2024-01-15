from ultralytics import YOLO

model = YOLO('yolov8x.pt')

results = model.train(
    data='../data/recycle.yaml',
    epochs=500,
    imgsz=1024,
    batch=16,
    name='trash_yolo',
    optimizer="AdamW",
    save_period=1,
    seed=42,
    workers=16,
    amp=True,
    lr0=1e-3,
    exist_ok=True,
    cls=1,
)
