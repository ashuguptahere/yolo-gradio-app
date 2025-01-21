import os
from ultralytics import YOLO


# Function to train YOLO model
def train_yolo(
    model_choice="yolo11s",
    data_path=None,
    epochs=100,
    time=None,
    patience=100,
    batch=16,
    imgsz=640,
    save=True,
    save_period=-1,
    cache=False,
    device=None,
    workers=8
):
    if not os.path.exists(data_path):
        return "Error: The specified data file path does not exist."

    # Initialize YOLO model
    model = YOLO(model_choice)

    # Train the model
    model.train(data=data_path, epochs=epochs, batch=batch, imgsz=imgsz)

    return f"Model trained successfully for {epochs} epochs with {model_choice}."


def val_yolo(
    model_choice="yolo11s",
    imgsz=640,
    batch=16,
    save_json=False,
    save_hybrid=False,
    conf=0.001,
    iou=0.6,
    max_det=300,
    half=True,
    device=None,
    dnn=False,
    plots=False,
    rect=False,
    split="val",
):
    # Initialize YOLO model
    model = YOLO(model_choice)
    metrics = model.val()

    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category


def predict_yolo():
    pass


def export_yolo():
    pass


def track_yolo():
    pass


def benchmark_yolo():
    pass
