import torch
from ultralytics import YOLO

# Load model
model = YOLO('yolov9e.pt')

# Export to ONNX format
model.export(
    format='onnx',
    imgsz=640,
    device='cuda:0',
    half=False,  # Use FP32
    int8=False,
    dynamic=False,
    simplify=True,
    opset=11,
    data=None,
    verbose=True
)
