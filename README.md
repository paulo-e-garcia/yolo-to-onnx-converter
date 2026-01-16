# YOLO to ONNX Converter/Exporter

Simple Python script to export YOLOv8 models to ONNX format for deployment and inference optimization.

## Overview

This script converts a pretrained YOLOv8 model to ONNX format, making it compatible with various inference frameworks and enabling deployment across different platforms. The export uses FP32 precision with model simplification for optimal compatibility.

## Requirements

```python
pip install ultralytics torch
```
For GPU acceleration, ensure you have CUDA-compatible PyTorch installed.

## Usage

First of all you have to edit the script and set what model version you want it to download and convert. Do this by changing:

```python
model = YOLO('yolov8m.pt')
```

After that you have to change the export parameters to suit your needs. 

```python
model.export(
    format='onnx',
    imgsz=640,           # Change input image size
    device='cuda:0',     # Use 'cpu' for CPU export
    half=True,           # Enable FP16 for smaller model size
    simplify=True,       # Simplify ONNX graph
    opset=11             # ONNX opset version
)
```

After these changes, save the python file and run it. It will store the ONNX file in the same directory as the script.

```python
python yolo-to-onnx.py
```

## Notes

Dynamic input shapes are disabled for maximum compatibility. Enable `dynamic=True` if you need variable input dimensions during inference.



