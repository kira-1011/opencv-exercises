# Lab 11: Real-Time Object Detection

## 🎯 Overview
This lab implements Chapter 11's real-time object detection using YOLOv8, SSD, and MobileNet-SSD with OpenCV's DNN module. We explore state-of-the-art detection models optimized for speed and accuracy.

## 📁 Files
- `lab11_realtime_object_detection.ipynb` - Main notebook with implementations
- `Lab11_Report.md` - Simple lab report with results
- `images/` - Folder for test images and detection results

## 🚀 Quick Start

### 1. Dependencies
All packages managed by `uv` in `pyproject.toml`:
```bash
uv sync
uv add ultralytics  # For YOLOv8
```

### 2. Run the Notebook
```bash
uv run jupyter notebook lab11_realtime_object_detection.ipynb
```

## 🔍 What You'll Learn

### Detection Models
1. **YOLOv8** - Latest YOLO version, ultra-fast
2. **Model Variants** - Nano, Small, Medium sizes
3. **FPS Benchmarking** - Performance measurement
4. **OpenCV DNN** - Deployment approach

### Key Concepts
- **Real-time processing** - Low latency detection
- **Multi-object detection** - Find multiple objects per image
- **Speed vs accuracy** - Model size trade-offs
- **Deployment options** - Ultralytics vs OpenCV DNN

## 🏋️ Exercises Implemented
1. ✅ **YOLO Version Comparison** - Test different model sizes
2. ✅ **FPS Benchmarking** - Measure real-time performance
3. ✅ **Custom Training Setup** - How to train on your data
4. ✅ **OpenCV DNN Deployment** - Production-ready approach

## 🎥 Real-Time Capabilities
- **Live webcam detection** with bounding boxes
- **Multiple object classes** (80 COCO categories)
- **Confidence scores** for each detection
- **Hardware optimization** for your device

## 📊 Performance Comparison

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| YOLOv8n | ~30-60 FPS | Good | Mobile apps |
| YOLOv8s | ~20-40 FPS | Better | General use |
| YOLOv8m | ~10-25 FPS | Best | High accuracy |

## 🎓 Learning Progression
- **Lab 8**: Basic ML (KNN, SVM, Decision Trees)
- **Lab 9**: Classical Vision (Haar, HOG+SVM) 
- **Lab 10**: Deep Learning (CNNs, Transfer Learning)
- **Lab 11**: Real-Time Detection (YOLO, Multi-object)

Each lab builds toward modern computer vision capabilities! 🚀🔍📷
