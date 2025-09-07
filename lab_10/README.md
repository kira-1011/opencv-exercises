# Lab 10: Deep Learning for Vision

## 🎯 Overview
This lab introduces deep learning for computer vision using PyTorch and TensorFlow. We explore Convolutional Neural Networks (CNNs) and pretrained models for image classification.

## 📁 Files
- `lab10_deep_learning.ipynb` - Main notebook with implementations
- `Lab10_Report.md` - Simple lab report with results
- `images/cat.jpeg` - Test image for classification
- `README.md` - This guide

## 🚀 Quick Start

### 1. Dependencies
All required packages are in `pyproject.toml`:
```bash
uv sync
```

### 2. Run the Notebook
```bash
uv run jupyter notebook lab10_deep_learning.ipynb
```

## 🧠 What You'll Learn

### Deep Learning Concepts
1. **CNNs** - Convolutional Neural Networks
2. **Pretrained Models** - ResNet, MobileNet
3. **Transfer Learning** - Using existing models
4. **Feature Visualization** - Understanding what CNNs learn

### Frameworks Covered
- **PyTorch** - Research-friendly, dynamic graphs
- **TensorFlow/Keras** - Production-ready, easy deployment

## 🏋️ Exercises Implemented
1. ✅ **Batch Prediction** - Process multiple images
2. ✅ **Model Comparison** - ResNet vs MobileNet
3. ✅ **Fine-Tuning Setup** - Custom classification head
4. ✅ **Feature Visualization** - CNN activation maps

## 🔍 Testing Your Images
Replace with your own images:
```python
# Single image
results = compare_models('your_image.jpg')

# Multiple images  
results = predict_image_dataset('your_folder/')
```

## 🎓 Learning Progression
- **Lab 8**: Machine Learning basics (KNN, SVM, Decision Trees)
- **Lab 9**: Classical Computer Vision (Haar, HOG+SVM)
- **Lab 10**: Modern Deep Learning (CNNs, Transfer Learning)

Each lab builds on the previous, showing the evolution of computer vision techniques! 🚀🧠📷
