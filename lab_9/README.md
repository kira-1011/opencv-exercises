# Lab 9: Object Detection with OpenCV

## 🎯 Overview
This lab implements Chapter 9's object detection techniques using OpenCV. We cover classical computer vision methods for detecting faces and pedestrians in images and video streams.

## 📁 Files
- `lab9_object_detection.ipynb` - Main notebook with all implementations
- `Lab9_Report.md` - Simple lab report with results
- `requirements.txt` - Required Python packages
- `images/` - Folder for test images and results

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
```bash
jupyter notebook lab9_object_detection.ipynb
```

## 🔍 What You'll Learn

### Detection Methods
1. **Haar Cascades** - Fast face detection
2. **HOG + SVM** - Robust pedestrian detection
3. **Real-time Processing** - Live webcam detection
4. **Video Analysis** - Frame-by-frame processing

### Exercises Implemented
1. ✅ Eye and smile detection
2. ✅ Video file processing  
3. ✅ HOG parameter tuning
4. ✅ Combined detection pipeline

## 📊 Key Concepts
- **Object Detection**: Finding and locating objects in images
- **Haar Features**: Fast pattern-based detection
- **HOG Features**: Gradient-based object description
- **Real-time Processing**: Live video analysis
- **Parameter Tuning**: Balancing speed vs accuracy

## 🎥 Testing Your Own Images/Videos
Replace the demo code with your own files:
```python
# For images
img = cv2.imread('your_image.jpg')
result, count = detect_faces(img)

# For videos  
results = process_video_file('your_video.mp4')

# For webcam (uncomment in notebook)
real_time_face_detection()
```

## 🏆 Results
- Successfully implemented all detection methods
- Created working real-time detection system
- Demonstrated parameter tuning effects
- Built combined detection pipeline

Perfect for learning classical computer vision before moving to deep learning approaches! 📷🤖
