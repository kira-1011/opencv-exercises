# 🔢 Digit Recognition GUI App

A complete GUI-based digit recognition application implementing Exercise 4 from Lab 8: Machine Learning with OpenCV + Scikit-learn.

## 🎯 Features

### ✨ Interactive Interface
- **Drawing Canvas**: Draw digits (0-9) with your mouse
- **Model Selection**: Choose between KNN, SVM, and Decision Tree
- **Real-time Predictions**: Get instant results with confidence scores
- **Image Loading**: Load digit images from files
- **Clear Canvas**: Start over with a clean slate

### 🤖 Machine Learning Models
- **K-Nearest Neighbors (KNN)**: Simple and effective
- **Support Vector Machine (SVM)**: Excellent generalization
- **Decision Tree**: Interpretable predictions
- **Model Accuracy Display**: See each model's performance

### 🔧 Technical Features
- OpenCV image preprocessing
- PIL/Pillow for image handling
- tkinter professional GUI
- Real-time model switching
- Confidence score display

## 🚀 Quick Start

### 1. Installation
```bash
# Install required packages
uv sync 

# Or install individually
uv add opencv-python scikit-learn pillow matplotlib numpy
```

### 2. Run the Application
```bash
# Simple launcher
uv run run_gui.py

# Or run directly
uv run digit_recognition_gui.py
```

## 🎨 How to Use

### Drawing Mode
1. **Draw**: Use your mouse to draw a digit (0-9) on the black canvas
2. **Select Model**: Choose KNN, SVM, or Decision Tree
3. **Predict**: Click "Predict Digit" to get results
4. **Clear**: Use "Clear Canvas" to start over

### Image Loading Mode
1. **Load Image**: Click "Load Image" to select a digit image file
2. **Supported Formats**: PNG, JPG, JPEG, BMP, TIFF
3. **Automatic Processing**: Image is automatically resized and processed
4. **Predict**: Click "Predict Digit" to analyze the loaded image

## 📊 Model Performance

The GUI displays real-time accuracy for each model:
- **KNN**: Typically ~99% accuracy on digits dataset
- **SVM**: Usually ~98% accuracy with excellent generalization
- **Decision Tree**: Around ~87% accuracy, highly interpretable

## 🔬 Technical Implementation

### Image Preprocessing Pipeline
```python
# 1. Canvas/Image → PIL Image
# 2. Convert to grayscale
# 3. Resize to 8x8 pixels (digits dataset format)
# 4. Normalize to 0-16 range
# 5. Flatten for model input
```

### Model Integration
- All models trained on scikit-learn digits dataset
- Real-time model switching without retraining
- Confidence scores where available (SVM probability, KNN distances)
- Professional error handling and user feedback

### GUI Architecture
- **Main Window**: 800x600 pixel professional interface
- **Left Panel**: Drawing canvas and controls
- **Right Panel**: Model selection and results
- **Status Bar**: Real-time feedback and instructions

## 🎯 Educational Value

### Learning Objectives Met
1. ✅ **Practical ML Application**: Real-world implementation
2. ✅ **OpenCV Integration**: Image preprocessing pipeline
3. ✅ **Model Comparison**: Interactive model switching
4. ✅ **GUI Development**: Professional interface design
5. ✅ **User Experience**: Intuitive interaction design

### Skills Demonstrated
- Machine learning model deployment
- Computer vision preprocessing
- GUI application development
- Real-time prediction systems
- Professional software design

## 🛠️ Customization Options

### Modify Drawing Settings
```python
# In digit_recognition_gui.py
self.canvas_size = 280      # Canvas size in pixels
self.brush_size = 15        # Drawing brush size
```

### Add New Models
```python
# In train_models() method
new_model = YourModelClass()
new_model.fit(X_train, y_train)
self.models['Your Model'] = {
    'model': new_model, 
    'accuracy': accuracy_score(y_test, new_model.predict(X_test))
}
```

### Customize GUI Appearance
```python
# Colors and styling
bg_color = '#f0f0f0'        # Background color
accent_color = '#27ae60'    # Accent color
font_family = 'Arial'       # Font family
```

## 🔍 Troubleshooting

### Common Issues

**"Module not found" errors:**
```bash
pip install -r requirements.txt
```

**Canvas not drawing:**
- Ensure mouse events are properly bound
- Check PIL image initialization

**Prediction errors:**
- Verify model training completed
- Check image preprocessing pipeline

**GUI not appearing:**
- Ensure tkinter is installed (usually built-in)
- Check display settings on remote systems

## 📈 Performance Tips

1. **Drawing Quality**: Draw digits clearly and large
2. **Model Selection**: Try different models for comparison
3. **Image Loading**: Use high-contrast digit images
4. **Canvas Size**: Optimal size is already set for best results

## 🚀 Future Enhancements

- [ ] Save/Load trained models
- [ ] Batch image processing
- [ ] Webcam integration
- [ ] Deep learning models (CNN)
- [ ] Data augmentation options
- [ ] Model training interface
- [ ] Export prediction results

## 📝 File Structure

```
lab_8/
├── digit_recognition_gui.py    # Main GUI application
├── run_gui.py                 # Simple launcher
├── requirements.txt           # Package dependencies
├── GUI_README.md             # This documentation
└── lab8_machine_learning.ipynb # Complete lab notebook
```

## 🎓 Academic Context

This GUI application serves as the practical implementation of Exercise 4 from Lab 8: Machine Learning. It demonstrates the real-world application of the concepts learned in the lab, bridging the gap between theoretical understanding and practical deployment.

**Lab 8 Connection:**
- Uses models trained in the notebook
- Implements OpenCV preprocessing techniques
- Demonstrates all three ML algorithms
- Provides interactive learning experience

---