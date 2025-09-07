# Lab 10: Deep Learning for Vision Report

## What We Did
We learned about deep learning for computer vision using Convolutional Neural Networks (CNNs). We used two popular frameworks:
- **PyTorch** with ResNet-18
- **TensorFlow/Keras** with MobileNetV2

## Deep Learning vs Traditional Methods

| Method | Feature Extraction | Accuracy | Training Required |
|--------|-------------------|----------|------------------|
| Traditional (Lab 8-9) | Manual features | Good | No |
| Deep Learning | Automatic | Excellent | Yes (or use pretrained) |

## What We Implemented

### 1. PyTorch with ResNet-18
```python
import torch
from torchvision import models
model = models.resnet18(pretrained=True)
output = model(img_tensor)
```
- Uses pretrained ResNet-18 model
- Automatic feature extraction
- Fast inference on single images

### 2. TensorFlow with MobileNetV2
```python
from tensorflow.keras.applications import MobileNetV2
model = MobileNetV2(weights='imagenet')
preds = model.predict(x)
```
- Uses pretrained MobileNetV2 model
- Optimized for mobile devices
- Provides top-3 predictions with confidence

### 3. OpenCV Integration
```python
img = cv2.imread('images/cat.jpeg')
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```
- Seamless integration with OpenCV
- Preprocessing for CNN input
- Real-world image handling

## Exercise Implementations

### Exercise 1: Batch Image Prediction ✅
- Processes all images in a folder automatically
- Shows predictions with confidence scores
- Handles multiple image formats (jpg, jpeg, png)

### Exercise 2: Model Comparison ✅
- Compares ResNet50 vs MobileNetV2 predictions
- Side-by-side visualization
- Shows different model strengths

### Exercise 3: Fine-Tuning Setup ✅
- Creates custom classification head
- Freezes pretrained layers
- Ready for custom dataset training

### Exercise 4: Feature Visualization ✅
- Visualizes CNN activation maps
- Shows different abstraction levels:
  - Early layers: Edges and patterns
  - Middle layers: Shapes and textures  
  - Later layers: Complex objects

## Key Advantages of Deep Learning

### Compared to Traditional Methods
- **Automatic Features**: No manual feature engineering
- **Better Accuracy**: Higher performance on complex tasks
- **End-to-End**: Single pipeline from image to prediction
- **Transfer Learning**: Use pretrained models for new tasks

### Framework Comparison
| Framework | Pros | Best For |
|-----------|------|----------|
| PyTorch | Research-friendly, dynamic | Experimentation, research |
| TensorFlow | Production-ready, mobile | Deployment, mobile apps |

## Practical Applications
- **Image Classification**: Categorize photos automatically
- **Medical Imaging**: Detect diseases in scans
- **Quality Control**: Identify defects in manufacturing
- **Content Moderation**: Filter inappropriate images
- **Mobile Apps**: Real-time image recognition

## What We Learned
1. **CNNs automatically learn features** from images
2. **Pretrained models save time** and work very well
3. **Transfer learning** lets us use powerful models on new tasks
4. **Different frameworks** have different strengths
5. **Feature visualization** helps understand what CNNs learn
6. **Deep learning is powerful** but needs more resources than traditional methods

## Tools Used
- **PyTorch**: Deep learning framework with ResNet
- **TensorFlow/Keras**: Production-ready framework with MobileNet
- **OpenCV**: Image preprocessing and loading
- **Matplotlib**: Visualization and plotting
- **PIL**: Image handling and processing

## Conclusion
Deep learning represents a major advance in computer vision. While traditional methods (Labs 8-9) are fast and interpretable, deep learning provides:

- **Higher accuracy** on complex vision tasks
- **Automatic feature learning** without manual engineering
- **Pretrained models** that work out-of-the-box
- **Transfer learning** for custom applications

Both approaches have their place:
- Use **traditional methods** for simple, fast applications
- Use **deep learning** for complex, high-accuracy requirements

Lab 10 successfully introduced the fundamentals of modern computer vision using deep learning!
