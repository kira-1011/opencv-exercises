#!/usr/bin/env python3
"""
Simple GUI-based Digit Recognition App
Lab 8: Machine Learning with OpenCV + Scikit-learn

This app allows users to:
1. Draw digits on a canvas
2. Load digit images from files
3. Get real-time predictions using trained ML models
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import pickle
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class DigitRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognition App - Lab 8")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.canvas_size = 280
        self.brush_size = 15
        self.models = {}
        self.current_model = None
        
        # Create drawing canvas
        self.drawing_canvas = None
        self.drawing_image = None
        self.drawing_draw = None
        
        # Setup GUI
        self.setup_gui()
        
        # Train models
        self.train_models()
        
    def setup_gui(self):
        """Setup the GUI components"""
        # Title
        title_label = tk.Label(
            self.root, 
            text="🔢 Digit Recognition App", 
            font=("Arial", 20, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=10)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=20, pady=10)
        
        # Left frame for drawing
        left_frame = tk.Frame(main_frame, bg='#f0f0f0')
        left_frame.pack(side='left', fill='both', expand=True)
        
        # Canvas frame
        canvas_frame = tk.LabelFrame(
            left_frame, 
            text="Draw a Digit (0-9)", 
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        canvas_frame.pack(pady=10)
        
        # Drawing canvas
        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='black',
            cursor='pencil'
        )
        self.canvas.pack(padx=10, pady=10)
        
        # Initialize drawing
        self.setup_drawing()
        
        # Canvas controls
        controls_frame = tk.Frame(left_frame, bg='#f0f0f0')
        controls_frame.pack(pady=10)
        
        clear_btn = tk.Button(
            controls_frame,
            text="🗑️ Clear Canvas",
            command=self.clear_canvas,
            font=("Arial", 10, "bold"),
            bg='#e74c3c',
            fg='white',
            relief='raised',
            padx=20
        )
        clear_btn.pack(side='left', padx=5)
        
        load_btn = tk.Button(
            controls_frame,
            text="📁 Load Image",
            command=self.load_image,
            font=("Arial", 10, "bold"),
            bg='#3498db',
            fg='white',
            relief='raised',
            padx=20
        )
        load_btn.pack(side='left', padx=5)
        
        # Right frame for results
        right_frame = tk.Frame(main_frame, bg='#f0f0f0')
        right_frame.pack(side='right', fill='both', padx=(20, 0))
        
        # Model selection
        model_frame = tk.LabelFrame(
            right_frame,
            text="Select Model",
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        model_frame.pack(fill='x', pady=(0, 10))
        
        self.model_var = tk.StringVar(value="KNN")
        models = ["KNN", "SVM", "Decision Tree"]
        
        for model in models:
            rb = tk.Radiobutton(
                model_frame,
                text=model,
                variable=self.model_var,
                value=model,
                font=("Arial", 10),
                bg='#f0f0f0',
                command=self.update_model
            )
            rb.pack(anchor='w', padx=10, pady=2)
        
        # Prediction results
        results_frame = tk.LabelFrame(
            right_frame,
            text="Prediction Results",
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        results_frame.pack(fill='both', expand=True, pady=10)
        
        # Prediction display
        self.prediction_label = tk.Label(
            results_frame,
            text="Draw a digit to predict",
            font=("Arial", 24, "bold"),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        self.prediction_label.pack(pady=20)
        
        # Confidence display
        self.confidence_label = tk.Label(
            results_frame,
            text="Confidence: --",
            font=("Arial", 12),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        self.confidence_label.pack(pady=5)
        
        # Model accuracy display
        self.accuracy_label = tk.Label(
            results_frame,
            text="Model Accuracy: --",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        self.accuracy_label.pack(pady=5)
        
        # Predict button
        predict_btn = tk.Button(
            results_frame,
            text="🎯 Predict Digit",
            command=self.predict_digit,
            font=("Arial", 12, "bold"),
            bg='#27ae60',
            fg='white',
            relief='raised',
            padx=30,
            pady=10
        )
        predict_btn.pack(pady=20)
        
        # Status bar
        self.status_label = tk.Label(
            self.root,
            text="Ready - Draw a digit and click Predict!",
            font=("Arial", 10),
            bg='#34495e',
            fg='white',
            relief='sunken'
        )
        self.status_label.pack(side='bottom', fill='x')
        
    def setup_drawing(self):
        """Setup drawing functionality"""
        self.drawing_image = Image.new('RGB', (self.canvas_size, self.canvas_size), 'black')
        self.drawing_draw = ImageDraw.Draw(self.drawing_image)
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        
        self.drawing = False
        
    def start_drawing(self, event):
        """Start drawing on canvas"""
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y
        
    def draw(self, event):
        """Draw on canvas"""
        if self.drawing:
            # Draw on tkinter canvas
            self.canvas.create_oval(
                event.x - self.brush_size//2,
                event.y - self.brush_size//2,
                event.x + self.brush_size//2,
                event.y + self.brush_size//2,
                fill='white',
                outline='white'
            )
            
            # Draw on PIL image
            self.drawing_draw.ellipse([
                event.x - self.brush_size//2,
                event.y - self.brush_size//2,
                event.x + self.brush_size//2,
                event.y + self.brush_size//2
            ], fill='white')
            
            self.last_x, self.last_y = event.x, event.y
            
    def stop_drawing(self, event):
        """Stop drawing"""
        self.drawing = False
        
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.delete("all")
        self.drawing_image = Image.new('RGB', (self.canvas_size, self.canvas_size), 'black')
        self.drawing_draw = ImageDraw.Draw(self.drawing_image)
        self.prediction_label.config(text="Draw a digit to predict", fg='#7f8c8d')
        self.confidence_label.config(text="Confidence: --")
        self.status_label.config(text="Canvas cleared - Draw a new digit!")
        
    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Digit Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load and process image
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError("Could not load image")
                
                # Resize to canvas size
                img_resized = cv2.resize(img, (self.canvas_size, self.canvas_size))
                
                # Convert to PIL for display
                img_pil = Image.fromarray(img_resized).convert('RGB')
                
                # Update canvas
                self.clear_canvas()
                self.drawing_image = img_pil
                self.drawing_draw = ImageDraw.Draw(self.drawing_image)
                
                # Display on canvas
                photo = ImageTk.PhotoImage(img_pil)
                self.canvas.create_image(0, 0, anchor='nw', image=photo)
                self.canvas.image = photo  # Keep reference
                
                self.status_label.config(text=f"Image loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {str(e)}")
                
    def train_models(self):
        """Train the machine learning models"""
        self.status_label.config(text="Training models... Please wait.")
        self.root.update()
        
        try:
            # Load digits dataset
            data = load_digits()
            X = data.images.reshape((len(data.images), -1))
            y = data.target
            
            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )
            
            # Train KNN
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train, y_train)
            knn_acc = accuracy_score(y_test, knn.predict(X_test))
            
            # Train SVM
            svm = SVC(kernel='linear', C=1, probability=True)
            svm.fit(X_train, y_train)
            svm_acc = accuracy_score(y_test, svm.predict(X_test))
            
            # Train Decision Tree
            dt = DecisionTreeClassifier(max_depth=10, random_state=42)
            dt.fit(X_train, y_train)
            dt_acc = accuracy_score(y_test, dt.predict(X_test))
            
            # Store models and accuracies
            self.models = {
                'KNN': {'model': knn, 'accuracy': knn_acc},
                'SVM': {'model': svm, 'accuracy': svm_acc},
                'Decision Tree': {'model': dt, 'accuracy': dt_acc}
            }
            
            # Set default model
            self.update_model()
            
            self.status_label.config(text="Models trained successfully! Ready to predict.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not train models: {str(e)}")
            self.status_label.config(text="Error training models.")
            
    def update_model(self):
        """Update the current model selection"""
        model_name = self.model_var.get()
        if model_name in self.models:
            self.current_model = self.models[model_name]
            accuracy = self.current_model['accuracy']
            self.accuracy_label.config(text=f"Model Accuracy: {accuracy:.4f}")
            self.status_label.config(text=f"Selected model: {model_name}")
            
    def preprocess_image(self, img_pil):
        """Preprocess the drawn image for prediction"""
        # Convert PIL to numpy array
        img_array = np.array(img_pil.convert('L'))
        
        # Resize to 8x8 (digits dataset format)
        img_resized = cv2.resize(img_array, (8, 8))
        
        # Normalize to 0-16 range (like digits dataset)
        img_normalized = img_resized.astype(np.float32) / 255.0 * 16.0
        
        # Flatten for model input
        img_flat = img_normalized.reshape(1, -1)
        
        return img_flat
        
    def predict_digit(self):
        """Predict the drawn digit"""
        if self.current_model is None:
            messagebox.showwarning("Warning", "No model selected!")
            return
            
        try:
            # Preprocess the drawn image
            processed_img = self.preprocess_image(self.drawing_image)
            
            # Make prediction
            model = self.current_model['model']
            prediction = model.predict(processed_img)[0]
            
            # Get confidence (if available)
            confidence = "N/A"
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(processed_img)[0]
                confidence = f"{max(probabilities):.2%}"
            elif hasattr(model, 'decision_function'):
                # For SVM
                decision_scores = model.decision_function(processed_img)[0]
                confidence = f"{max(decision_scores):.2f}"
            
            # Update display
            self.prediction_label.config(
                text=f"Predicted: {prediction}",
                fg='#27ae60',
                font=("Arial", 28, "bold")
            )
            self.confidence_label.config(text=f"Confidence: {confidence}")
            
            model_name = self.model_var.get()
            self.status_label.config(text=f"Prediction complete using {model_name}: {prediction}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.status_label.config(text="Prediction failed.")


def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = DigitRecognitionGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()
