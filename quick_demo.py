#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cassava Leaf Disease Classification - Quick Inference Demo
=========================================================
A simple console-based demo for testing the model on sample images.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import json
import random
from PIL import Image

# Configuration
MODEL_PATH = "./model_384_4.h5"
WEIGHTS_PATH = "./EffNetB0_384_4_best.weights.h5"
SAMPLE_DIR = "./test15"
IMAGE_SIZE = 384

def load_class_mapping():
    """Load class mapping from JSON file"""
    default_mapping = {
        "0": "Cassava Bacterial Blight (CBB)",
        "1": "Cassava Brown Streak Disease (CBSD)",
        "2": "Cassava Green Mottle (CGM)",
        "3": "Cassava Mosaic Disease (CMD)",
        "4": "Healthy"
    }
    
    try:
        # Try to find the mapping file
        for root, dirs, files in os.walk("."):
            for file in files:
                if file == "label_num_to_disease_map.json":
                    with open(os.path.join(root, file)) as f:
                        return json.load(f)
        return default_mapping
    except Exception as e:
        print(f"Error loading class mapping: {e}")
        return default_mapping

def load_model_for_inference():
    """Load the trained model"""
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        try:
            # Try to import from CNN.py
            from main import create_model
            model = create_model()
            model.load_weights(WEIGHTS_PATH)
            print("Model loaded from weights successfully!")
            return model
        except Exception as e2:
            print(f"Error loading model weights: {e2}")
            return None

def preprocess_image(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    """Preprocess an image for inference"""
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

def main():
    """Main function"""
    print("\n===== Cassava Leaf Disease Classification - Quick Demo =====\n")
    
    # Load class mapping
    class_mapping = load_class_mapping()
    print("Class mapping loaded:", list(class_mapping.values()))
    
    # Load model
    model = load_model_for_inference()
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Get sample images
    if not os.path.exists(SAMPLE_DIR):
        print(f"Sample directory {SAMPLE_DIR} not found.")
        return
    
    sample_images = [os.path.join(SAMPLE_DIR, f) for f in os.listdir(SAMPLE_DIR) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not sample_images:
        print(f"No images found in {SAMPLE_DIR}")
        return
    
    # Process random images
    num_images = min(3, len(sample_images))
    selected_images = random.sample(sample_images, num_images)
    
    plt.figure(figsize=(15, 5))
    
    for i, img_path in enumerate(selected_images):
        # Load and preprocess image
        img, img_array = preprocess_image(img_path)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100
        
        # Get disease name
        disease_name = class_mapping.get(str(predicted_class), f"Unknown Class {predicted_class}")
        
        # Print results
        print(f"\nImage: {os.path.basename(img_path)}")
        print(f"Prediction: {disease_name}")
        print(f"Confidence: {confidence:.2f}%")
        
        # Display image and prediction
        plt.subplot(1, num_images, i+1)
        plt.imshow(img)
        plt.title(f"{disease_name}\nConfidence: {confidence:.1f}%", fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('quick_demo_results.png')
    plt.show()
    print("\nResults saved as 'quick_demo_results.png'")

if __name__ == "__main__":
    main()
