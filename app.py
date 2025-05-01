#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cassava Leaf Disease Classification - Streamlit Interface
========================================================
This script provides a web interface to interact with the trained CNN model
for classifying cassava leaf diseases.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import zipfile
import io

# Set page configuration
st.set_page_config(
    page_title="Cassava Leaf Disease Classifier",
    page_icon="🍃",
    layout="wide"
)

# Constants
MODEL_PATH = "./model_384_4.h5"
WEIGHTS_PATH = "./EffNetB0_384_4_best.weights.h5"
IMAGE_SIZE = 384
DATASET_ZIP = "./cassava-leaf-disease-classification.zip"
SAMPLE_DIR = "./test15_flat"

# Ensure the sample directory exists
if not os.path.exists(SAMPLE_DIR):
    os.makedirs(SAMPLE_DIR)

# Load class mapping
def load_class_mapping():
    try:
        # Try to load from the ZIP file first
        with zipfile.ZipFile(DATASET_ZIP, 'r') as z:
            try:
                with z.open("label_num_to_disease_map.json") as f:
                    return json.loads(f.read().decode('utf-8'))
            except KeyError:
                # If not found, create a default mapping
                st.warning("Could not find class mapping in ZIP file. Using default mapping.")
                return {
                    "0": "Cassava Bacterial Blight (CBB)",
                    "1": "Cassava Brown Streak Disease (CBSD)",
                    "2": "Cassava Green Mottle (CGM)",
                    "3": "Cassava Mosaic Disease (CMD)",
                    "4": "Healthy"
                }
    except Exception as e:
        st.error(f"Error loading class mapping: {e}")
        return {
            "0": "Cassava Bacterial Blight (CBB)",
            "1": "Cassava Brown Streak Disease (CBSD)",
            "2": "Cassava Green Mottle (CGM)",
            "3": "Cassava Mosaic Disease (CMD)",
            "4": "Healthy"
        }

# Load model
@st.cache_resource
def load_cassava_model():
    try:
        model = load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.warning("Trying to load model with weights...")
        try:
            # Alternative approach: Try to load weights
            from main import create_model
            model = create_model()
            model.load_weights(WEIGHTS_PATH)
            st.success("✅ Model weights loaded successfully!")
            return model
        except Exception as e2:
            st.error(f"Error loading model weights: {e2}")
            return None

# Extract random sample images from ZIP
def extract_sample_images(n=15):
    if not os.path.exists(DATASET_ZIP):
        st.error(f"Dataset ZIP file not found: {DATASET_ZIP}")
        return []
    
    try:
        with zipfile.ZipFile(DATASET_ZIP, 'r') as z:
            # Get list of all image files in the ZIP
            image_files = [f for f in z.namelist() if 
                          (f.startswith('train_images/') or f.startswith('test_images/')) and 
                          (f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png'))]
            
            # Check if we have enough image files
            if len(image_files) < n:
                st.warning(f"Only {len(image_files)} images available in the ZIP file.")
                n = len(image_files)
            
            # Select random images
            selected_images = random.sample(image_files, n)
            
            # Extract selected images to sample directory
            for img_path in selected_images:
                img_name = os.path.basename(img_path)
                output_path = os.path.join(SAMPLE_DIR, img_name)
                
                if not os.path.exists(output_path):
                    with z.open(img_path) as source, open(output_path, 'wb') as target:
                        target.write(source.read())
            
            return [os.path.join(SAMPLE_DIR, os.path.basename(img_path)) for img_path in selected_images]
    except Exception as e:
        st.error(f"Error extracting sample images: {e}")
        return []

# Preprocess image
def preprocess_image(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict disease
def predict_disease(model, img_array, class_mapping):
    try:
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100
        
        disease_name = class_mapping.get(str(predicted_class), f"Unknown Class {predicted_class}")
        return predicted_class, disease_name, confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, "Error", 0

# Main function
def main():
    st.title("🌱 Cassava Leaf Disease Classifier")
    st.write("This application uses a trained EfficientNetB0 model to classify cassava leaf diseases from images.")
    
    # Load class mapping
    class_mapping = load_class_mapping()
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_cassava_model()
    
    if model is None:
        st.error("Failed to load model. Please check the model path.")
        return
    
    # Tabs
    tab1, tab2 = st.tabs(["🔄 Random Samples", "📤 Upload Image"])
    
    # Tab 1: Random samples
    with tab1:
        st.header("Random Sample Images")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            n_samples = st.slider("Number of random samples to display", 1, 15, 5)
        with col2:
            if st.button("Get Random Samples", type="primary"):
                with st.spinner("Extracting sample images..."):
                    sample_images = extract_sample_images(n_samples)
                    if sample_images:
                        st.session_state['sample_images'] = sample_images
                    else:
                        st.error("Failed to extract sample images.")
        
        # Display sample images
        if 'sample_images' in st.session_state and st.session_state['sample_images']:
            results = []
            
            # Use two columns for display
            cols = st.columns(2)
            
            for i, img_path in enumerate(st.session_state['sample_images']):
                col_idx = i % 2
                with cols[col_idx]:
                    st.subheader(f"Sample {i+1}")
                    
                    # Display image
                    img = Image.open(img_path)
                    st.image(img, use_column_width=True, caption=os.path.basename(img_path))
                    
                    # Make prediction when user clicks
                    if st.button(f"Classify Image {i+1}", key=f"classify_{i}"):
                        img_array = preprocess_image(img_path)
                        predicted_class, disease_name, confidence = predict_disease(model, img_array, class_mapping)
                        
                        # Display results
                        if predicted_class is not None:
                            st.success(f"**Prediction**: {disease_name}")
                            st.info(f"**Confidence**: {confidence:.2f}%")
                            
                            # Add to results for summary
                            results.append({
                                'Image': os.path.basename(img_path),
                                'Predicted Class': predicted_class,
                                'Disease': disease_name,
                                'Confidence': f"{confidence:.2f}%"
                            })
                    
                    st.divider()
            
            # Display results summary if available
            if results:
                st.subheader("Prediction Summary")
                st.dataframe(pd.DataFrame(results))
    
    # Tab 2: Upload image
    with tab2:
        st.header("Upload Your Own Image")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read and display the uploaded image
            image_data = uploaded_file.read()
            img = Image.open(io.BytesIO(image_data))
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(img, caption="Uploaded Image", use_column_width=True)
            
            # Create a temporary file for the uploaded image
            temp_path = os.path.join(SAMPLE_DIR, "temp_upload.jpg")
            with open(temp_path, "wb") as f:
                f.write(image_data)
            
            # Process and predict
            with col2:
                st.subheader("Prediction")
                
                if st.button("Classify Uploaded Image", type="primary"):
                    with st.spinner("Processing..."):
                        # Preprocess the image
                        img_array = preprocess_image(temp_path)
                        
                        # Make prediction
                        predicted_class, disease_name, confidence = predict_disease(model, img_array, class_mapping)
                        
                        # Display results
                        if predicted_class is not None:
                            st.success(f"**Prediction**: {disease_name}")
                            st.info(f"**Confidence**: {confidence:.2f}%")
                            
                            # Display additional information about the disease
                            st.subheader("Disease Information")
                            if "Healthy" in disease_name:
                                st.write("The plant appears to be healthy with no visible signs of disease.")
                            elif "CBB" in disease_name:
                                st.write("**Cassava Bacterial Blight** is characterized by water-soaked angular leaf spots, blight, wilting, and dieback.")
                            elif "CBSD" in disease_name:
                                st.write("**Cassava Brown Streak Disease** causes yellow patches on leaves and brown streaks on stems and root necrosis.")
                            elif "CGM" in disease_name:
                                st.write("**Cassava Green Mottle** shows as light green/yellow mosaic patterns on leaves.")
                            elif "CMD" in disease_name:
                                st.write("**Cassava Mosaic Disease** causes distorted leaves with yellow/green mosaic pattern.")

    # Footer
    st.divider()
    st.caption("Cassava Leaf Disease Classification System | IMLS2 Project")

if __name__ == "__main__":
    main()
