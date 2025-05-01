
import json
import random
import zipfile

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import os
from main import create_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Configuration
MODEL_PATH = "./model_384_4.h5"
WEIGHTS_PATH = "./EffNetB0_384_4_best.weights.h5"
DATASET_ZIP = "./cassava-leaf-disease-classification.zip"
IMAGE_SIZE = 384
OUTPUT_DIR = "./visuals"

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_class_mapping():
    """Load class mapping from dataset or use default"""
    default_mapping = {
        "0": "Cassava Bacterial Blight (CBB)",
        "1": "Cassava Brown Streak Disease (CBSD)",
        "2": "Cassava Green Mottle (CGM)",
        "3": "Cassava Mosaic Disease (CMD)",
        "4": "Healthy"
    }
    
    try:
        # Try to load from the ZIP file first
        with zipfile.ZipFile(DATASET_ZIP, 'r') as z:
            try:
                with z.open("label_num_to_disease_map.json") as f:
                    return json.loads(f.read().decode('utf-8'))
            except KeyError:
                # If not found, use default mapping
                print("Could not find class mapping in ZIP file. Using default mapping.")
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
            # Alternative approach: Try to load weights
            model = create_model()
            model.load_weights(WEIGHTS_PATH)
            print("Model weights loaded successfully!")
            return model
        except Exception as e2:
            print(f"Error loading model weights: {e2}")
            return None

def preprocess_image(img, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    """Preprocess an image for inference"""
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def generate_confusion_matrix(samples_per_class=856):
    """
    Generate confusion matrix from validation data
    
    Parameters:
    -----------
    samples_per_class : int
        Number of validation samples to use per class
    """
    print(f"\n===== Generating Confusion Matrix (using {samples_per_class} samples per class) =====\n")
    
    # Load class mapping
    class_mapping = load_class_mapping()
    print("Class mapping loaded:", list(class_mapping.values()))
    
    # Load model
    model = load_model_for_inference()
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Check if dataset exists
    if not os.path.exists(DATASET_ZIP):
        print(f"Dataset ZIP file not found: {DATASET_ZIP}")
        return
    
    try:
        with zipfile.ZipFile(DATASET_ZIP, 'r') as z:
            # Load train.csv to get labeled data
            with z.open('train.csv') as f:
                train_data = pd.read_csv(f)
                print(f"Loaded {len(train_data)} labeled examples from train.csv")
            
            # Create temporary directory for validation images
            val_dir = "./temp_val"
            os.makedirs(val_dir, exist_ok=True)
            
            # Get validation samples (stratified by class)
            val_data = pd.DataFrame()
            for class_id in range(5):
                class_samples = train_data[train_data['label'] == class_id].sample(
                    min(samples_per_class, sum(train_data['label'] == class_id)),
                    random_state=42
                )
                val_data = pd.concat([val_data, class_samples])
            
            print(f"Selected {len(val_data)} validation samples ({val_data['label'].value_counts().to_dict()})")
            
            # Process validation images
            y_true = []
            y_pred = []
            img_paths = []
            
            for i, (_, row) in enumerate(val_data.iterrows()):
                print(f"Processing image {i+1}/{len(val_data)}: {row['image_id']}")
                
                img_path = f"train_images/{row['image_id']}"
                temp_path = os.path.join(val_dir, row['image_id'])
                img_paths.append(row['image_id'])
                y_true.append(row['label'])
                
                # Extract image
                with z.open(img_path) as source, open(temp_path, 'wb') as target:
                    target.write(source.read())
                
                # Preprocess and predict
                img = Image.open(temp_path)
                img_array = preprocess_image(img)
                predictions = model.predict(img_array, verbose=0)
                predicted_class = np.argmax(predictions[0])
                y_pred.append(predicted_class)
            
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot confusion matrix
            plt.figure(figsize=(12, 10))
            display_labels = [class_mapping.get(str(i), f"Class {i}") for i in range(5)]
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            disp.plot(cmap='Blues', values_format='d', xticks_rotation='vertical')
            plt.title('Confusion Matrix - Cassava Leaf Disease Classification')
            plt.tight_layout()
            
            # Save and show the plot
            cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
            plt.savefig(cm_path)
            print(f"Confusion matrix saved to {cm_path}")
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            print("\n===== Model Performance Metrics =====")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1 Score:  {f1:.4f}\n")
            
            # Per-class metrics
            class_precision = precision_score(y_true, y_pred, average=None)
            class_recall = recall_score(y_true, y_pred, average=None)
            class_f1 = f1_score(y_true, y_pred, average=None)
            
            print("===== Per-class Performance =====")
            for i in range(5):
                disease_name = class_mapping.get(str(i), f"Class {i}")
                print(f"{disease_name}:")
                print(f"  Precision: {class_precision[i]:.4f}")
                print(f"  Recall:    {class_recall[i]:.4f}")
                print(f"  F1 Score:  {class_f1[i]:.4f}")
            
            # Save predictions to CSV
            results_df = pd.DataFrame({
                'Image': img_paths,
                'True_Label': y_true,
                'True_Label_Name': [class_mapping.get(str(lbl), f"Class {lbl}") for lbl in y_true],
                'Predicted_Label': y_pred,
                'Predicted_Label_Name': [class_mapping.get(str(lbl), f"Class {lbl}") for lbl in y_pred],
                'Correct': [t == p for t, p in zip(y_true, y_pred)]
            })
            
            results_path = os.path.join(OUTPUT_DIR, "confusion_matrix_results.csv")
            results_df.to_csv(results_path, index=False)
            print(f"Detailed results saved to {results_path}")
            
            # Clean up temporary files
            for file in os.listdir(val_dir):
                os.remove(os.path.join(val_dir, file))
            os.rmdir(val_dir)
            print("Temporary files cleaned up")
            
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate confusion matrix for cassava leaf disease classification")
    parser.add_argument("--samples", type=int, default=856, help="Number of validation samples per class")
    
    args = parser.parse_args()
    generate_confusion_matrix(args.samples)
