#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cassava Leaf Disease Classification using EfficientNetB0
=======================================================
This script implements a CNN model for classifying cassava leaf diseases into 5 categories:
0. Cassava Bacterial Blight (CBB)
1. Cassava Brown Streak Disease (CBSD)
2. Cassava Green Mottle (CGM)
3. Cassava Mosaic Disease (CMD)
4. Healthy

The model uses EfficientNetB0 architecture with transfer learning approach.
"""

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import json
import warnings

# Importing machine learning and deep learning libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from PIL import Image

# Suppress warnings for cleaner output
warnings.simplefilter("ignore")

# Force TensorFlow to use GPU 3 which has available memory
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Enable memory growth to avoid consuming all memory at once
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Memory growth enabled for {len(physical_devices)} GPU(s)")
    except Exception as e:
        print(f"Error setting memory growth: {e}")

# Set global parameters
WORK_DIR = './cassava-leaf-disease-classification'  # Path to dataset directory
BATCH_SIZE = 32      # Reduced batch size to save memory
TARGET_SIZE = 512   # Reduced image size to save memory (from 512)
EPOCHS = 3         # Number of training epochs
NUM_CLASSES = 5     # Number of disease categories

def explore_data():
    """
    Function to explore and visualize the dataset
    """
    print("\n" + "="*50)
    print("DATA EXPLORATION")
    print("="*50)
    
    # Count the number of training images
    print('Train images: %d' % len(os.listdir(os.path.join(WORK_DIR, "train_images"))))
    
    # Load class mapping
    with open(os.path.join(WORK_DIR, "label_num_to_disease_map.json")) as file:
        class_mapping = json.loads(file.read())
        print("\nDisease Classes:")
        for key, value in class_mapping.items():
            print(f"Class {key}: {value}")
    
    # Load training labels
    train_labels = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
    print("\nTraining data head:")
    print(train_labels.head())
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax = sns.countplot(x=train_labels.label, 
                      palette=reversed(sns.color_palette("viridis", NUM_CLASSES)))
    
    # Customize plot
    for i in ['top', 'right', 'left']:
        ax.spines[i].set_visible(False)
    ax.spines['bottom'].set_color('black')
    
    plt.xlabel('Disease Classes', fontsize=15)
    plt.ylabel('Number of Samples', fontsize=15)
    plt.title('Class Distribution in Training Set', fontsize=16)
    
    # Add class labels
    total_samples = len(train_labels)
    for i, p in enumerate(ax.patches):
        class_name = class_mapping.get(str(i), "Unknown")
        count = p.get_height()
        percentage = 100 * count / total_samples
        ax.annotate(f'{count} ({percentage:.1f}%)',
                   (p.get_x() + p.get_width()/2., p.get_height() + 50),
                   ha='center', fontsize=10)
        
    plt.xticks(range(NUM_CLASSES), [class_mapping.get(str(i), "Unknown").split('(')[0] for i in range(NUM_CLASSES)], 
              rotation=15, fontsize=10)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    print("Class distribution plot saved as 'class_distribution.png'")
    
    return train_labels

def display_sample_images(train_labels):
    """
    Function to display sample images from each class
    
    Parameters:
    -----------
    train_labels : pandas DataFrame
        DataFrame containing image IDs and labels
    """
    # Load class mapping
    with open(os.path.join(WORK_DIR, "label_num_to_disease_map.json")) as file:
        class_mapping = json.loads(file.read())
    
    # Create a figure to display sample images
    plt.figure(figsize=(15, 10))
    
    # Display 3 sample images from each class
    for class_id in range(NUM_CLASSES):
        # Get sample images from this class
        class_samples = train_labels[train_labels['label'] == class_id]['image_id'].values[:3]
        
        for i, img_id in enumerate(class_samples):
            # Load and display the image
            img_path = os.path.join(WORK_DIR, "train_images", img_id)
            img = image.load_img(img_path, target_size=(TARGET_SIZE, TARGET_SIZE))
            img = image.img_to_array(img) / 255.0
            
            plt.subplot(NUM_CLASSES, 3, class_id*3 + i + 1)
            plt.imshow(img)
            if i == 0:  # Only add class name to first image in each row
                plt.title(f"Class {class_id}: {class_mapping[str(class_id)]}", fontsize=8)
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    print("Sample images saved as 'sample_images.png'")

def prepare_data_generators(train_labels):
    """
    Prepare train and validation data generators with data augmentation
    
    Parameters:
    -----------
    train_labels : pandas DataFrame
        DataFrame containing image IDs and labels
        
    Returns:
    --------
    train_generator : DirectoryIterator
        Generator for training data
    validation_generator : DirectoryIterator
        Generator for validation data
    steps_per_epoch : int
        Number of batches per training epoch
    validation_steps : int
        Number of batches per validation epoch
    """
    print("\n" + "="*50)
    print("DATA PREPARATION")
    print("="*50)
    
    # Convert labels to strings to work with flow_from_dataframe
    train_labels.label = train_labels.label.astype('str')
      # Apply data augmentation to increase effective training set size
    train_datagen = ImageDataGenerator(
        validation_split=0.2,  # Reserve 20% for validation
        rotation_range=45,     # Rotate by ±45°
        zoom_range=0.2,        # Zoom by ±20%
        horizontal_flip=True,  # Horizontal flips
        vertical_flip=True,    # Vertical flips
        fill_mode='nearest',   # Fill strategy
        shear_range=0.1,       # Shear transform
        height_shift_range=0.1,  # Vertical shift
        width_shift_range=0.1,   # Horizontal shift
    )
      # Create training generator
    train_generator = train_datagen.flow_from_dataframe(
        train_labels,
        directory=os.path.join(WORK_DIR, "train_images"),
        subset="training",     # Use training subset
        x_col="image_id",      # Column with image file names
        y_col="label",         # Column with class labels
        target_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="sparse"  # Integer labels
    )

    # For validation, we don't want data augmentation, only rescaling
    validation_datagen = ImageDataGenerator(validation_split=0.2)
    
    # Create validation generator
    validation_generator = validation_datagen.flow_from_dataframe(
        train_labels,
        directory=os.path.join(WORK_DIR, "train_images"),
        subset="validation",   # Use validation subset
        x_col="image_id",
        y_col="label",
        target_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="sparse"
    )
    
    # Calculate steps per epoch
    steps_per_epoch = len(train_generator)
    validation_steps = len(validation_generator)
    
    # Demonstrate data augmentation
    visualize_augmentation(train_labels, train_datagen)
    
    return train_generator, validation_generator, steps_per_epoch, validation_steps

def visualize_augmentation(train_labels, train_datagen):
    """
    Visualize the effect of data augmentation on a sample image
    
    Parameters:
    -----------
    train_labels : pandas DataFrame
        DataFrame containing image IDs and labels
    train_datagen : ImageDataGenerator
        Data generator with augmentation settings
    """
    # Choose a random image
    try:
        sample_idx = np.random.randint(0, len(train_labels))
        img_path = os.path.join(WORK_DIR, "train_images", train_labels.image_id[sample_idx])
        
        # Check if file exists
        if not os.path.exists(img_path):
            print(f"WARNING: Image file not found: {img_path}")
            print("Skipping data augmentation visualization.")
            return
            
        # Load and display original image
        img = image.load_img(img_path, target_size=(TARGET_SIZE, TARGET_SIZE))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        # Create a generator for the single image to show augmentations
        generator = train_datagen.flow_from_dataframe(
            train_labels.iloc[sample_idx:sample_idx+1],
            directory=os.path.join(WORK_DIR, "train_images"),
            x_col="image_id",
            y_col="label",
            target_size=(TARGET_SIZE, TARGET_SIZE),
            batch_size=1,
            class_mode="sparse"
        )
        
        # Get the augmented image - using next() function instead of .next() method
        aug_batch = next(generator)
        aug_img = aug_batch[0][0]
        
        plt.subplot(1, 2, 2)
        plt.imshow(aug_img)
        plt.title('Augmented Image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('data_augmentation.png')
        print("Data augmentation visualization saved as 'data_augmentation.png'")
        
        # Show multiple augmentation examples
        plt.figure(figsize=(15, 8))
        
        # Get 8 different augmented versions of the same image
        aug_images = []
        for i in range(8):
            aug_batch = next(generator)
            aug_images.append(aug_batch[0][0])
        
        for i, aug_img in enumerate(aug_images):
            plt.subplot(2, 4, i+1)
            plt.imshow(aug_img)
            plt.title(f'Augmentation {i+1}')
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig('multiple_augmentations.png')
        print("Multiple augmentations visualization saved as 'multiple_augmentations.png'")
    
    except Exception as e:
        print(f"Error in visualize_augmentation: {e}")
        print("Skipping data augmentation visualization and continuing.")
        
def create_model():
    """
    Create and compile the CNN model using EfficientNetB0 architecture
    
    Returns:
    --------
    model : Keras Model
        The compiled CNN model
    """
    print("\n" + "="*50)
    print("MODEL CREATION")
    print("="*50)
    
    try:
        # Get GPU info
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            device_details = tf.config.experimental.get_device_details(gpu_devices[0])
            print(f"Using GPU: {device_details.get('device_name', 'Unknown')}")
        
        print("Loading EfficientNetB0 with memory optimizations...")
        
        # Create base model with transfer learning
        conv_base = EfficientNetB0(
            include_top=False,  # Remove the classification layer
            weights='imagenet',  # Use pre-trained weights from ImageNet
            input_shape=(TARGET_SIZE, TARGET_SIZE, 3),  # RGB image input shape
            pooling='avg'  # Use global average pooling to reduce parameters
        )
          # Build final model
        x = conv_base.output
        
        # Add dropout for regularization
        x = layers.Dropout(0.2)(x)
        
        # fully connected layer with 5 outputs and softmax activation
        # These correspond to the 5 disease categories
        predictions = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        
        # Create model
        model = models.Model(inputs=conv_base.input, outputs=predictions)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        print("\nModel Summary:")
        model.summary()
        
        return model
    except Exception as e:
        print(f"Error in model creation: {e}")
        print("\nTrying alternative approach with lighter model...")
        
        # Alternative lightweight model if EfficientNetB0 fails
        inputs = layers.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3))
        
        # Use a simpler CNN architecture
        x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nUsing lightweight model instead")
        model.summary()
        
        return model

def train_model(model, train_generator, validation_generator, steps_per_epoch, validation_steps):
    """
    Train the model with callbacks for early stopping and learning rate adjustment
    
    Parameters:
    -----------
    model : Keras Model
        The compiled CNN model
    train_generator : DirectoryIterator
        Generator for training data
    validation_generator : DirectoryIterator
        Generator for validation data
    steps_per_epoch : int
        Number of batches per training epoch
    validation_steps : int
        Number of batches per validation epoch
        
    Returns:
    --------
    history : History object
        The training history
    """
    print("\n" + "="*50)
    print("MODEL TRAINING")
    print("="*50)
      # Training callbacks
    callbacks = [
        # Save best model weights based on validation loss
        ModelCheckpoint(
            './EffNetB0_384_4_best.weights.h5',
            save_best_only=True,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=5,
            mode='min',
            verbose=1,
            restore_best_weights=True
        ),
        
        # Reduce learning rate when learning plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=2,
            min_delta=0.001,
            mode='min',
            verbose=1
        )
    ]
    
    # # Memory optimization callback
    # class MemoryOptimizationCallback(tf.keras.callbacks.Callback):
    #     def on_epoch_end(self, epoch, logs=None):
    #         # Clear memory between epochs
    #         import gc
    #         gc.collect()
    #         tf.keras.backend.clear_session()
    #         print("\nMemory cleared after epoch")
    
    # # Add memory optimization to callbacks
    # callbacks.append(MemoryOptimizationCallback())
    
    # Start timer for training
    start_time = datetime.datetime.now()
    print(f"Training started at: {start_time}")
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    # End timer and print duration
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"Training finished at: {end_time}")
    print(f"Total training time: {duration}")
    
    # Save the full model
    model.save('./model_384_4.h5')
    print("Full model saved as 'model_384_4.h5'")
    
    # Plot training history
    plot_training_history(history)
    
    return history

def plot_training_history(history):
    """
    Plot training and validation metrics
    
    Parameters:
    -----------
    history : History object
        The training history
    """
    # Create figure with two subplots
    plt.figure(figsize=(15, 6))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")

def activation_layer_vis(model, img_tensor, activation_layer=0, layers=10):
    """
    Visualize activations of specific layer in the CNN
    
    Parameters:
    -----------
    model : Keras Model
        The trained CNN model
    img_tensor : numpy array
        Input image tensor
    activation_layer : int
        Index of the layer to visualize
    layers : int
        Number of layers to include in the activation model
    """
    # Get outputs of first few layers
    layer_outputs = [layer.output for layer in model.layers[:layers]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)
    
    # Calculate grid size based on number of channels
    rows = int(activations[activation_layer].shape[3] / 3)
    cols = int(activations[activation_layer].shape[3] / rows)
    
    # Create figure and display activations
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15 * rows/cols))
    axes = axes.flatten()
    
    for i, ax in zip(range(activations[activation_layer].shape[3]), axes):
        ax.matshow(activations[activation_layer][0, :, :, i], cmap='viridis')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'layer_{activation_layer}_activations.png')
    print(f"Layer {activation_layer} activation visualization saved as 'layer_{activation_layer}_activations.png'")

def all_activations_vis(model, img_tensor, layers=10):
    """
    Visualize activations for multiple layers
    
    Parameters:
    -----------
    model : Keras Model
        The trained CNN model
    img_tensor : numpy array
        Input image tensor
    layers : int
        Number of layers to visualize
    """
    # Get layer outputs
    layer_outputs = [layer.output for layer in model.layers[:layers]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)
    
    # Get layer names
    layer_names = []
    for layer in model.layers[:layers]: 
        layer_names.append(layer.name)

    # Visualize each layer
    images_per_row = 3
    for layer_name, layer_activation in zip(layer_names, activations): 
        # Get feature dimensions
        n_features = layer_activation.shape[-1] 
        size = layer_activation.shape[1]

        # Create display grid
        n_cols = n_features // images_per_row 
        display_grid = np.zeros((size * n_cols, images_per_row * size)) 

        # Fill grid with processed channel images
        for col in range(n_cols): 
            for row in range(images_per_row): 
                channel_image = layer_activation[0, :, :, col * images_per_row + row] 
                # Normalize for visualization
                channel_image -= channel_image.mean() 
                channel_image /= channel_image.std() 
                channel_image *= 64 
                channel_image += 128 
                channel_image = np.clip(channel_image, 0, 255).astype('uint8') 
                display_grid[col * size : (col + 1) * size, 
                             row * size : (row + 1) * size] = channel_image 
        
        # Display the grid
        scale = 1. / size 
        plt.figure(figsize=(scale * 5 * display_grid.shape[1], 
                            scale * 5 * display_grid.shape[0])) 
        plt.title(layer_name) 
        plt.grid(False)
        plt.axis('off')
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.savefig(f'all_activations_layer_{layer_name}.png')
    
    print("All activations visualizations saved")

def make_predictions(model):
    """
    Make predictions on the test data
    
    Parameters:
    -----------
    model : Keras Model
        The trained CNN model
    """
    print("\n" + "="*50)
    print("PREDICTIONS ON TEST DATA")
    print("="*50)
    
    # Load sample submission
    sample_submission = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))
    print("Making predictions on", len(sample_submission), "test images")
    
    # Make predictions on test images
    preds = []
    
    for image_id in sample_submission.image_id:
        # Load and preprocess image
        image = Image.open(os.path.join(WORK_DIR, "test_images", image_id))
        image = image.resize((TARGET_SIZE, TARGET_SIZE))
        image = np.expand_dims(image, axis=0)
        
        # Get prediction and find most likely class
        pred = np.argmax(model.predict(image))
        preds.append(pred)
    
    # Update submission file
    sample_submission['label'] = preds
    sample_submission.to_csv('submission.csv', index=False)
    print("Predictions saved to 'submission.csv'")
    
    return sample_submission

def main():
    """
    Main function to orchestrate the entire workflow
    """
    print("\n" + "="*50)
    print("CASSAVA LEAF DISEASE CLASSIFICATION")
    print("="*50)
    
    # Step 1: Explore the dataset
    train_labels = explore_data()
    
    # Step 2: Display sample images
    display_sample_images(train_labels)
    
    # Step 3: Prepare data generators
    train_generator, validation_generator, steps_per_epoch, validation_steps = prepare_data_generators(train_labels)
      # Create and train model
    model = create_model()
    history = train_model(model, train_generator, validation_generator, steps_per_epoch, validation_steps)
    
    # Visualize layer activations
    random_idx = np.random.randint(0, len(train_labels))
    img_path = os.path.join(WORK_DIR, "train_images", train_labels.image_id[random_idx])
    img = image.load_img(img_path, target_size=(TARGET_SIZE, TARGET_SIZE))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    activation_layer_vis(model, img_tensor, activation_layer=0)
    all_activations_vis(model, img_tensor, layers=5)
    
    # Generate predictions
    make_predictions(model)
    
    print("\n" + "="*50)
    print("CASSAVA LEAF DISEASE CLASSIFICATION COMPLETE")
    print("="*50)

if __name__ == "__main__":
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU(s) detected: {len(gpus)}")
        # Print visible devices
        print(f"Using GPU(s): {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
        try:
            # Set memory growth to avoid consuming all memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected, using CPU")
    
    # Run the main function
    main()
