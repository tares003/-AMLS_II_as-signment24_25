# Cassava Leaf Disease Classification using EfficientNetB0

This repository contains the implementation for the ELEC0135 Applied Machine Learning Systems II (24/25) assignment, based on the [Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/overview) competition on Kaggle.

## Project Overview

Cassava is a crucial food security crop in Africa, providing a basic diet for around 800 million people. However, viral diseases can cause major crop losses of up to 100%. This project implements a deep learning model to identify disease categories in cassava plants using leaf images, helping farmers diagnose problems and apply appropriate interventions.

### Disease Categories

The model classifies leaf images into five categories:
- Cassava Bacterial Blight (CBB)
- Cassava Brown Streak Disease (CBSD)
- Cassava Green Mottle (CGM)
- Cassava Mosaic Disease (CMD)
- Healthy

## Implementation Details

### Architecture

This project implements a CNN (Convolutional Neural Network) using EfficientNetB0 architecture with transfer learning. The implementation:

1. Leverages pre-trained weights from ImageNet
2. Adds a custom classification layer for the 5 disease categories
3. Uses data augmentation techniques to improve model robustness
4. Implements callbacks for early stopping and learning rate reduction

### Technical Specifications

- **Framework**: TensorFlow/Keras
- **Base Model**: EfficientNetB0
- **Input Size**: 512×512 pixels (RGB images)
- **Batch Size**: 32
- **Learning Rate**: 0.001 with ReduceLROnPlateau
- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

### Data Augmentation

To improve model generalization, the following augmentation techniques are applied:
- Random rotation (up to 45 degrees)
- Zoom variation (up to 20%)
- Horizontal and vertical flips
- Shear transformations
- Height and width shifts

## Project Structure

```
├── CNN.py                  # Main implementation file
├── test_gpu.py             # Script to test GPU availability
├── requirements.txt        # Required Python packages
├── README.md               # This file
├── kaggle.json             # Kaggle API credentials (not included in repo)
└── cassava-leaf-disease-classification.zip  # Dataset archive
```

## Setup and Running Instructions

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended)
- TensorFlow 2.x
- Required packages listed in `requirements.txt`

### Environment Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd IMLS2
```

2. Create and activate a virtual environment:
```bash
# Using conda (recommended for GPU support)
conda create -n cassava_cnn python=3.9
conda activate cassava_cnn

# Or using venv
python -m venv cassava_env
# On Windows
cassava_env\Scripts\activate
# On Unix/MacOS
source cassava_env/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
   - Option 1: Download directly from [Kaggle](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data)
   - Option 2: Use Kaggle API (requires `kaggle.json` credentials):
   ```bash
   # Place kaggle.json in ~/.kaggle/ or %USERPROFILE%\.kaggle\
   kaggle competitions download -c cassava-leaf-disease-classification
   ```

5. Extract the dataset:
```bash
# Windows
powershell Expand-Archive -Path cassava-leaf-disease-classification.zip -DestinationPath cassava-leaf-disease-classification

# Unix/MacOS
unzip cassava-leaf-disease-classification.zip -d cassava-leaf-disease-classification
```

### Running the Model

Execute the main CNN implementation:
```bash
python CNN.py
```

The script will:
1. Explore the dataset and visualize class distribution
2. Prepare the data with augmentation
3. Create and compile the EfficientNetB0-based model
4. Train the model with the specified parameters
5. Generate visualizations of model performance
6. Create a submission file with predictions on the test set

### Output Files

After training, the following files will be generated:
- `model_384_4.h5`: Full trained model
- `EffNetB0_384_4_best.weights.h5`: Best model weights
- `training_history.png`: Plot of training and validation metrics
- `class_distribution.png`: Visualization of class distribution
- `data_augmentation.png`: Visualization of augmentation effects
- Various activation visualizations
- `submission.csv`: Predictions for submission to Kaggle

## Performance Optimization

The current implementation includes:
- Memory growth control for GPU(s)
- Early stopping to prevent overfitting
- Learning rate reduction on plateaus
- Optional memory optimization callback (commented in code)

## References

1. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. Tan, M., & Le, Q. V. (2019). ICML.
2. [Cassava Leaf Disease Classification Competition](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/overview)
3. Ramcharan, A., et al. (2017). Deep learning for image-based cassava disease detection. Frontiers in Plant Science, 8, 1852.
