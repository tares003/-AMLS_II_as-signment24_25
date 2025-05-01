# Cassava Leaf Disease Classification using EfficientNetB0

This repository contains the implementation for the ELEC0135 Applied Machine Learning Systems II (24/25) assignment, based on the [Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/overview) competition on Kaggle.

## Project Overview

Cassava is a crucial food security crop in Africa, providing a basic diet for around 800 million people. However, viral diseases can cause major crop losses of up to 100%. This project implements a deep learning model to identify disease categories in cassava plants using leaf images, helping farmers diagnose problems and apply appropriate interventions.

## Dataset

- **Source**: Cassava Leaf Disease Classification dataset (Kaggle)
- **Classes**: 5 disease categories
  - Class 0: Cassava Bacterial Blight (CBB)
  - Class 1: Cassava Brown Streak Disease (CBSD)
  - Class 2: Cassava Green Mottle (CGM)
  - Class 3: Cassava Mosaic Disease (CMD)
  - Class 4: Healthy
- **Training images**: ~21,000 images
- **Image format**: JPEG
- **Image dimensions**: Varied; resized to 512x512 during processing

## Implementation Details

### Model Architecture

This project implements a CNN (Convolutional Neural Network) using EfficientNetB0 architecture with transfer learning:

- **Base model**: EfficientNetB0 (pre-trained on ImageNet)
- **Model structure**:
  - EfficientNetB0 (without top classification layer)
  - Global Average Pooling 2D
  - Dense layer (5 outputs) with softmax activation
- **Model parameters**:
  - Total parameters: 4,055,976 (15.47 MB)
  - Trainable parameters: 4,013,953 (15.31 MB)
  - Non-trainable parameters: 42,023 (164.16 KB)

### Hardware Configuration

- **GPU**: NVIDIA A100-SXM4-80GB
- **GPU Memory**: 40,749 MB
- **Compute Capability**: 8.0

### Training Configuration

- **Framework**: TensorFlow/Keras
- **Input Size**: 384×384 pixels (RGB images)
- **Batch Size**: 8 (limited due to large image size)
- **Epochs**: 20 (with early stopping)
- **Learning Rate**: Initial 0.001 with progressive reductions via ReduceLROnPlateau:
  - Epoch 8: Reduced to 0.0003
  - Epoch 13: Reduced to 0.00009
  - Epoch 16: Reduced to 0.000027
  - Epoch 19: Reduced to 0.0000081
- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

### Data Preprocessing & Augmentation

To improve model generalization, the following techniques are applied:
- **Image resizing**: All images resized to 512x512 pixels
- **Data split**: 80% training, 20% validation
- **Augmentation techniques**:
  - Random rotation (±45°)
  - Zoom variation (±20%)
  - Horizontal and vertical flips
  - Shear transformations (10%)
  - Height and width shifts (10%)
  - Fill mode: Nearest

## Project Structure

```
├── CNN.py                  # Main implementation file
├── test_gpu.py             # Script to test GPU availability
├── requirements.txt        # Required Python packages
├── README.md               # This file
├── model_cnn_implementation_info.txt  # Detailed implementation documentation
├── model_384_4.h5          # Trained model file
├── EffNetB0_384_4_best.weights.h5  # Best model weights
├── submission.csv          # Kaggle submission file
├── kaggle.json             # Kaggle API credentials (not included in repo)
├── cassava-leaf-disease-classification.zip  # Dataset archive
└── visuals/                # Directory containing visualization images
    ├── training_history.png
    ├── class_distribution.png
    ├── data_augmentation.png
    ├── sample_images.png
    └── multiple_augmentations.png
    └── [multiple activation visualizations]
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

## Training Callbacks and Optimization

The implementation includes several callbacks and optimization techniques:

1. **ModelCheckpoint**:
   - Saves best model weights based on validation loss
   - Only saves weights when improvement is observed

2. **EarlyStopping**:
   - Monitors validation loss
   - Patience: 5 epochs
   - Minimum delta: 0.001
   - Restores best weights

3. **ReduceLROnPlateau**:
   - Monitors validation loss
   - Reduces learning rate by factor of 0.3 when plateau is detected
   - Patience: 2 epochs
   - Minimum delta: 0.001

4. **Other optimizations**:
   - Memory growth control for GPU(s)
   - Optional memory optimization callback

## Visualization Components

The implementation includes multiple visualization tools:

1. **Dataset visualization**:
   - Class distribution
   - Sample images from each class

2. **Data augmentation visualization**:
   - Side-by-side comparison of original and augmented images
   - Multiple augmentation examples

3. **Model activation visualization**:
   - Layer-specific feature map visualization
   - Multiple layer activation visualization

4. **Training history**:
   - Accuracy curves (training and validation)
   - Loss curves (training and validation)

## Model Performance

### Training Results
- **Best Validation Accuracy**: 87.12% (Epoch 17)
- **Best Validation Loss**: 0.38851

### Training Progression
The model showed consistent improvement through training:
- Initial validation accuracy: 75.84% (Epoch 1)
- Early performance jump: 80.51% → 83.73% (Epochs 2-7)
- Learning rate reduction at Epoch 8 (0.001 → 0.0003) led to improved validation accuracy: 86.31% (Epoch 11)
- Further refinement with reduced learning rate (0.00009) at Epoch 14: 87.01%
- Best performance achieved at Epoch 17 with learning rate 0.000027: 87.12%

### Training Duration
- Total training time: 2 hours and 3 minutes
- Average time per epoch: ~6 minutes

## Future Improvements

Several potential improvements could enhance the model performance:

### Data Processing
- Test-time augmentation
- Balanced sampling for class imbalance
- Progressive resizing
- Advanced augmentation techniques (MixUp, CutMix)

### Model Architecture
- Larger EfficientNet variants (B3-B7)
- Ensemble models
- Attention mechanisms
- Multi-layer feature extraction

### Training Strategy
- Multi-stage training with gradual unfreezing
- Advanced learning rate scheduling
- Mixed-precision training
- Alternative loss functions (focal loss, label smoothing)

### Validation & Deployment
- K-fold cross-validation
- Model quantization and pruning
- ONNX conversion for deployment
- Multi-GPU training support

## References

1. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. Tan, M., & Le, Q. V. (2019). ICML.
2. [Cassava Leaf Disease Classification Competition](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/overview)
3. Ramcharan, A., et al. (2017). Deep learning for image-based cassava disease detection. Frontiers in Plant Science, 8, 1852.
