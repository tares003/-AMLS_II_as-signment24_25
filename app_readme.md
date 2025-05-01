# Cassava Leaf Disease Classifier App

This folder contains two applications for the cassava leaf disease classification project:

## 1. Streamlit Web Interface (`app.py`)

A web-based interface for interacting with the trained model:

- View and classify random sample images from the dataset
- Upload and classify your own images
- View disease information and prediction confidence

### How to Run:

```powershell
# Install required packages
pip install -r requirements_streamlit.txt

# Run the app
streamlit run app.py
```

The app will open in your default web browser at http://localhost:8501

## 2. Quick Demo Script (`quick_demo.py`)

A simple console application that:
- Loads the model
- Selects random sample images
- Makes predictions and displays results in a matplotlib figure

### How to Run:

```powershell
python quick_demo.py
```

## Sample Images

The `test15` directory contains 15 random sample images extracted from the dataset for quick testing and demonstration. These are used by both the Streamlit app and the quick demo script.

## Notes:

- Both applications require the trained model file `model_384_4.h5` or the model weights file `EffNetB0_384_4_best.weights.h5`.
- If you want to extract different sample images, you can use the `extract_samples.py` script:

```powershell
python extract_samples.py --num 15 --output ./test15
```
