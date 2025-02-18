# SolarCell-Defect-Classification

PyTorch-based project that uses a ResNet convolutional neural network to classify defects in solar cells.

## Overview

This project focuses on detecting defects in solar cells using electroluminescence images. The model classifies two types of defects:

1. **Cracks** – Visible fractures that may or may not impact performance.
2. **Inactive regions** – Areas where the cell has lost connectivity and cannot generate power.

We use a **ResNet-based Convolutional Neural Network (CNN)** implemented in PyTorch to classify images of solar cells into these categories.

---


## Installation

### **1. Clone the Repository**
```bash
git clone https://github.com/kurikkal/SolarCell-Defect-Classification.git
cd SolarCell-Defect-Classification
```

### **2. Install Dependencies**
Ensure you have Python 3.7 installed, then run:
```bash
pip3 install -r requirements.txt
```

### **3. Extract the Dataset**
Since the dataset is uploaded as `images.zip`, you need to extract it:
```bash
cd data/
unzip images.zip
```
This will extract the contents into the `data/images/` directory.


### **4. Run the Training Script**
Once the dataset is ready, you can start training the model by running the training script:
```bash
python3 src/training/train.py
```
This will train the model using the provided dataset, and logs will be saved for analysis.


---
