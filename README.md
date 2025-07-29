
# 🧠 Deep Learning for Tumor Classification

Medical Image Analysis for Benign and Malignant Tumor Classification using PyTorch and the MedMNIST dataset.

## 📌 Project Overview

This project leverages deep learning to classify **Breast MRI** images as either **benign** or **malignant** tumors. The model was built using **PyTorch** and trained on the **BreastMNIST** subset of the MedMNIST dataset. The goal is to provide a lightweight, interpretable, and efficient pipeline for medical image classification tasks.

## 🛠️ Tools & Technologies

The project uses the following core tools:

- **PyTorch**: Primary framework for defining and training the deep learning model.
- **Torchvision.transforms**: Used for data preprocessing and augmentation (e.g., normalization, resizing).
- **TQDM**: For visually appealing progress bars in training loops.
- **NumPy**: For array/tensor manipulation and helper operations.
- **MedMNIST**: A curated collection of lightweight medical image datasets. We use the **BreastMNIST** subset.
- **medmnist.INFO & Evaluator**: For dataset metadata and built-in evaluation with metrics like accuracy and AUC.

## 📂 Dataset

We used the **BreastMNIST** dataset from MedMNIST, which contains grayscale breast MRI images labeled as:
- **0** = Benign tumor
- **1** = Malignant tumor

Each image is standardized and simplified to 28×28 pixels, enabling quick experimentation and benchmarking.

> *MedMNIST is designed for educational and research purposes. The BreastMNIST subset is derived from the Breast Cancer MRI Cohort (TCIA).*

## 🧠 Model Architecture

A simple but effective **Convolutional Neural Network (CNN)** was built in PyTorch with the following layers:
- Convolution + ReLU + MaxPooling
- Fully Connected (FC) layers
- Sigmoid output for binary classification

Training pipeline includes:
- Binary Cross-Entropy Loss
- Adam optimizer
- Accuracy tracking and MedMNIST's built-in evaluator

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tumor-classification-pytorch.git
   cd tumor-classification-pytorch

2. Install dependencies:

   ```bash
   pip install torch torchvision medmnist tqdm numpy
   ```

3. Run training:

   ```bash
   python train.py
   ```

4. Run evaluation:

   ```bash
   python evaluate.py
   ```

## 📈 Results

* Achieved **86% validation accuracy** on binary classification of breast tumor MRI scans.
* Trained using only a few convolutional layers with minimal preprocessing, demonstrating strong performance on a lightweight dataset.

## 🔍 Sample Code Snippet

```python
from medmnist import INFO, Evaluator
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
```

## 🧪 Future Work

- **Increase accuracy by**:
  - 📉 Using a smarter loss function: Implemented `BCEWithLogitsLoss` with `pos_weight` to handle class imbalance more effectively and reduce false negatives.
  - 🕒 Training longer: Increased the number of training epochs (e.g., 25–50) and used early stopping based on validation loss to avoid overfitting.
  - 🧪 Hyperparameter tuning:
    - Learning rates: `1e-3`, `1e-4`, `1e-5`
    - Batch sizes: `32`, `64`
    - Optimizers: **Adam**, **AdamW**, **RMSprop**
    - Learning rate scheduler:
      ```python
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
      ```
- 🔍 Integrate **Grad-CAM** or other explainability tools for visual model interpretability
- 🧠 Experiment with deeper CNNs or **pre-trained architectures** like ResNet or EfficientNet
- 🚀 Deploy as an **interactive web app** using **Streamlit** or **Flask**


*Disclaimer: This project is for educational and research purposes only and is not intended for clinical use.*
