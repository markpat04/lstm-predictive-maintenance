<div align="center">

# 🏭 LSTM Predictive Maintenance


![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep_Learning-FF6F00?logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-Preprocessing-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Maintenance-Active-blueviolet)

[View Architecture](#-system-architecture) • [Theory](#-theoretical-background)

</div>

---

## 📖 Overview

This project implements an end-to-end **Predictive Maintenance (PdM)** pipeline using Deep Learning. It is designed to interpret raw accelerometer data from industrial motors and classify their health status in real-time.

By training a **Long Short-Term Memory (LSTM)** neural network, the system moves beyond simple threshold-based detection to identify complex temporal patterns associated with specific mechanical faults.

---

## ⚡ Key Features

* **🔍 Multi-Class Fault Detection:** Capable of distinguishing between Normal, Bearing Fault, Misalignment, and Imbalance states.
* **🧠 Deep Temporal Learning:** Utilizes LSTM layers to capture long-term dependencies in time-series vibration data.
* **📉 Smart Preprocessing:** Automated Z-Score normalization and 3D tensor reshaping for neural network ingestion.
* **📊 Comprehensive Evaluation:**
    * **Confusion Matrix:** To visualize misclassification risks.
    * **Learning Curves:** To monitor overfitting/underfitting during training.
* **💾 Model Persistence:** Automatic serialization of both the model (`.h5`) and the scaler (`.pkl`) for immediate deployment.

---

## 📸 Visual Output

The training script generates high-resolution visualization artifacts to validate model performance:

| Artifact | Analysis Layer | Description |
| :--- | :--- | :--- |
| **1** | **Training History** | Tracks Loss and Accuracy over epochs to ensure convergence. |
| **2** | **Confusion Matrix** | Heatmap showing True Labels vs. Predicted Labels (Diagonal = Success). |
| **3** | **Classification Report** | Precision, Recall, and F1-Score breakdown for every fault class. |

> *Note: Plots are automatically saved to the `output/` directory after training.*

---

## 📐 Theoretical Background

This project applies advanced mathematical concepts to industrial reliability.

### 1. Z-Score Normalization
Raw sensor data varies in amplitude. We normalize inputs to zero mean and unit variance to stabilize gradient descent.

$$z = \frac{x - \mu}{\sigma}$$

Where $\mu$ is the mean and $\sigma$ is the standard deviation of the training set.

### 2. LSTM Memory Cell
Unlike standard RNNs, LSTMs mitigate the vanishing gradient problem using a cell state $C_t$ regulated by gates.

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

### 3. Softmax Classification
The final dense layer converts the network's raw logits into a probability distribution over the 4 classes ($K=4$).

$$P(y=j|x) = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}$$
