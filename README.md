# pytorch-mlp-classification-regression-lab
"PyTorch lab project implementing DNN/MLP models for classification and regression with regularization and performance analysis."

# 📘 **Lab 1 — Deep Learning with PyTorch**

### **DNN/MLP for Regression & Multi-Class Classification**

---

## 🎯 **Objective**

The main purpose of this lab is to become familiar with the **PyTorch** library by implementing **Regression** and **Multi-Class Classification** tasks using **Deep Neural Network (DNN/MLP) architectures**.
You will explore data, build models, tune hyperparameters, apply regularization, and evaluate performance.

---

## 🧰 **Tools Used**

* **Python**
* **PyTorch**
* **Scikit-Learn**
* **Pandas / NumPy**
* **Matplotlib / Seaborn**
* **Google Colab or Kaggle**
* **Git / GitHub**

---

# 🧩 **Part 1 — Regression Task**

### 📌 **Dataset**

➡️ NYSE Dataset:
[https://www.kaggle.com/datasets/dgawlik/nyse](https://www.kaggle.com/datasets/dgawlik/nyse)

---

### ✔️ **1. Exploratory Data Analysis (EDA)**

* Descriptive statistics
* Missing values
* Correlation matrix
* Visualizations (histograms, boxplots, heatmaps, etc.)

---

### ✔️ **2. Build a DNN/MLP for Regression (PyTorch)**

* Input/hidden/output layers
* Activation functions
* Optimizer
* Loss function (MSELoss)
* Training loop

---

### ✔️ **3. Hyperparameter Tuning (GridSearch – sklearn)**

Parameters explored:

* Learning rate
* Optimizer (SGD, Adam, RMSProp…)
* Number of layers & neurons
* Batch size
* Epochs

---

### ✔️ **4. Plot & Interpret Training Curves**

* **Loss vs Epochs (Train/Test)**
* **Accuracy vs Epochs (Train/Test)** *(if accuracy applies)*

Explain:

* Underfitting / overfitting
* Convergence
* Stability

---

### ✔️ **5. Apply Regularization Techniques**

Regularization methods tested:

* **Dropout**
* **Weight decay (L2)**
* **Batch Normalization**
* **Early Stopping**

Compare results with the first (non-regularized) model.


# 🧩 **Part 2 — Multi-Class Classification Task**

### 📌 **Dataset**

➡️ Predictive Maintenance Dataset:
[https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)

---

### ✔️ **1. Data Cleaning & Pre-Processing**

* Handle missing values
* Encoding categorical features
* Standardization / Normalization

---

### ✔️ **2. Exploratory Data Analysis (EDA)**

* Class distribution
* Correlation analysis
* Feature distributions
* Outliers

---

### ✔️ **3. Apply Data Augmentation**

Since the dataset is imbalanced:

* Oversampling (SMOTE)
* Undersampling
* Synthetic data generation

---

### ✔️ **4. Build DNN/MLP for Multi-Class Classification**

* CrossEntropyLoss
* One-hot labels (if needed)
* Softmax output layer

---

### ✔️ **5. Hyperparameter Tuning (GridSearch – sklearn)**

Tune:

* LR
* Optimizer
* Number of neurons
* Number of layers
* Batch size
* Epoch count

---

### ✔️ **6. Plot & Interpret Training Curves**

* Loss vs Epochs (Train/Test)
* Accuracy vs Epochs (Train/Test)

---

### ✔️ **7. Compute Performance Metrics**

For both **train** and **test** sets:

* Accuracy
* Precision
* Recall (Sensitivity)
* F1-Score
* Confusion Matrix

---

### ✔️ **8. Apply Regularization Techniques**

Compare model performance **before/after**:

* Dropout
* Weight decay (L2)
* BatchNorm
* Early stopping

---

# 📝 **Conclusion**

### 📌 **Key Learnings Synthesis**

* Understanding PyTorch workflow
* Building DNN/MLP architectures
* Applying data pre-processing
* Hyperparameter tuning with GridSearch
* Interpreting training curves
* Using mathematical metrics (Precision, Recall, F1, Cross-Entropy, MSE)
* Evaluating overfitting and using regularization

### 📌 **Comparative Analysis**

* Performance improvement after tuning
* Effectiveness of regularization
* Comparison between regression and classification behaviors
* Impact of balanced vs imbalanced data on model learning

---

# 📦 **Repository Structure**

```
📁 pytorch-mlp-classification-regression-lab
│── 📁 regression
│   ├── eda.ipynb
│   ├── regression_model.py
│   ├── gridsearch.ipynb
│   ├── results/
│
│── 📁 classification
│   ├── preprocessing.ipynb
│   ├── classification_model.py
│   ├── regularization_experiments.ipynb
│   ├── results/
│
│── README.md
│── requirements.txt
```

---

# 👨‍🏫 **Instructor**

**Pr. ELAACHAK LOTFI**
*MBD Master — Deep Learning*
*Université Abdelmalek Essaadi, FST Tanger*

