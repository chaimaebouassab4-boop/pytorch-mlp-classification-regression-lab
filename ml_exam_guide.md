# Deep Learning Exam Revision Guide 🧠
## PyTorch DNN/MLP for Classification & Regression

---

## Table of Contents
1. [Neural Network Fundamentals](#1-neural-network-fundamentals)
2. [Mathematical Concepts](#2-mathematical-concepts)
3. [Loss Functions](#3-loss-functions)
4. [Optimization & Gradient Descent](#4-optimization--gradient-descent)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Overfitting vs Underfitting](#6-overfitting-vs-underfitting)
7. [Regularization Techniques](#7-regularization-techniques)
8. [Hyperparameter Tuning](#8-hyperparameter-tuning)
9. [Model Evaluation Metrics](#9-model-evaluation-metrics)
10. [Data Augmentation](#10-data-augmentation)
11. [Practical PyTorch Implementation](#11-practical-pytorch-implementation)

---

## 1. Neural Network Fundamentals

### What is a Deep Neural Network (DNN)?
A DNN is composed of multiple layers of interconnected neurons that transform input data into output predictions through weighted connections.

**Architecture Components:**
- **Input Layer**: Receives raw features (size = number of features)
- **Hidden Layers**: Perform transformations and feature extraction
- **Output Layer**: Produces final prediction
  - Regression: 1 neuron
  - Binary Classification: 1 neuron (sigmoid)
  - Multi-class Classification: n neurons (softmax)

### Multi-Layer Perceptron (MLP)
An MLP is a feedforward neural network where information flows in one direction: input → hidden layers → output.

**Example Architecture (from lab):**
```
Input (n features) → Hidden1 (64) → Hidden2 (32) → Hidden3 (16) → Output
```

---

## 2. Mathematical Concepts

### 2.1 Activation Functions

#### ReLU (Rectified Linear Unit)
```
ReLU(x) = max(0, x)
```
- **Properties**: 
  - Non-linear transformation
  - Solves vanishing gradient problem
  - Computationally efficient
  - Can cause "dying ReLU" (neurons output 0 forever)

#### Sigmoid
```
σ(x) = 1 / (1 + e^(-x))
```
- **Range**: (0, 1)
- **Use**: Binary classification, output probabilities
- **Problem**: Vanishing gradients for extreme values

#### Softmax (Multi-class Classification)
```
Softmax(x_i) = e^(x_i) / Σ(e^(x_j)) for all j
```
- Converts logits to probability distribution
- All outputs sum to 1
- Used in multi-class classification output layer

#### Tanh
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```
- **Range**: (-1, 1)
- Zero-centered (better than sigmoid)

---

## 3. Loss Functions

### 3.1 Mean Squared Error (MSE) - Regression
```
MSE = (1/n) Σ(y_true - y_pred)²
```
- Measures average squared difference
- Sensitive to outliers
- Always non-negative
- **From lab**: Best MSE = 0.0010 (with early stopping)

### 3.2 Cross-Entropy Loss - Classification

#### Binary Cross-Entropy
```
BCE = -(1/n) Σ [y_true × log(y_pred) + (1-y_true) × log(1-y_pred)]
```

#### Categorical Cross-Entropy (Multi-class)
```
CCE = -(1/n) Σ Σ y_true_i × log(y_pred_i)
```
- Used with softmax activation
- **PyTorch**: `nn.CrossEntropyLoss` combines softmax + negative log-likelihood

---

## 4. Optimization & Gradient Descent

### 4.1 Gradient Descent Concept
The goal is to minimize the loss function by updating weights in the direction of steepest descent.

**Basic Update Rule:**
```
θ_new = θ_old - η × ∇L(θ)
```
Where:
- θ = parameters (weights and biases)
- η = learning rate
- ∇L(θ) = gradient of loss with respect to parameters

### 4.2 Backpropagation
Algorithm to compute gradients efficiently using the chain rule:
```
∂L/∂w = ∂L/∂output × ∂output/∂w
```

### 4.3 Optimizer Variants

#### Stochastic Gradient Descent (SGD)
```
θ = θ - η × ∇L(θ; x_i, y_i)
```
- Updates weights after each sample
- Fast but noisy convergence

#### SGD with Momentum
```
v_t = β × v_(t-1) + η × ∇L(θ)
θ = θ - v_t
```
- Accumulates velocity in consistent directions
- Reduces oscillations
- β typically 0.9

#### Adam (Adaptive Moment Estimation) ⭐
```
m_t = β1 × m_(t-1) + (1-β1) × ∇L(θ)     [First moment]
v_t = β2 × v_(t-1) + (1-β2) × (∇L(θ))²  [Second moment]

m̂_t = m_t / (1 - β1^t)  [Bias correction]
v̂_t = v_t / (1 - β2^t)

θ = θ - η × m̂_t / (√v̂_t + ε)
```
- **Default values**: β1=0.9, β2=0.999, ε=1e-8
- Combines momentum + adaptive learning rates
- **From lab**: Adam was optimal optimizer
- Most popular for deep learning

### 4.4 Learning Rate (η)

**Critical Hyperparameter:**
- **Too high**: Divergence, overshooting minimum
- **Too low**: Slow convergence, stuck in local minima
- **From lab**: η = 0.001 was optimal

**Learning Rate Schedules:**
- Step Decay: Reduce by factor every n epochs
- Exponential Decay: η = η₀ × e^(-kt)
- Cosine Annealing: Smooth periodic decay

---

## 5. Data Preprocessing

### 5.1 Handling Missing Values
**Strategies:**
1. **Remove**: Drop rows/columns with missing data
2. **Imputation**: 
   - Mean/Median (numerical)
   - Mode (categorical)
   - Forward/Backward fill (time series)

### 5.2 Standardization (Z-score Normalization)
```
x_standardized = (x - μ) / σ
```
- Mean μ = 0, Standard deviation σ = 1
- **Use when**: Features have different scales
- **Preserves**: Outliers information
- **Required for**: Distance-based algorithms, gradient descent

### 5.3 Normalization (Min-Max Scaling)
```
x_normalized = (x - x_min) / (x_max - x_min)
```
- Scales to [0, 1] range
- **Sensitive to**: Outliers
- **Use when**: Need bounded range

### 5.4 Encoding Categorical Variables

#### One-Hot Encoding
```
Color: [Red, Blue, Green]
Red   → [1, 0, 0]
Blue  → [0, 1, 0]
Green → [0, 0, 1]
```

#### Label Encoding
```
Color: [Red, Blue, Green] → [0, 1, 2]
```
- Only for ordinal data (with natural order)

---

## 6. Overfitting vs Underfitting

### 6.1 Overfitting 📈
**Definition**: Model learns training data too well, including noise, failing to generalize.

**Signs:**
- High training accuracy, low test accuracy
- Large gap between training and validation loss
- **From lab**: Training accuracy 98% but poor minority class F1-scores

**Causes:**
- Model too complex
- Too many parameters
- Training too long
- Insufficient data

**Solutions:**
- Regularization (Dropout, L1/L2)
- More training data
- Data augmentation
- Early stopping
- Simpler architecture

### 6.2 Underfitting 📉
**Definition**: Model too simple to capture data patterns.

**Signs:**
- Low training AND test accuracy
- High bias
- **From lab**: High regularization (Dropout 0.5 + L2 1e-3) caused underfitting

**Causes:**
- Model too simple
- Too much regularization
- Insufficient training

**Solutions:**
- Increase model complexity
- Reduce regularization
- Train longer
- Better features

### 6.3 Bias-Variance Tradeoff

```
Total Error = Bias² + Variance + Irreducible Error
```

- **High Bias**: Underfitting (model too simple)
- **High Variance**: Overfitting (model too complex)
- **Goal**: Find sweet spot

---

## 7. Regularization Techniques

### 7.1 Dropout
**Mechanism**: Randomly deactivate neurons during training with probability p.

```
During Training:
output = (input × mask) / (1 - p)
where mask ~ Bernoulli(1 - p)

During Testing:
output = input  (no dropout)
```

**Effect:**
- Prevents co-adaptation of neurons
- Forces network to learn redundant representations
- Acts as ensemble learning
- **From lab**: Dropout 0.3 was optimal, 0.5 was too much

**Implementation:**
```python
self.dropout = nn.Dropout(p=0.3)
```

### 7.2 L2 Regularization (Weight Decay)
**Loss Function:**
```
L_total = L_original + λ × Σ(w²)
```

**Gradient Update:**
```
w = w - η × (∂L/∂w + 2λw)
  = (1 - 2ηλ) × w - η × ∂L/∂w
```

**Effect:**
- Penalizes large weights
- Encourages small, distributed weights
- Prevents overfitting
- **From lab**: L2 = 1e-5 optimal, 1e-3 too strong

### 7.3 L1 Regularization
```
L_total = L_original + λ × Σ|w|
```

**Effect:**
- Encourages sparsity (many weights → 0)
- Feature selection
- Less smooth than L2

### 7.4 Batch Normalization
**Formula:**
```
x̂ = (x - μ_batch) / √(σ²_batch + ε)
y = γ × x̂ + β
```

Where γ, β are learnable parameters.

**Benefits:**
- Stabilizes activations
- Allows higher learning rates
- Reduces internal covariate shift
- Acts as regularization

**Implementation:**
```python
self.bn1 = nn.BatchNorm1d(num_features)
```

### 7.5 Early Stopping ⭐
**Method**: Stop training when validation loss stops improving.

**Algorithm:**
1. Monitor validation loss each epoch
2. Save best model
3. Stop if no improvement for n epochs (patience)

**From lab**: 
- Stopped at epoch 16
- Best performance: MSE = 0.0010, R² = 0.9990

---

## 8. Hyperparameter Tuning

### 8.1 Key Hyperparameters

#### Model Architecture
- Number of hidden layers
- Neurons per layer
- **From lab**: 64 → 32 → 16 (regression), 50 neurons (classification)

#### Learning Rate (η)
- **Critical**: Most important hyperparameter
- **From lab**: 0.001 optimal
- **Range**: Typically [1e-5, 1e-1]

#### Batch Size
- **Small**: Noisy gradients, better generalization, slower
- **Large**: Stable gradients, faster, may overfit
- **Typical**: 16, 32, 64, 128, 256

#### Number of Epochs
- **From lab**: 100 epochs
- Use early stopping to prevent overtraining

#### Optimizer
- **From lab**: Adam performed best
- Alternatives: SGD, RMSprop, AdaGrad

#### Regularization Parameters
- Dropout probability: 0.2 - 0.5
- L2 weight decay: 1e-5 - 1e-3
- **From lab**: Dropout 0.3 + L2 1e-5 optimal

### 8.2 GridSearchCV

**Process:**
1. Define hyperparameter grid
2. Train model for each combination
3. Evaluate using cross-validation
4. Select best configuration

**Example Grid:**
```python
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'hidden_layers': [[64, 32], [128, 64, 32]],
    'dropout': [0.2, 0.3, 0.5],
    'optimizer': ['adam', 'sgd']
}
```

**From lab:**
- Best CV score: 90.88%
- 3-fold cross-validation used

### 8.3 Cross-Validation

**K-Fold Cross-Validation:**
1. Split data into K folds
2. Train on K-1 folds, validate on 1
3. Repeat K times
4. Average performance

**Benefits:**
- Robust performance estimate
- Uses all data for training and validation
- Reduces variance in evaluation

---

## 9. Model Evaluation Metrics

### 9.1 Regression Metrics

#### Mean Squared Error (MSE)
```
MSE = (1/n) Σ(y_true - y_pred)²
```
- Lower is better
- **From lab**: Best = 0.0010

#### Root Mean Squared Error (RMSE)
```
RMSE = √MSE
```
- Same units as target variable

#### R² Score (Coefficient of Determination)
```
R² = 1 - (SS_residual / SS_total)
  = 1 - [Σ(y_true - y_pred)² / Σ(y_true - ȳ)²]
```
- **Range**: (-∞, 1]
- **Perfect fit**: R² = 1
- **From lab**: R² = 0.9990 (excellent)

#### Mean Absolute Error (MAE)
```
MAE = (1/n) Σ|y_true - y_pred|
```
- Less sensitive to outliers than MSE

### 9.2 Classification Metrics

#### Confusion Matrix
```
                Predicted
              Pos     Neg
Actual  Pos   TP      FN
        Neg   FP      TN
```

#### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- **Misleading** with imbalanced data
- **From lab**: 98% accuracy but poor failure detection

#### Precision (Positive Predictive Value)
```
Precision = TP / (TP + FP)
```
- "Of all positive predictions, how many were correct?"
- Important when false positives are costly

#### Recall (Sensitivity, True Positive Rate)
```
Recall = TP / (TP + FN)
```
- "Of all actual positives, how many did we detect?"
- **Critical for failure detection**
- **From lab**: Improved from 0.38 to 0.63 with optimization

#### F1-Score (Harmonic Mean)
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Balances precision and recall
- **Range**: [0, 1], higher is better
- **From lab**: F1 = 0.00 for rare classes initially

#### Macro vs Micro Averaging

**Macro Average:**
```
Macro F1 = (1/n) × Σ F1_i for each class i
```
- Treats all classes equally
- **Use for**: Imbalanced datasets
- **From lab**: Macro F1 = 0.5246

**Micro Average:**
```
Micro F1 = 2 × (TP_total) / (2×TP_total + FP_total + FN_total)
```
- Weighted by class frequency
- Dominated by majority class

---

## 10. Data Augmentation

### 10.1 Why Augmentation?
- Increase training data diversity
- Prevent overfitting
- Improve model robustness
- **Critical for imbalanced datasets**

### 10.2 SMOTE (Synthetic Minority Over-sampling Technique)

**Algorithm:**
1. For each minority sample:
2. Find k nearest neighbors (typically k=5)
3. Randomly select one neighbor
4. Create synthetic sample:
```
x_new = x_original + λ × (x_neighbor - x_original)
where λ ~ Uniform(0, 1)
```

**Benefits:**
- Generates realistic synthetic samples
- Balances class distribution
- **From lab**: Critical for detecting failure classes

**Limitations:**
- May create unrealistic samples
- Doesn't work well with very high-dimensional data
- Can amplify noise

### 10.3 Other Techniques
- **Oversampling**: Duplicate minority samples (risk overfitting)
- **Undersampling**: Remove majority samples (lose information)
- **Class Weights**: Penalize misclassifying minority classes

---

## 11. Practical PyTorch Implementation

### 11.1 Basic MLP Structure
```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.3):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
```

### 11.2 Training Loop
```python
# Setup
model = MLP(input_size=10, hidden_sizes=[64, 32, 16], output_size=1)
criterion = nn.MSELoss()  # or nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training
model.train()
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights
```

### 11.3 Evaluation
```python
model.eval()  # Disable dropout, batch norm
with torch.no_grad():  # Don't compute gradients
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        # Compute metrics
```

---

## Key Takeaways from Lab 📚

### Part 1: Regression (NYSE Dataset)
- **Best Model**: Early Stopping (MSE = 0.0010, R² = 0.9990)
- **Architecture**: 64 → 32 → 16 neurons
- **Optimal Settings**: Adam optimizer, η = 0.001

### Part 2: Classification (Predictive Maintenance)
- **Challenge**: Severe class imbalance
- **Solution**: SMOTE + balanced regularization
- **Best Config**: Dropout 0.3 + L2 1e-5, F1 Macro = 0.5246
- **Key Insight**: Accuracy misleading, use macro F1 and confusion matrix

### Critical Lessons
1. **Regularization Balance**: Too little → overfitting, too much → underfitting
2. **Class Imbalance**: High accuracy doesn't mean good performance
3. **Metrics Matter**: Use appropriate metrics for your problem
4. **Hyperparameter Tuning**: Essential but requires systematic approach
5. **Early Stopping**: Simple yet most effective regularization

---

## Exam Tips ✅

1. **Understand the math**: Be able to write formulas for loss, activation, optimization
2. **Know when to use what**: Regression vs classification, different metrics, regularization
3. **Overfitting vs Underfitting**: Identify from training/test curves
4. **Hyperparameters**: Know their effects and typical ranges
5. **PyTorch syntax**: Basic model structure, training loop, evaluation mode
6. **Problem diagnosis**: Use confusion matrix and multiple metrics
7. **Data preprocessing**: Always normalize/standardize, handle imbalance

---

## Quick Reference Formulas 🔢

| Concept | Formula |
|---------|---------|
| **ReLU** | max(0, x) |
| **Sigmoid** | 1/(1 + e^(-x)) |
| **Softmax** | e^(x_i) / Σe^(x_j) |
| **MSE** | (1/n)Σ(y - ŷ)² |
| **Cross-Entropy** | -Σy×log(ŷ) |
| **Gradient Descent** | θ = θ - η∇L(θ) |
| **L2 Regularization** | L + λΣw² |
| **Dropout** | output × mask / (1-p) |
| **Accuracy** | (TP+TN) / Total |
| **Precision** | TP / (TP+FP) |
| **Recall** | TP / (TP+FN) |
| **F1-Score** | 2PR / (P+R) |
| **R² Score** | 1 - (SS_res/SS_tot) |

---

## GitHub Repository
🔗 https://github.com/chaimaebouassab4-boop/pytorch-mlp-classification-regression-lab

Good luck with your exam! 🎓