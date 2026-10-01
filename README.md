# Iris Flower Classification Using KNN

## DecodeLabs Artificial Intelligence — Project 2

A beginner-friendly supervised machine learning project that classifies Iris flowers into three species using the **K-Nearest Neighbors (KNN)** algorithm.

This project follows the Data Classification Using AI workflow described in the DecodeLabs Industrial Training Kit: load and understand the Iris dataset, split the data into training and testing sets, scale the features, train a KNN classifier, make predictions, and evaluate the model.

---

## 📌 Project Overview

The objective of this project is to build a basic **classification model** using a small, labeled dataset.

The project demonstrates the fundamental supervised learning pipeline:

```text
Iris Dataset
     ↓
Understand Data
     ↓
Train/Test Split
     ↓
Feature Scaling
     ↓
KNN Model
     ↓
Model Training
     ↓
Prediction
     ↓
Model Evaluation
```

The model learns from known examples of Iris flowers and predicts the species of new, unseen flowers.

---

## 🎯 Objectives

- Load and understand the Iris dataset
- Separate features and target labels
- Split the data into training and testing sets
- Standardize the numerical features
- Build a K-Nearest Neighbors classification model
- Train the model using training data
- Predict the classes of test data
- Evaluate the model using classification metrics

---

## 🧠 What is Supervised Learning?

Supervised learning is a type of machine learning where a model learns from examples that already contain the correct answers.

In this project:

```text
Flower measurements → Correct flower species
```

The model learns the relationship between the measurements and the known species.

After training, it receives new measurements and predicts the corresponding species.

---

## 🌸 Dataset

This project uses the **Iris dataset** provided through Scikit-learn.

According to the project specification, the dataset contains:

| Property | Value |
|---|---:|
| Samples | 150 |
| Features | 4 |
| Classes | 3 |

### Features

Each flower is described using four measurements:

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

### Classes

The model classifies flowers into:

- Setosa
- Versicolor
- Virginica

---

## ⚙️ Machine Learning Workflow

### 1. Load the Dataset

The Iris dataset is loaded using Scikit-learn:

```python
from sklearn.datasets import load_iris

iris = load_iris()
```

The feature data is stored in `X` and the target labels are stored in `y`.

```python
X = iris.data
y = iris.target
```

---

### 2. Train/Test Split

The dataset is divided into:

```text
80% → Training data
20% → Testing data
```

The training data is used to teach the model.

The testing data is kept separate and used to check how well the model performs on unseen examples.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

### 3. Feature Scaling

The project uses `StandardScaler` to standardize the numerical features.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

The scaler learns the scaling information from the training data and applies the same transformation to the test data.

---

### 4. K-Nearest Neighbors

The classification algorithm used is **K-Nearest Neighbors (KNN)**.

The implementation uses:

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
```

Here:

```text
K = 5
```

means that the model considers the five nearest training examples when making a classification decision.

The general idea is:

```text
New flower
    ↓
Find nearby training examples
    ↓
Look at their classes
    ↓
Majority vote
    ↓
Predicted flower species
```

---

### 5. Model Training

The model is trained using:

```python
model.fit(X_train, y_train)
```

`fit()` means that the model learns from the training data.

---

### 6. Prediction

After training, the model predicts the classes of the test data:

```python
predictions = model.predict(X_test)
```

---

### 7. Model Evaluation

The project evaluates the predictions using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1 Score

Example evaluation code:

```python
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

accuracy = accuracy_score(y_test, predictions)

cm = confusion_matrix(y_test, predictions)

print(classification_report(
    y_test,
    predictions,
    target_names=iris.target_names
))
```

---

## 📊 Evaluation Metrics

### Accuracy

Accuracy represents the proportion of predictions that were correct.

```text
Correct predictions
-------------------
Total predictions
```

### Confusion Matrix

A confusion matrix shows how the actual classes compare with the predicted classes.

For this three-class Iris problem, the matrix represents:

```text
Setosa
Versicolor
Virginica
```

and shows where predictions were correct or incorrect.

### Precision

Precision indicates how reliable predictions for a class are.

### Recall

Recall indicates how many of the actual examples of a class were correctly identified.

### F1 Score

F1 Score combines precision and recall into a single measure.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Scikit-learn | Machine learning library |
| StandardScaler | Feature scaling |
| KNeighborsClassifier | KNN classification |
| Iris Dataset | Training/testing dataset |

The project specification does not require a web frontend, backend API, database, authentication system, or cloud deployment.

---

## 📁 Project Structure

```text
AI-Iris-Classification/
│
├── venv/
│   └── Virtual environment
│
├── main.py
│   └── Complete machine learning implementation
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
    └── Project documentation
```

> `venv/` is a local virtual environment and should normally not be committed to GitHub. Add it to `.gitignore` before pushing the project.

---

## 🚀 Installation and Setup

### Prerequisites

Make sure Python is installed.

Check the version:

```powershell
python --version
```

---

### 1. Clone or download the project

Open the project folder in VS Code.

---

### 2. Create a virtual environment

```powershell
python -m venv venv
```

---

### 3. Activate the virtual environment

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

After activation, the terminal should show:

```text
(venv)
```

---

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

If `requirements.txt` has not been created yet, install Scikit-learn directly:

```powershell
pip install scikit-learn
```

Then save the installed dependencies:

```powershell
pip freeze > requirements.txt
```

---

## ▶️ Run the Project

Make sure the virtual environment is active:

```text
(venv)
```

Then run:

```powershell
python main.py
```

The program will:

1. Load the Iris dataset
2. Display dataset information
3. Split the data
4. Scale the features
5. Create the KNN model
6. Train the model
7. Generate predictions
8. Calculate accuracy
9. Display the confusion matrix
10. Display precision, recall, and F1 score

---

## 📌 Expected Output

The exact values depend on the train/test split and implementation.

The terminal should contain output similar to:

```text
Dataset loaded successfully!

--- Dataset Information ---
Number of samples: 150
Number of features: 4

--- Data Split ---
Training samples: 120
Testing samples: 30

Feature scaling completed!

KNN model created!
Number of neighbors (K): 5

Model training completed!
Predictions completed!

--- Model Accuracy ---
Accuracy: ...

--- Confusion Matrix ---
...

--- Classification Report ---
...
```

---

## 🔄 Complete Application Pipeline

```text
                 ┌─────────────────┐
                 │   Iris Dataset  │
                 │ 150 samples     │
                 │ 4 features      │
                 │ 3 classes       │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Data Inspection │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ 80/20 Split     │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ StandardScaler  │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ KNN Classifier   │
                 │ K = 5            │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Model Training  │
                 │ model.fit()     │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Prediction      │
                 │ model.predict() │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Evaluation      │
                 │ Accuracy        │
                 │ Confusion Matrix│
                 │ Precision       │
                 │ Recall          │
                 │ F1 Score        │
                 └─────────────────┘
```

---

## 📚 Key Concepts Learned

This project provides practical exposure to:

- Artificial Intelligence
- Machine Learning
- Supervised Learning
- Classification
- Dataset
- Features
- Labels
- Training Data
- Testing Data
- Train/Test Split
- Feature Scaling
- StandardScaler
- K-Nearest Neighbors
- Model Training
- Prediction
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score

---

## ⚠️ Project Scope

This is a **basic machine-learning classification project** intended to demonstrate the fundamental supervised-learning pipeline.

The project specification does not define:

- Web application
- REST API
- Database
- User authentication
- Cloud deployment
- Production monitoring
- Docker/containerization

These are therefore outside the required scope of this implementation.

---

## 🎓 Learning Outcome

After completing this project, the learner should understand the basic process of taking labeled data and building a classification model:

```text
Data
 ↓
Preparation
 ↓
Training
 ↓
Prediction
 ↓
Evaluation
```

The project provides a foundation for moving toward more advanced AI topics, including the deep-learning and CNN direction mentioned in the training material.

---

## 📄 Project Reference

**Program:** Industrial Training Kit  
**Domain:** Artificial Intelligence  
**Project:** Project 2 — Data Classification Using AI  
**Organization:** DecodeLabs  
**Batch:** 2026

---

## 👤 Author

**P. Nanda Kishore**

B.Tech Student | Artificial Intelligence & Machine Learning Enthusiast

---

## 📜 License

This project was created as part of an educational/internship training project.
