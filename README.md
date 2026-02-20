# Credit Score Classification using Random Forest

## 📌 Project Overview

This project builds a **Credit Score Classification Model** using the Random Forest algorithm.  
The objective is to predict a customer's credit score category based on financial and behavioral features.

Hyperparameter tuning is performed using **GridSearchCV** to improve model performance.

---

## 📂 Dataset Description

The dataset contains customer financial information such as:

- Age  
- Occupation  
- Annual Income  
- Monthly In-hand Salary  
- Number of Bank Accounts  
- Number of Credit Cards  
- Outstanding Debt  
- EMI per Month  
- Payment Behaviour  
- Credit Mix  

### 🎯 Target Variable:
- `Credit_Score` (3 classes)

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Removed unwanted characters (`_`) from object columns.
2. Converted numeric-like string columns into proper numeric format.
3. Dropped high-cardinality column:
   - `Type_of_Loan`
4. Applied One-Hot Encoding to categorical features:
   - `Month`
   - `Occupation`
   - `Credit_Mix`
   - `Payment_Behaviour`

---

## 🤖 Model Used

### Random Forest Classifier

Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines their outputs to improve accuracy and reduce overfitting.

---

## ⚙️ Hyperparameter Tuning

GridSearchCV was used with the following parameter grid:

- `n_estimators`: [100, 200]
- `max_depth`: [10, 20]
- `min_samples_split`: [2, 7]

The best estimator selected by GridSearchCV was used for final predictions.

---

## 📊 Train-Test Split

- 80% Training Data
- 20% Testing Data
- Random State: 42

---

## 📈 Model Evaluation

The model was evaluated using:

- Classification Report
  - Precision
  - Recall
  - F1-Score

### 🔍 Results

Accuracy: 76%
