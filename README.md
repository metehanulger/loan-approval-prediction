# 🏦 Loan Approval Prediction — Logistic Regression

A machine learning model that predicts whether a loan application will be approved or rejected, built with scikit-learn and trained on real-world loan application data.

---

## 📌 Overview

This project applies a **Logistic Regression** classifier to predict loan approval status based on applicant information such as income, loan amount, credit history, and demographic details.

The pipeline covers end-to-end ML workflow:
- Data cleaning & missing value handling
- Feature encoding & scaling
- Train/test split with leak-free preprocessing
- Model training, evaluation, and export

---

## 📂 Project Structure

```
loan-approval-prediction/
│
├── loan_prediction.py     # Main training script
├── loan-train.csv         # Dataset (not included — see below)
├── loan_model.pkl         # Trained model (generated after running)
├── scaler.pkl             # Fitted scaler (generated after running)
└── README.md
```

---

## 📊 Dataset

The dataset used is the [Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) from Kaggle.

**Features used:**

| Feature | Description |
|---|---|
| `Gender` | Male / Female |
| `Married` | Applicant marital status |
| `Dependents` | Number of dependents |
| `Education` | Graduate / Not Graduate |
| `Self_Employed` | Self-employed or not |
| `ApplicantIncome` | Applicant's monthly income |
| `CoapplicantIncome` | Co-applicant's monthly income |
| `LoanAmount` | Loan amount requested |
| `Loan_Amount_Term` | Term of loan in months |
| `Credit_History` | Credit history meets guidelines (1/0) |
| `Property_Area` | Urban / Semi-Urban / Rural |

**Target:** `Loan_Status` — Y (approved) or N (rejected)

---

## ⚙️ How It Works

1. **Missing values** are filled using median (for numeric columns) and mode (for categorical columns)
2. **Categorical features** are one-hot encoded with `pd.get_dummies()`
3. **Data is split** into 80% train / 20% test before any scaling is applied
4. **Numerical features** are standardized using `StandardScaler` fitted only on training data (no data leakage)
5. **Logistic Regression** is trained and evaluated on the test set
6. Both the **model** and **scaler** are saved with `joblib` for future use

---

## 🚀 Getting Started

### Requirements

```bash
pip install pandas scikit-learn joblib
```

### Run

```bash
python loan_prediction.py
```

### Output

```
--- Model Performansı ---
Doğruluk Skoru (Accuracy): %XX.XX

Detaylı Rapor:
              precision    recall  f1-score   support
           0       ...
           1       ...
```

After running, `loan_model.pkl` and `scaler.pkl` will be saved in the project directory.

---

## 🛠️ Tech Stack

- Python 3.x
- pandas
- scikit-learn
- joblib

---

## 👤 Author

**Metehan Ülger**  
Software Engineering Student @ Ankara University  
[GitHub](https://github.com/metehanulger)
