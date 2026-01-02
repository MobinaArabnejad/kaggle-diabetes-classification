# Kaggle Diabetes Classification (Random Forest + GridSearchCV)

Binary classification project using the Kaggle diabetes dataset (commonly known as the Pima Indians Diabetes dataset).
The goal is to predict diabetes outcome from clinical measurements.

> Disclaimer: This project is for learning/demo purposes only. It is not medical advice and must not be used for clinical decisions.

# What this project does
- Loads the dataset from CSV
- Performs basic data inspection (shape, missing values)
- Treats zero values as invalid/missing for:
- Glucose, BloodPressure, SkinThickness, Insulin, BMI
- Replaces invalid zeros with NaN and imputes missing values using median
- Splits data into train/test sets with stratification
- Trains a RandomForestClassifier inside a scikit-learn Pipeline
- Uses GridSearchCV (5-fold) with ROC-AUC scoring to tune hyperparameters
- Evaluates performance on the test set:
- Accuracy
- Confusion Matrix
- Classification Report
- ROC Curve + AUC

# Tech Stack
- Python
- pandas, numpy
- scikit-learn
- matplotlib

# How to run
1. Clone:
```bash
git clone https://github.com/Mobina Arabnejad/kaggle-diabetes-classification.git
cd kaggle-diabetes-classification
