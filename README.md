# Can You Cook It?
This repository contains baseline models for the **Can You Cook It?** challenge. The goal is to create a machine learning model that predicts the difficulty of cooking recipes on a scale of **1 to 5**.

## Baseline Models
The baseline models included are simple yet effective for establishing a starting point:

1. **Average Model**: Always predicts the average value of the training labels.
2. **Random Model**: Always predicts the a random value between 1 and 5.
3. **Linear Regression Model (BoW)**: Trained using linear regression with a **Bag-of-Words (BoW)** format for the input features.

## Model Performance
### **Linear Regression (BoW) Model**
- Mean Squared Error (MSE): `0.6587`
- R² Score: `0.4040`

### **Average Model**
- Mean Squared Error (MSE): `1.0977`
-*R² Score: `-0.0293`

### **Random Model**
- Mean Squared Error (MSE): `3.1964`
-*R² Score: `-2.2340`