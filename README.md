# Can You Cook It?

This repository contains baseline models for the **Can You Cook It?** challenge. The goal is to create a machine learning model that predicts the difficulty of cooking recipes on a scale of **1 to 5**.

## Baseline Models
The baseline models included are simple yet effective for establishing a starting point:

1. **Average Model**: Always predicts the average value of the training labels.
2. **Linear Regression Model (BoW)**: Trained using linear regression with a **Bag-of-Words (BoW)** format for the input features.

## Model Performance
### **Linear Regression (BoW) Model**
- Mean Squared Error (MSE): `0.7867`
- R² Score: `0.2624`

### **Average Model**
- Mean Squared Error (MSE): `1.0977`
-*R² Score: `-0.0293`