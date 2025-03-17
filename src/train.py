from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import numpy as np
import joblib

# Simple baseline model: Always predicts the average of the traininig labels
class AverageModel:
    def __init__(self):
        self.average = None
        
    def fit(self, X, y):
        self.average = np.mean(y)
        
    def predict(self, X):
        predictions = []
        for _ in range(X.shape[0]):
            predictions.append(self.average)
        return np.array(predictions)

# Trained model: Simple linear regression using BoW features
def train_logistic_regression(X_train, y_train, max_iter=1000, C=1.0):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "mean_squared_error": mean_squared_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred),
    }

    return metrics