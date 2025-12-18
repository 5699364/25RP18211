# -*- coding: utf-8 -*-
"""
Created on Dec 2025
Project: Diabetes Prediction using Gradient Boosting
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# --- 1. Load Data ---
try:
    df = pd.read_csv('diabetes.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Dataset not found. Creating dummy data for demonstration...")
    data = {
        'Pregnancies': [6, 1, 8, 1, 0, 5],
        'Glucose': [148, 85, 183, 89, 137, 116],
        'BloodPressure': [72, 66, 64, 66, 40, 74],
        'SkinThickness': [35, 29, 0, 23, 35, 0],
        'Insulin': [0, 0, 0, 94, 168, 0],
        'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6],
        'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288, 0.201],
        'Age': [50, 31, 32, 21, 33, 30],
        'Outcome': [1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)

# --- 2. Data Cleaning / Preprocessing ---
# Replace invalid 0s with median
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, df[col].median())

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. Initial Model Training ---
print("\n--- Training Initial Gradient Boosting Classifier ---")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = gb_model.predict(X_test_scaled)
print(f"Initial Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# --- 4. Hyperparameter Tuning ---
print("\n--- Starting Grid Search (Tuning) ---")
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# --- 5. Final Evaluation ---
best_y_pred = best_model.predict(X_test_scaled)
print("\n--- Final Model Evaluation ---")
print(f"Tuned Accuracy Score: {accuracy_score(y_test, best_y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, best_y_pred))

# --- 6. Save Model and Scaler ---
# Using joblib (more efficient for large NumPy arrays/sklearn models)
try:
    joblib.dump(best_model, "GROUP_05_model.joblib")
    joblib.dump(scaler, "GROUP_05_scaler.joblib")
    print("\n✅ Model and Scaler saved successfully as .joblib files.")
    
    # Also saving as pickle per notebook requirements
    with open('GROUP_05_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("✅ Model also saved as GROUP_05_model.pkl.")
except Exception as e:
    print(f"Error saving files: {e}")