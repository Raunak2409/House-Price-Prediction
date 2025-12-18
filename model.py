import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def train_model():
    print("Loading California Housing dataset...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    # Features and Target
    X = df[data.feature_names]
    y = data.target # Median house value in 100k
    
    print(f"Dataset shape: {df.shape}")
    print("Features:", data.feature_names)

    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize model
    print("Training Random Forest Regressor (this may take a moment)...")
    # Using specific parameters for a balance of speed and performance
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"MAE: {mae:.4f} (Avg error in 100k units)")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Save
    model_filename = "house_price_model.pkl"
    joblib.dump(model, model_filename)
    print(f"\nModel saved to {model_filename}")

    # Save feature names to ensure order during inference
    joblib.dump(data.feature_names, "feature_names.pkl")
    print("Feature names saved to feature_names.pkl")

if __name__ == "__main__":
    train_model()
