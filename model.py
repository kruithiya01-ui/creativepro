import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

from data_processing import load_data, preprocess_data, split_and_scale_data

def train_model(X_train, y_train):
    """Train Random Forest model."""
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model and save performance metrics."""
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Evaluation -> RMSE: {rmse:.2f}, R-squared: {r2:.2f}")
    return rmse, r2, predictions

def generate_visualizations(model, feature_names, y_test, predictions):
    """Generate and save visualizations using Matplotlib & Seaborn."""
    os.makedirs('static/images', exist_ok=True)
    
    # 1. Feature Importance Plot
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    
    sns.barplot(x=importances[indices], y=sorted_features, hue=sorted_features, palette='viridis', legend=False)
    plt.title('Feature Importance for Knowledge Retention')
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('static/images/feature_importance.png')
    plt.close()
    
    # 2. Actual vs Predicted Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Actual vs Predicted Retention Score')
    plt.xlabel('Actual Retention Score')
    plt.ylabel('Predicted Retention Score')
    plt.tight_layout()
    plt.savefig('static/images/actual_vs_pred.png')
    plt.close()

def plot_distributions(df):
    """Generate generic distribution plots of the dataset."""
    os.makedirs('static/images', exist_ok=True)
    
    # Study Time vs Retention Score
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Study_Time_Hours', y='Knowledge_Retention_Score', hue='Risk_Level', palette={'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'})
    plt.title('Study Time vs Knowledge Retention')
    plt.tight_layout()
    plt.savefig('static/images/study_vs_retention.png')
    plt.close()
    
    # Retention Score Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Knowledge_Retention_Score'], bins=30, kde=True, color='purple')
    plt.title('Distribution of Knowledge Retention Scores')
    plt.xlabel('Retention Score')
    plt.tight_layout()
    plt.savefig('static/images/retention_dist.png')
    plt.close()

def get_risk_level_and_suggestions(score):
    """Determine risk level and return actionable suggestions."""
    if score >= 75:
        risk = "Low Risk"
        suggestion = "Excellent retention! Maintain your current study habits and revision schedule."
    elif score >= 50:
        risk = "Medium Risk"
        suggestion = "Fair retention. Consider increasing your study time or reducing the gap between revisions."
    else:
        risk = "High Risk"
        suggestion = "Critical retention level. You should urgently increase revisions and dedicate more hours to studying."
    return risk, suggestion

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        X, y = preprocess_data(df)
        feature_names = X.columns.tolist()
        
        # Split and scale
        X_train, X_test, y_train, y_test = split_and_scale_data(X, y)
        
        # Train
        model = train_model(X_train, y_train)
        
        # Evaluate
        rmse, r2, predictions = evaluate_model(model, X_test, y_test)
        
        # Generate Visualizations
        generate_visualizations(model, feature_names, y_test, predictions)
        plot_distributions(df)
        
        # Save Model
        joblib.dump(model, 'rf_model.pkl')
        print("Model and visualizations successfully created and saved.")
