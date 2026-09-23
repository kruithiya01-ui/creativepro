import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(filepath='dataset/student_data.csv'):
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: Dataset not found at {filepath}")
        return None

def preprocess_data(df):
    """Clean and preprocess data for training."""
    # Drop missing values if any exist
    df = df.dropna()
    
    # Feature columns
    feature_cols = [
        'Previous_Score', 
        'Study_Time_Hours', 
        'Num_Revisions', 
        'Time_Gap_Days', 
        'Quiz_Score'
    ]
    
    # Target column
    target_col = 'Knowledge_Retention_Score'
    
    X = df[feature_cols]
    y = df[target_col]
    
    return X, y

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """Split into train/test sets and apply standard scaling."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for inference later
    joblib.dump(scaler, 'scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def preprocess_input(input_dict, scaler_path='scaler.pkl'):
    """Preprocess a single input for prediction based on the saved scaler."""
    # Convert input dict to DataFrame to maintain column names if needed, 
    # but scaler expects 2D array of same shape
    features = [
        input_dict.get('Previous_Score', 0),
        input_dict.get('Study_Time_Hours', 0),
        input_dict.get('Num_Revisions', 0),
        input_dict.get('Time_Gap_Days', 0),
        input_dict.get('Quiz_Score', 0)
    ]
    
    X = np.array(features).reshape(1, -1)
    
    try:
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
        return X_scaled
    except FileNotFoundError:
        print("Scaler not found. Please train the model first.")
        # Fallback to unscaled if scaler not found (not ideal)
        return X
