# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and prepare the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# 2. Basic EDA function
def perform_eda(df):
    print("Basic Data Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

# 3. Prepare features and target
def prepare_data(df):
    # Separate features and target
    X = df.drop('HeartDiseaseorAttack', axis=1)
    y = df['HeartDiseaseorAttack']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# 4. Train and evaluate model
def train_evaluate_model(X_train, X_test, y_train, y_test):
    # Initialize and train the model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Important Features:")
    print(feature_importance.head())
    
    return rf_model

if __name__ == "__main__":
    file_path = "HeartDisease.csv"    
    df = load_data(file_path)
    perform_eda(df)
    X_train, X_test, y_train, y_test = prepare_data(df)
    feature_names = df.drop('HeartDiseaseorAttack', axis=1).columns
    model = train_evaluate_model(X_train, X_test, y_train, y_test)