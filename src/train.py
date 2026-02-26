import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from feature_engineering import FeatureEngineer

def train_model():
    processed_dir = "/Users/dmac/Desktop/ml/data/processed"
    models_dir = "/Users/dmac/Desktop/ml/models"
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(processed_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(processed_dir, 'val.csv'))
    
    print("Extracting features...")
    fe = FeatureEngineer()
    # FeatureEngineer now handles extraction internally for consistency
    fe.fit(train_df)
    X_train = fe.transform(train_df)
    y_train = train_df['label']
    
    X_val = fe.transform(val_df)
    y_val = val_df['label']
    
    print("Training Logistic Regression model (Stable Sparse)...")
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print("Evaluating on validation set...")
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred))
    
    # Save model and vectorizer
    joblib.dump(model, os.path.join(models_dir, 'model.joblib'))
    fe.save(os.path.join(models_dir, 'vectorizer.joblib'))
    print(f"Model and vectorizer saved to {models_dir}")

if __name__ == "__main__":
    train_model()
