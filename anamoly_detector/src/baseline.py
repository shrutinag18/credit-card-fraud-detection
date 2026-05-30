import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def train_isolation_forest(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Train only on normal transactions
    X_normal = X[y == 0]
    
    print("Training Isolation Forest...")
    model = IsolationForest(contamination=0.002, random_state=42, n_estimators=100)
    model.fit(X_normal)
    
    # Evaluate on full dataset
    predictions = (model.predict(X) == -1).astype(int)
    scores = -model.score_samples(X)
    
    print("\nClassification Report:")
    print(classification_report(y, predictions, target_names=['Normal', 'Fraud']))
    
    auc = roc_auc_score(y, scores)
    print(f"ROC-AUC Score: {auc:.4f}")
    
    # Save model
    joblib.dump(model, 'isolation_forest.pkl')
    print("Model saved as isolation_forest.pkl")
    
    return model, predictions, scores

if __name__ == "__main__":
    train_isolation_forest('data/creditcard_cleaned.csv')