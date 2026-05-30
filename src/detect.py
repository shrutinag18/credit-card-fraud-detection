import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

def calculate_threshold(model, X_train):
    reconstructions = model.predict(X_train)
    mae = np.mean(np.abs(reconstructions - X_train), axis=(1, 2))
    threshold = np.mean(mae) + 1 * np.std(mae)
    print(f"Threshold: {threshold:.6f}")
    return threshold

def detect_anomalies(model, X_test, y_test, threshold):
    reconstructions = model.predict(X_test)
    mae_scores = np.mean(np.abs(reconstructions - X_test), axis=(1, 2))
    predictions = (mae_scores > threshold).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Normal', 'Fraud']))

    auc = roc_auc_score(y_test, mae_scores)
    print(f"ROC-AUC Score: {auc:.4f}")

    return predictions, mae_scores

if __name__ == "__main__":
    from keras.models import load_model
    from src.preprocess import load_and_preprocess

    # Load data
    X_train, X_test, y_test, scaler = load_and_preprocess('data/creditcard_cleaned.csv')

    # Load trained model
    model = load_model('model.keras')

    # Calculate threshold from training data
    threshold = calculate_threshold(model, X_train)

    # Detect anomalies
    predictions, mae_scores = detect_anomalies(model, X_test, y_test, threshold)