import numpy as np
from keras.callbacks import EarlyStopping
from src.preprocess import load_and_preprocess
from src.model import build_autoencoder

def train(data_path, seq_len=30, epochs=20, batch_size=32):
    # Load and preprocess data
    X_train, X_test, y_test, scaler = load_and_preprocess(data_path, seq_len)

    # Get number of features
    n_features = X_train.shape[2]

    # Build model
    model = build_autoencoder(seq_len, n_features)

    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train
    print("Training model...")
    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        shuffle=True
    )

    # Save model
    model.save('model.keras')
    print("Model saved as model.keras")

    return model, X_test, y_test, history

if __name__ == "__main__":
    train('data/creditcard_cleaned.csv')