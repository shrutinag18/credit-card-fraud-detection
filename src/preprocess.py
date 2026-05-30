import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath, seq_len=30):
    print("Loading cleaned data...")
    df = pd.read_csv(filepath)

    print(f"Dataset shape: {df.shape}")
    print(f"Normal transactions: {(df['Class']==0).sum()}")
    print(f"Fraud transactions: {(df['Class']==1).sum()}")

    # Separate normal transactions only for training
    normal = df[df['Class'] == 0].drop('Class', axis=1)
    full = df.drop('Class', axis=1)
    labels = df['Class'].values

    # Scale all features
    scaler = StandardScaler()
    normal_scaled = scaler.fit_transform(normal)
    full_scaled = scaler.transform(full)

    # Create sequences
    def make_sequences(data, seq_len):
        sequences = []
        for i in range(len(data) - seq_len):
            sequences.append(data[i:i+seq_len])
        return np.array(sequences)

    print("Creating sequences...")
    X_train = make_sequences(normal_scaled, seq_len)
    X_test = make_sequences(full_scaled, seq_len)
    y_test = labels[seq_len:]

    print(f"Training sequences: {X_train.shape}")
    print(f"Test sequences: {X_test.shape}")

    return X_train, X_test, y_test, scaler