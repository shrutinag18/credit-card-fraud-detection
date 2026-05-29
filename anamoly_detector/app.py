import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

st.title("💳 Credit Card Fraud Detector")
st.write("Upload a CSV file to detect anomalies")

# Model selection
model_choice = st.selectbox("Choose Model", ["Isolation Forest (Recommended)", "LSTM Autoencoder"])

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    features = df.drop('Class', axis=1) if 'Class' in df.columns else df
    true_labels = df['Class'].values if 'Class' in df.columns else None

    if model_choice == "Isolation Forest (Recommended)":
        if os.path.exists("isolation_forest.pkl"):
            import joblib
            model = joblib.load("isolation_forest.pkl")
            
            predictions = (model.predict(features) == -1).astype(int)
            scores = -model.score_samples(features)

            st.subheader("Results")
            st.write(f"Total transactions: {len(predictions)}")
            st.write(f"Flagged as fraud: {predictions.sum()}")

            result_df = df.copy()
            result_df['anomaly_score'] = scores
            result_df['is_fraud'] = predictions
            
            st.dataframe(result_df[['Amount', 'anomaly_score', 'is_fraud']].head(50))

            if true_labels is not None:
                from sklearn.metrics import classification_report, roc_auc_score
                st.subheader("Model Performance")
                auc = roc_auc_score(true_labels, scores)
                st.write(f"ROC-AUC Score: {auc:.4f}")
        else:
            st.warning("Isolation Forest model not found. Run src/baseline.py first.")

    else:
        if os.path.exists("model.keras"):
            from keras.models import load_model

            scaler = StandardScaler()
            scaled = scaler.fit_transform(features)

            seq_len = 30
            sequences = []
            for i in range(len(scaled) - seq_len):
                sequences.append(scaled[i:i+seq_len])
            X = np.array(sequences)

            model = load_model("model.keras")
            X_pred = model.predict(X)
            mae = np.mean(np.abs(X_pred - X), axis=(1, 2))
            threshold = np.mean(mae) + np.std(mae)
            predictions = (mae > threshold).astype(int)

            st.subheader("Results")
            st.write(f"Total transactions analysed: {len(predictions)}")
            st.write(f"Flagged as fraud: {predictions.sum()}")

            result_df = df.iloc[seq_len:].copy()
            result_df['anomaly_score'] = mae
            result_df['is_fraud'] = predictions
            st.dataframe(result_df[['Amount', 'anomaly_score', 'is_fraud']].head(50))
        else:
            st.warning("LSTM model not found. Run src/train.py first.")