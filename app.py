import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide")
st.title("💳 Credit Card Fraud Detector")
st.markdown("Detect fraudulent credit card transactions using Machine Learning")
st.divider()

model_choice = st.selectbox("Choose Model", ["Isolation Forest (Recommended)", "LSTM Autoencoder"])
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    with st.expander("Preview Data"):
        st.dataframe(df.head())

    features = df.drop("Class", axis=1) if "Class" in df.columns else df.copy()
    true_labels = df["Class"].values if "Class" in df.columns else None

    if model_choice == "Isolation Forest (Recommended)":
        if not os.path.exists("isolation_forest.pkl"):
            st.warning("Model not found. Run src/baseline.py first.")
        else:
            import joblib
            with st.spinner("Detecting fraud..."):
                model = joblib.load("isolation_forest.pkl")
                predictions = (model.predict(features) == -1).astype(int)
                scores = -model.score_samples(features)

            st.subheader("Detection Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Transactions", len(predictions))
            c2.metric("Flagged as Fraud", int(predictions.sum()))
            c3.metric("Fraud Rate", f"{predictions.mean()*100:.2f}%")

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Anomaly Score Distribution")
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(scores, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
                ax.axvline(np.percentile(scores, 99), color="red", linestyle="--", label="99th percentile")
                ax.set_xlabel("Anomaly Score")
                ax.set_ylabel("Count")
                ax.legend()
                st.pyplot(fig)
                plt.close()

            with col2:
                st.subheader("Fraud vs Normal")
                fig, ax = plt.subplots(figsize=(6, 3))
                labels = ["Normal", "Fraud"]
                values = [(predictions == 0).sum(), (predictions == 1).sum()]
                colors = ["steelblue", "crimson"]
                ax.bar(labels, values, color=colors, edgecolor="black")
                ax.set_ylabel("Count")
                for i, v in enumerate(values):
                    ax.text(i, v + 10, str(v), ha="center", fontweight="bold")
                st.pyplot(fig)
                plt.close()

            st.divider()
            st.subheader("Transaction Results Table")
            result_df = df.copy()
            result_df["anomaly_score"] = scores
            result_df["is_fraud"] = predictions.astype(bool)
            st.dataframe(result_df[["Amount", "anomaly_score", "is_fraud"]].head(50))

            if true_labels is not None:
                from sklearn.metrics import roc_auc_score, classification_report
                auc = roc_auc_score(true_labels, scores)
                st.divider()
                st.subheader("Model Performance")
                st.success(f"ROC-AUC Score: {auc:.4f}")
                st.code(classification_report(true_labels, predictions, target_names=["Normal", "Fraud"]))

    else:
        if not os.path.exists("model.keras"):
            st.warning("Model not found. Run src/train.py first.")
        else:
            from keras.models import load_model
            from sklearn.preprocessing import StandardScaler

            with st.spinner("Running LSTM detection..."):
                scaler = StandardScaler()
                scaled = scaler.fit_transform(features)
                seq_len = 30
                scaled_sample = scaled[:5000]
                sequences = np.array([scaled_sample[i:i+seq_len] for i in range(len(scaled_sample)-seq_len)])
                model = load_model("model.keras")
                X_pred = model.predict(sequences, verbose=0)
                mae = np.mean(np.abs(X_pred - sequences), axis=(1, 2))
                threshold = np.mean(mae) + np.std(mae)
                predictions = (mae > threshold).astype(int)

            st.subheader("Detection Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Sequences Analysed", len(predictions))
            c2.metric("Flagged as Fraud", int(predictions.sum()))
            c3.metric("Fraud Rate", f"{predictions.mean()*100:.2f}%")

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Reconstruction Error Distribution")
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(mae, bins=50, color="darkorange", edgecolor="black", alpha=0.7)
                ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold: {threshold:.3f}")
                ax.set_xlabel("Reconstruction Error")
                ax.set_ylabel("Count")
                ax.legend()
                st.pyplot(fig)
                plt.close()

            with col2:
                st.subheader("Fraud vs Normal")
                fig, ax = plt.subplots(figsize=(6, 3))
                labels = ["Normal", "Fraud"]
                values = [(predictions == 0).sum(), (predictions == 1).sum()]
                colors = ["steelblue", "crimson"]
                ax.bar(labels, values, color=colors, edgecolor="black")
                ax.set_ylabel("Count")
                for i, v in enumerate(values):
                    ax.text(i, v + 10, str(v), ha="center", fontweight="bold")
                st.pyplot(fig)
                plt.close()

            st.divider()
            st.subheader("Transaction Results Table")
            result_df = df.iloc[seq_len:seq_len+len(predictions)].copy()
            result_df["anomaly_score"] = mae
            result_df["is_fraud"] = predictions.astype(bool)
            st.dataframe(result_df[["Amount", "anomaly_score", "is_fraud"]].head(50))

            if true_labels is not None:
                from sklearn.metrics import roc_auc_score, classification_report
                full_mae = mae
                auc = roc_auc_score(true_labels[seq_len:seq_len+len(full_mae)], full_mae)
                st.divider()
                st.subheader("Model Performance")
                st.success(f"ROC-AUC Score: {auc:.4f}")
                st.code(classification_report(true_labels[seq_len:seq_len+len(predictions)], predictions, target_names=["Normal", "Fraud"]))