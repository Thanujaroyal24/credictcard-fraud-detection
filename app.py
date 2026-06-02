THANUJA TIRUMALASETTY

​
THANUJA TIRUMALASETTY​
# ============================================
# 💳 Optimized Credit Card Fraud Detection with Random Forest
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Streamlit Config ---
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection: Predict 0 (Valid) or 1 (Fraud)")

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 📘 About this App")
    st.write("""
    Predict whether a credit card transaction is **Fraud (1)** or **Valid (0)** using **Random Forest**.
    
    🔹 Upload training & test datasets  
    🔹 Automatic preprocessing (fill missing, encode categorical)  
    🔹 Train Random Forest classifier  
    🔹 Show validation metrics, confusion matrix & feature importance  
    🔹 Predict 0/1 on test dataset and download CSV  
    """)

# --- Upload Training Data ---
st.header("1️⃣ Upload Training Dataset")
train_file = st.file_uploader("Upload training CSV", type=["csv"], key="train")

if train_file is not None:
    train_df = pd.read_csv(train_file)
    st.success(f"✅ Training data uploaded. Rows: {len(train_df)}")
    st.subheader("📄 Full Training Dataset")
    st.dataframe(train_df, use_container_width=True)

    if "Class" not in train_df.columns:
        st.error("❌ Training data must include 'Class' column!")
    else:
        if st.button("🚀 Train Random Forest Classifier"):
            with st.spinner("Training Random Forest... This may take a few seconds."):
                # ---- Optional: Sample large data for faster training ----
                if len(train_df) > 50000:
                    train_df = train_df.sample(50000, random_state=42)

                # ---- Preprocessing ----
                df = train_df.copy()
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].fillna('Unknown')
                        df[col] = LabelEncoder().fit_transform(df[col])
                    else:
                        df[col] = df[col].fillna(0)

                X = df.drop(['Class'], axis=1)
                y = df['Class']

                # ---- Balance Data ----
                ros = RandomOverSampler(random_state=42)
                X_res, y_res = ros.fit_resample(X, y)

                # ---- Train/Validation Split ----
                X_train, X_val, y_train, y_val = train_test_split(
                    X_res, y_res, test_size=0.3, stratify=y_res, random_state=42
                )

                # ---- Random Forest ----
                rf = RandomForestClassifier(
                    n_estimators=100, # reduced for faster training
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                rf.fit(X_train, y_train)
                val_preds = rf.predict(X_val)
                val_acc = accuracy_score(y_val, val_preds)
                val_report = classification_report(y_val, val_preds, output_dict=True)

                # ---- Save in Session ----
                st.session_state['rf'] = rf
                st.session_state['X_columns'] = X.columns
                st.session_state['train_df'] = train_df

            st.success("✅ Random Forest trained successfully!")

            # ---- Validation Metrics ----
            st.subheader("📈 Model Performance Metrics (Validation)")
            st.markdown(f"**Random Forest Accuracy:** {val_acc:.4f}")

            # ---- Classification Report ----
            st.subheader("📑 Classification Report (Random Forest)")
            report_df = pd.DataFrame(val_report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap="Blues"), use_container_width=True)

            # ---- Confusion Matrix ----
            st.subheader("🔲 Confusion Matrix (Random Forest)")
            fig_cm, ax_cm = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_val, val_preds, ax=ax_cm, cmap='Blues')
            st.pyplot(fig_cm)

            # ---- Feature Importance ----
            st.subheader("⭐ Feature Importance (Random Forest)")
            fig_fi, ax_fi = plt.subplots(figsize=(8,5))
            sns.barplot(x=rf.feature_importances_, y=X.columns, ax=ax_fi)
            ax_fi.set_title("Random Forest Feature Importance")
            st.pyplot(fig_fi)

            # ---- EDA Plots ----
            st.subheader("📊 Exploratory Data Analysis (EDA)")
            # Class Distribution
            fig1, ax1 = plt.subplots()
            sns.countplot(x='Class', data=train_df, ax=ax1)
            ax1.set_title("Transaction Class Distribution (0=Valid, 1=Fraud)")
            st.pyplot(fig1)

            # Feature Correlation Heatmap
            fig2, ax2 = plt.subplots(figsize=(10,10))
            sns.heatmap(train_df.corr(), vmax=0.8, square=True, cmap="coolwarm", ax=ax2)
            ax2.set_title("Feature Correlation Heatmap")
            st.pyplot(fig2)

# --- Upload Test Data and Predict 0/1 ---
if 'rf' in st.session_state:
    st.header("2️⃣ Upload Test Dataset for Prediction")
    test_file = st.file_uploader("Upload test CSV", type=["csv"], key="test")

    if test_file is not None:
        test_df = pd.read_csv(test_file)
        st.success(f"✅ Test data uploaded. Rows: {len(test_df)}")
        st.subheader("📄 Full Test Dataset (Before Prediction)")
        st.dataframe(test_df, use_container_width=True)

        if st.button("🔍 Predict Fraud/Valid"):
            try:
                df_test = test_df.copy()
                for col in df_test.columns:
                    if df_test[col].dtype == 'object':
                        df_test[col] = df_test[col].fillna('Unknown')
                        df_test[col] = LabelEncoder().fit_transform(df_test[col])
                    else:
                        df_test[col] = df_test[col].fillna(0)

                # Align columns
                df_test = df_test.reindex(columns=st.session_state['X_columns'], fill_value=0)

                # Predict
                preds = st.session_state['rf'].predict(df_test)
                df_test['RF_Prediction'] = preds

                st.success("✅ Predictions completed!")
                st.subheader("📄 Test Dataset with 0/1 Predictions")
                st.dataframe(df_test, use_container_width=True)

                # Download predictions
                csv = df_test.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name="creditcard_rf_predictions.csv",
                    mime='text/csv'
                )

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
