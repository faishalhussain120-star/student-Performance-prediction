import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("🎓 Student Performance Prediction (Dataset Version)")

# Upload CSV file
uploaded_file = st.file_uploader("Upload Student Dataset (CSV file)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.subheader("📄 Uploaded Dataset")
    st.write(data.head())

    try:
        # Make predictions
        predictions = model.predict(data)
        data["Predicted Performance"] = predictions

        st.subheader("📊 Prediction Results")
        st.write(data)

        # Download button
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Predictions",
            csv,
            "predicted_students.csv",
            "text/csv"
        )

    except Exception as e:
        st.error("Error: Please check if dataset columns match model features.")
        st.write(e)
