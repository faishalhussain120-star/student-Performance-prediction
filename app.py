import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("model.pkl")

st.title("🎓 Student Performance Predictor")
st.write("You can either enter details manually for one student or upload a CSV for multiple students.")

# ------------------- Input Method Selection -------------------
option = st.radio("Select Input Method:", ("Manual Input", "Upload CSV"))

# ------------------- Manual Input -------------------
if option == "Manual Input":
    st.subheader("Enter Student Details")
    
    study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, value=5.0)
    attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0)
    previous_marks = st.number_input("Previous Marks", min_value=0.0, max_value=100.0, value=60.0)
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
    assignments_completed = st.number_input("Assignments Completed", min_value=0, max_value=20, value=10)
    
    if st.button("Predict Performance"):
        input_data = np.array([[study_hours, attendance, previous_marks, sleep_hours, assignments_completed]])
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("🎉 The student is likely to Perform Well!")
        else:
            st.error("⚠️ The student may need Improvement.")

# ------------------- CSV Upload -------------------
else:
    st.subheader("Upload CSV for Multiple Students")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Preview of your dataset:")
        st.dataframe(data.head())
        
        st.write("Map your CSV columns to the model features:")
        
        # Allow user to map CSV columns to required features
        study_col = st.selectbox("Column for Study Hours", data.columns)
        attendance_col = st.selectbox("Column for Attendance (%)", data.columns)
        prev_marks_col = st.selectbox("Column for Previous Marks", data.columns)
        sleep_col = st.selectbox("Column for Sleep Hours", data.columns)
        assign_col = st.selectbox("Column for Assignments Completed", data.columns)
        
        if st.button("Predict for CSV"):
            try:
                # Extract required columns based on mapping
                features = data[[study_col, attendance_col, prev_marks_col, sleep_col, assign_col]]
                
                # Ensure numeric
                features = features.apply(pd.to_numeric, errors='coerce').fillna(0)
                
                predictions = model.predict(features)
                data["Predicted Performance"] = predictions
                
                st.subheader("📊 Prediction Results")
                st.dataframe(data)
                
                # Download button
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Predictions CSV",
                    data=csv,
                    file_name="predicted_students.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error("❌ Error: Check your CSV file and column mapping.")
                st.write(e)
