import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.title("🎓 Student Performance Predictor")
st.write("You can either enter details manually for one student or upload a CSV for multiple students.")

# Option: Manual input or CSV
option = st.radio("Select Input Method:", ("Manual Input", "Upload CSV"))

# ---------------- MANUAL INPUT ----------------
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

# ---------------- CSV UPLOAD ----------------
else:
    st.subheader("Upload CSV File for Multiple Students")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            
            # Required columns
            required_cols = ["study_hours", "attendance", "previous_marks", "sleep_hours", "assignments_completed"]
            
            if not all(col in data.columns for col in required_cols):
                st.error(f"CSV must contain these columns: {required_cols}")
            else:
                predictions = model.predict(data[required_cols])
                data["Predicted Performance"] = predictions
                
                st.subheader("📊 Prediction Results")
                st.write(data)
                
                # Download CSV
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Predictions",
                    csv,
                    "predicted_students.csv",
                    "text/csv"
                )
        except Exception as e:
            st.error("❌ Error: Please check your CSV file format and values.")
            st.write(e)
