import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("model.pkl")

st.title("🎓 Student Performance Predictor")
st.write("Enter details manually for one student or upload a CSV for bulk predictions.")

# ---------------- Sample CSV Download ----------------
st.subheader("📄 Download Sample CSV")
sample_csv = """Study Hours,Attendance,Previous Marks,Sleep Hours,Assignments Completed
5,80,70,7,10
3,60,50,6,8
8,90,85,8,12
2,50,40,5,5
6,75,65,7,11
7,85,78,8,12
4,55,48,6,7
9,92,88,8,14"""
st.download_button(
    "📥 Download Sample Students CSV",
    data=sample_csv,
    file_name="students_sample.csv",
    mime="text/csv"
)

# ---------------- Input Method ----------------
option = st.radio("Select Input Method:", ["Manual Input", "Upload CSV"])

# ---------------- Manual Input ----------------
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
        st.success(f"Predicted Performance: {'Perform Well' if prediction[0]==1 else 'Needs Improvement'}")

# ---------------- CSV Upload ----------------
else:
    st.subheader("Upload CSV for Bulk Prediction")
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Preview of your CSV:")
            st.dataframe(data.head())

            # ---------------- Fix Column Names ----------------
            rename_dict = {
                "Study Hours": "study_hours",
                "Attendance": "attendance",
                "Previous Marks": "previous_marks",
                "Sleep Hours": "sleep_hours",
                "Assignments Completed": "assignment"
            }
            data.rename(columns=rename_dict, inplace=True)

            # ---------------- Fill missing columns ----------------
            required_cols = ["study_hours","attendance","previous_marks","sleep_hours","assignment"]
            for col in required_cols:
                if col not in data.columns:
                    st.warning(f"Missing column '{col}' added with zeros")
                    data[col] = 0

            # ---------------- Prepare features ----------------
            features = data[required_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

            # ---------------- Predict ----------------
            predictions = model.predict(features)
            data["Predicted Performance"] = ["Perform Well" if p==1 else "Needs Improvement" for p in predictions]
            
            st.subheader("📊 Prediction Results")
            st.dataframe(data)

            # ---------------- Download Predictions ----------------
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Predictions CSV",
                data=csv,
                file_name="predicted_students.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error("❌ Error processing the CSV. Make sure the file is valid.")
            st.write(e)

