import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("model.pkl")

st.title("🎓 Student Performance Predictor")
st.write("You can either enter details manually for one student or upload a CSV for multiple students.")

# ------------------- Sample CSV Download -------------------
st.subheader("📄 Download Sample CSV")
sample_csv = """Study Hours,Attendance,Previous Marks,Sleep Hours,Assignments Completed
5,80,70,7,10
3,60,50,6,8
8,90,85,8,12
2,50,40,5,5
6,75,65,7,11
7,85,78,8,12
4,55,48,6,7
9,92,88,8,14
1,45,35,5,4
6,70,60,7,10
5,65,58,6,9
8,88,82,8,13
3,50,45,6,6
7,80,75,7,12
4,60,50,6,8
9,90,86,8,15
2,55,40,5,5
6,72,65,7,11
5,68,60,7,10
8,85,80,8,13
3,52,48,6,7
7,78,74,7,12
4,60,55,6,8
9,93,90,8,15
2,50,42,5,5
6,75,68,7,11
5,66,60,7,10
8,89,84,8,14
3,54,50,6,7
7,82,76,7,12
4,61,55,6,8
9,91,87,8,15
2,53,43,5,5
6,74,66,7,11
5,67,61,7,10
8,87,83,8,14
3,56,51,6,7
7,79,75,7,12
4,62,56,6,8
9,92,88,8,15
2,51,41,5,5
6,73,65,7,11
5,69,62,7,10
8,86,81,8,13
3,55,50,6,7
7,81,77,7,12
4,63,57,6,8
9,94,91,8,15
2,52,42,5,5
6,76,69,7,11"""

# Convert to bytes
csv_bytes = sample_csv.encode('utf-8')

# Download button
st.download_button(
    label="📥 Download Sample Students CSV",
    data=csv_bytes,
    file_name="students_sample.csv",
    mime="text/csv"
)

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
