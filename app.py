import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.title("🎓 Student Performance Predictor")

st.write("Enter student details below to predict performance.")

# User Inputs
study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, value=5.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0)
previous_marks = st.number_input("Previous Marks", min_value=0.0, max_value=100.0, value=60.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
assignments_completed = st.number_input("Assignments Completed", min_value=0, max_value=20, value=10)

# Predict Button
if st.button("Predict Performance"):
    input_data = np.array([[study_hours, attendance, previous_marks, sleep_hours, assignments_completed]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🎉 The student is likely to Perform Well!")
    else:
        st.error("⚠️ The student may need Improvement.")
