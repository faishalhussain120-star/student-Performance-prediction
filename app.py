import streamlit as st
import pickle

# Page configuration
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

# Custom CSS Styling
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.title("🎓 Student Performance Predictor")
st.write("Fill in the details below to predict student performance.")

st.divider()

# Create two columns
col1, col2 = st.columns(2)

with col1:
    study = st.number_input("📚 Study Hours", min_value=0, max_value=15, step=1)
    attendance = st.number_input("🏫 Attendance (%)", min_value=0, max_value=100, step=1)
    sleep = st.number_input("😴 Sleep Hours", min_value=0, max_value=12, step=1)

with col2:
    marks = st.number_input("📝 Previous Marks", min_value=0, max_value=100, step=1)
    assignment = st.selectbox("📌 Assignment Completed?", ["No", "Yes"])

st.divider()

# Convert Yes/No to 0/1
assignment_value = 1 if assignment == "Yes" else 0

# Predict Button
if st.button("🔮 Predict Performance"):
    result = model.predict([[study, attendance, marks, sleep, assignment_value]])
    
    if result[0] == 1:
        st.success("✅ The student is likely to PASS 🎉")
    else:
        st.error("❌ The student is likely to FAIL 📉")

st.markdown("---")
st.caption("Built using Machine Learning with Streamlit 🚀")