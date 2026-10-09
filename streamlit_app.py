import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Page configuration
st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Pass Prediction")
st.write("Predict a student's result using Machine Learning.")

# Load historical dataset
df = pd.read_csv("student_data.csv")

features = [
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Assignments"
]

X = df[features]
y = df["Final_Result"].astype(str).str.strip().str.title()

# Train the model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

st.subheader("Enter Student Details")

study_hours = st.number_input(
    "Daily Study Hours", min_value=0.0, max_value=24.0, value=6.0
)

attendance = st.number_input(
    "Attendance (%)", min_value=0.0, max_value=100.0, value=80.0
)

previous_marks = st.number_input(
    "Previous Marks", min_value=0.0, max_value=100.0, value=70.0
)

assignments = st.number_input(
    "Assignments Completed (out of 10)",
    min_value=0, max_value=10, value=8
)

if st.button("Predict Result"):
    student = pd.DataFrame([{
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Previous_Marks": previous_marks,
        "Assignments": assignments
    }])

    prediction = model.predict(student)[0]
    probabilities = model.predict_proba(student)[0]
    predicted_index = list(model.classes_).index(prediction)
    probability = probabilities[predicted_index] * 100

    st.subheader("Prediction Result")

    if prediction == "Pass":
        st.success("Predicted Result: PASS")
    else:
        st.error("Predicted Result: FAIL")

    st.write(
        f"Model probability for the predicted class: "
        f"{probability:.2f}%"
    )

    st.caption(
        "This is an estimate, not a guarantee of actual academic performance."
    )