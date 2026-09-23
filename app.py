import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("model.pkl")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    if isinstance(artifact, dict) and "preprocessor" in artifact and "model" in artifact:
        return artifact["preprocessor"], artifact["model"]

    if hasattr(artifact, "predict") and hasattr(artifact, "named_steps"):
        return artifact, None

    raise ValueError(
        "model.pkl must contain a preprocessing object and trained model. "
        "Please save {'preprocessor': ..., 'model': ...}."
    )


def predict_heart_disease(raw_input):
    preprocessor, model = load_model()

    features = [
        "Age",
        "Sex",
        "ChestPainType",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "RestingECG",
        "MaxHR",
        "ExerciseAngina",
        "Oldpeak",
        "ST_Slope",
    ]

    df = pd.DataFrame([raw_input], columns=features)
    processed = preprocessor.transform(df)
    prediction = model.predict(processed)[0]
    proba = model.predict_proba(processed)[0]

    return int(prediction), float(max(proba))


st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("Heart Disease Prediction")
st.write("Enter the patient details and predict whether heart disease is likely.")

with st.form("heart_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=45)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["TA", "ATA", "NAP", "ASY"],
    )
    resting_bp = st.number_input("Resting BP", min_value=80, max_value=200, value=120)
    cholesterol = st.number_input("Cholesterol", min_value=80, max_value=600, value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Max HR", min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"])
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    submitted = st.form_submit_button("Predict")

if submitted:
    user_input = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
    }

    prediction, confidence = predict_heart_disease(user_input)
    label = "Heart Disease Present" if prediction == 1 else "No Heart Disease"

    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence * 100:.2f}%")
