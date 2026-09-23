import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_PATH = Path(__file__).with_name("classifier.pkl")
DATA_PATH = Path(__file__).with_name("employee_attrition.csv")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


def fill_missing_values(df):
    for column in ["MonthlyIncome", "JobSatisfaction"]:
        if column in df.columns and df[column].isna().any():
            df[column] = df[column].fillna(df[column].median())

    for column in ["Department", "OverTime"]:
        if column in df.columns and df[column].isna().any():
            mode_value = df[column].mode(dropna=True)
            if not mode_value.empty:
                df[column] = df[column].fillna(mode_value.iloc[0])

    return df


@st.cache_resource
def load_preprocessor():
    training_df = pd.read_csv(DATA_PATH)
    feature_df = training_df.drop(columns=["Attrition"])
    feature_df = fill_missing_values(feature_df)

    numeric_features = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = feature_df.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )
    preprocessor.fit(feature_df)
    return preprocessor


def predict_attrition(raw_input):
    model = load_model()
    preprocessor = load_preprocessor()

    df = pd.DataFrame([raw_input], columns=[
        "Age",
        "Department",
        "Education",
        "JobLevel",
        "Gender",
        "Experience",
        "MonthlyIncome",
        "OverTime",
        "JobSatisfaction",
        "MaritalStatus",
    ])
    df = fill_missing_values(df)

    transformed = preprocessor.transform(df)
    prediction = model.predict(transformed)[0]

    try:
        probabilities = model.predict_proba(transformed)[0]
        confidence = float(max(probabilities))
    except Exception:
        confidence = 0.5

    return prediction, confidence


st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")
st.title("Employee Attrition Prediction")
st.write("Enter employee details to predict whether they are likely to leave the company.")

with st.form("attrition_form"):
    age = st.number_input("Age", min_value=18, max_value=80, value=35)
    department = st.selectbox("Department", ["IT", "HR", "Finance", "Sales"])
    education = st.selectbox("Education", ["Bachelor", "Master", "PhD"])
    job_level = st.number_input("Job Level", min_value=1, max_value=5, value=2)
    gender = st.selectbox("Gender", ["Male", "Female"])
    experience = st.number_input("Experience (years)", min_value=0, max_value=40, value=5)
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=200000, value=5000)
    overtime = st.selectbox("Over Time", ["Yes", "No"])
    job_satisfaction = st.number_input("Job Satisfaction", min_value=1, max_value=5, value=3)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    submitted = st.form_submit_button("Predict")

if submitted:
    sample = {
        "Age": age,
        "Department": department,
        "Education": education,
        "JobLevel": job_level,
        "Gender": gender,
        "Experience": experience,
        "MonthlyIncome": monthly_income,
        "OverTime": overtime,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
    }

    prediction, confidence = predict_attrition(sample)
    label = "Likely to leave the company" if prediction == "Yes" else "Likely to stay"

    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence * 100:.2f}%")