import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_PATH = Path(__file__).resolve().with_name("model.pkl")
TRAINING_DATA_PATH = Path(__file__).resolve().with_name("svm.csv")

NUM_FEATURES = [
    "Age",
    "Annual_Income",
    "Credit_Score",
    "Employment_Years",
    "Debt_to_Income",
    "Loan_Amount",
    "Existing_Loans",
    "Monthly_Expenses",
]
CAT_FEATURES = ["Education", "Property_Type", "Loan_Purpose", "Loan_Approved"]
FEATURES = NUM_FEATURES + CAT_FEATURES

with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)

training_df = pd.read_csv(TRAINING_DATA_PATH)
cat_categories = [sorted(training_df[col].dropna().unique().tolist()) for col in CAT_FEATURES]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUM_FEATURES),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", categories=cat_categories),
            CAT_FEATURES,
        ),
    ]
)
preprocessor.fit(training_df[FEATURES])


def predict_employment(
    age,
    annual_income,
    credit_score,
    employment_years,
    debt_to_income,
    loan_amount,
    existing_loans,
    monthly_expenses,
    education,
    property_type,
    loan_purpose,
    loan_approved,
):
    df = pd.DataFrame(
        [{
            "Age": age,
            "Annual_Income": annual_income,
            "Credit_Score": credit_score,
            "Employment_Years": employment_years,
            "Debt_to_Income": debt_to_income,
            "Loan_Amount": loan_amount,
            "Existing_Loans": existing_loans,
            "Monthly_Expenses": monthly_expenses,
            "Education": education,
            "Property_Type": property_type,
            "Loan_Purpose": loan_purpose,
            "Loan_Approved": loan_approved,
        }]
    )

    transformed = preprocessor.transform(df[FEATURES])
    prediction = model.predict(transformed)[0]
    return prediction


st.set_page_config(page_title="Employment Type Prediction", page_icon="💼")
st.title("💼 Employment Type Prediction")
st.caption("SVM model to predict the applicant's employment type.")

with st.form("employment_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=28)
        annual_income = st.number_input("Annual Income", min_value=0, value=65000)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=720)
        employment_years = st.number_input("Employment Years", min_value=0, max_value=50, value=5)
        debt_to_income = st.number_input("Debt to Income (%)", min_value=0.0, max_value=100.0, value=25.5)

    with col2:
        loan_amount = st.number_input("Loan Amount", min_value=0, value=250000)
        existing_loans = st.number_input("Existing Loans", min_value=0, max_value=20, value=1)
        monthly_expenses = st.number_input("Monthly Expenses", min_value=0, value=30000)
        education = st.selectbox("Education", ["Graduate", "Postgraduate"], index=0)
        property_type = st.selectbox("Property Type", ["Owned", "Rented"], index=0)
        loan_purpose = st.selectbox("Loan Purpose", ["Business", "Education", "Home", "Personal", "Vehicle"], index=2)
        loan_approved = st.selectbox("Loan Approved", ["Yes", "No"], index=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    result = predict_employment(
        age,
        annual_income,
        credit_score,
        employment_years,
        debt_to_income,
        loan_amount,
        existing_loans,
        monthly_expenses,
        education,
        property_type,
        loan_purpose,
        loan_approved,
    )
    st.success(f"Predicted Employment Type: {result}")
