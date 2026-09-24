import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Loan Amount Predictor",
    page_icon="💰",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = Path(__file__).parent / "model.pkl"


@st.cache_resource
def load_model():

    with open(
        MODEL_PATH,
        "rb"
    ) as file:

        return pickle.load(file)


model = load_model()


# ============================================================
# HEADER
# ============================================================

st.title("💰 Loan Amount Prediction")

st.write(
    "Predict the expected loan amount using applicant "
    "financial and personal information."
)

st.divider()


# ============================================================
# INPUT FORM
# ============================================================

with st.form("loan_prediction_form"):

    st.subheader("👤 Applicant Information")

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # COLUMN 1
    # --------------------------------------------------------

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=80,
            value=30
        )

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Other"
            ]
        )

        education = st.selectbox(
            "Education",
            [
                "Diploma",
                "Bachelor",
                "Master",
                "PhD"
            ]
        )


    # --------------------------------------------------------
    # COLUMN 2
    # --------------------------------------------------------

    with col2:

        employment_type = st.selectbox(
            "Employment Type",
            [
                "Private",
                "Government",
                "Contract",
                "Self-Employed"
            ]
        )

        employment_years = st.number_input(
            "Employment Years",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5
        )

        city = st.selectbox(
            "City",
            [
                "Pune",
                "Mumbai",
                "Nashik",
                "Nagpur"
            ]
        )


    # --------------------------------------------------------
    # COLUMN 3
    # --------------------------------------------------------

    with col3:

        property_ownership = st.selectbox(
            "Property Ownership",
            [
                "Rent",
                "Own",
                "Mortgage"
            ]
        )

        loan_purpose = st.selectbox(
            "Loan Purpose",
            [
                "Personal",
                "Home",
                "Car",
                "Business",
                "Education"
            ]
        )

        existing_loans = st.number_input(
            "Existing Loans",
            min_value=0,
            max_value=10,
            value=2
        )


    st.divider()

    st.subheader("💳 Financial Information")

    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------------
    # FINANCIAL 1
    # --------------------------------------------------------

    with col1:

        annual_income = st.number_input(
            "Annual Income (₹)",
            min_value=10000,
            max_value=1000000,
            value=75000,
            step=5000
        )

        savings = st.number_input(
            "Savings (₹)",
            min_value=0,
            max_value=1000000,
            value=20000,
            step=1000
        )


    # --------------------------------------------------------
    # FINANCIAL 2
    # --------------------------------------------------------

    with col2:

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=680
        )

        debt_ratio = st.number_input(
            "Debt Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.01
        )


    # --------------------------------------------------------
    # FINANCIAL 3
    # --------------------------------------------------------

    with col3:

        late_payments = st.number_input(
            "Late Payments",
            min_value=0,
            max_value=20,
            value=1
        )

        interest_rate = st.number_input(
            "Interest Rate (%)",
            min_value=1.0,
            max_value=30.0,
            value=10.0,
            step=0.1
        )


    st.divider()

    predict_button = st.form_submit_button(
        "🔮 Predict Loan Amount",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "Age": [age],

        "Gender": [gender],

        "Education": [education],

        "Employment_Type": [employment_type],

        "Annual_Income": [annual_income],

        "Credit_Score": [credit_score],

        "City": [city],

        "Debt_Ratio": [debt_ratio],

        "Employment_Years": [employment_years],

        "Late_Payments": [late_payments],

        "Existing_Loans": [existing_loans],

        "Property_Ownership": [
            property_ownership
        ],

        "Savings": [savings],

        "Interest_Rate": [
            interest_rate
        ],

        "Loan_Purpose": [
            loan_purpose
        ]
    })


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            input_data
        )[0]


        # ----------------------------------------------------
        # SANITY CHECK
        # ----------------------------------------------------

        if prediction < 0:

            prediction = 0


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        st.success(
            "Prediction generated successfully!"
        )

        st.metric(
            "Predicted Loan Amount",
            f"₹{prediction:,.2f}"
        )


        # ----------------------------------------------------
        # INPUT DETAILS
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "📋 Applicant Information"
        )

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)