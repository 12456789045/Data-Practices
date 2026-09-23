
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

MODEL_PATH = Path(__file__).with_name("logic.pkl")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict(hours_studied, attendance):

    # Create DataFrame with the SAME columns
    # used during model training
    input_data = pd.DataFrame({
        "Hours_Studied": [hours_studied],
        "Attendance": [attendance]
    })

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probability = model.predict_proba(input_data)[0]

    return prediction, probability


# --------------------------------------------------
# Streamlit App
# --------------------------------------------------

def main():

    st.title("🎓 Student Pass Prediction")

    st.markdown(
        """
        <div style="background-color:tomato;padding:10px">
            <h2 style="color:white;text-align:center;">
                Student Pass Prediction ML App
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("Enter the student's study hours and attendance.")

    # Input fields
    Hours_Studied = st.number_input(
        "Hours Studied",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    Attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0
    )

    # Prediction button
    if st.button("Predict"):

        prediction, probability = predict(
            Hours_Studied,
            Attendance
        )

        # Convert prediction to readable result
        if prediction == 1:
            result = "PASS ✅"
        else:
            result = "FAIL ❌"

        # Probability of passing
        pass_probability = probability[1] * 100

        st.success(f"Prediction: {result}")

        st.metric(
            "Probability of Passing",
            f"{pass_probability:.2f}%"
        )

        # Show input data
        st.subheader("Input Data")

        input_data = pd.DataFrame({
            "Hours Studied": [Hours_Studied],
            "Attendance": [Attendance]
        })

        st.dataframe(input_data, use_container_width=True)


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":
    main()
