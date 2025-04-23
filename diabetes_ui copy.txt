import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title of the Web App
st.title("Diabetes Prediction App")

st.sidebar.header("Enter Patient Details")

def get_user_input():
    pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1)
    glucose = st.sidebar.slider("Glucose", 80, 200, 120)
    bp = st.sidebar.slider("Blood Pressure", 40, 130, 70)
    skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 300, 80)
    bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.1, 2.5, 0.5)
    age = st.sidebar.slider("Age", 10, 100, 33)

    # Return a DataFrame with user input
    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    return pd.DataFrame(data, index=[0])

# Get user input data
input_df = get_user_input()

# Scale the input features
scaled_input = scaler.transform(input_df)

# Model prediction
prediction = model.predict(scaled_input)
prediction_proba = model.predict_proba(scaled_input)

# Display results
st.subheader("Prediction Result:")
if prediction[0] == 1:
    st.write("🩺 **Diabetic**")
else:
    st.write("✅ **Not Diabetic**")

st.subheader("Prediction Probability:")
st.write(f"**Not Diabetic**: {prediction_proba[0][0]:.2f}")
st.write(f"**Diabetic**: {prediction_proba[0][1]:.2f}")


