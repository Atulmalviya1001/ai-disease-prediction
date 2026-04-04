import streamlit as st
import joblib

# load models
d_model = joblib.load("models/diabetes_model.pkl")
h_model = joblib.load("models/heart_model.pkl")

st.title("AI Disease Prediction System")

option = st.selectbox("Select Disease", ["Diabetes", "Heart Disease"])

# -------- DIABETES --------
if option == "Diabetes":
    st.header("Diabetes Prediction")

    Glucose = st.number_input("Glucose Level")
    BMI = st.number_input("BMI")
    Age = st.number_input("Age")
    Pregnancies = st.number_input("Pregnancies")

    data = [Glucose, BMI, Age, Pregnancies]

    if st.button("Predict Diabetes"):
        prob = d_model.predict_proba([data])[0][1]

        st.write("Probability:", round(prob, 2))

        if prob > 0.5:
            st.error("High Risk")
        else:
            st.success("Low Risk")

# -------- HEART --------
elif option == "Heart Disease":
    st.header("Heart Disease Prediction")

    Age = st.number_input("Age")

    # Sex
    sex_option = st.selectbox("Sex", ["Male", "Female"])
    sex = 1 if sex_option == "Male" else 0

    # Chest Pain (Human-friendly)
    cp_option = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "No Pain"]
    )
    cp = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "No Pain"].index(cp_option)

    # Blood Pressure (Simple)
    bp_option = st.selectbox(
        "Blood Pressure Level",
        ["Low", "Normal", "High"]
    )
    if bp_option == "Low":
        trestbps = 100
    elif bp_option == "Normal":
        trestbps = 120
    else:
        trestbps = 150

    # Cholesterol (Simple)
    chol_option = st.selectbox(
        "Cholesterol Level",
        ["Normal", "High"]
    )
    chol = 180 if chol_option == "Normal" else 250

    data = [Age, sex, cp, trestbps, chol]

    if st.button("Predict Heart Disease"):
        prob = h_model.predict_proba([data])[0][1]

        st.write("Probability:", round(prob, 2))

        if prob > 0.7:
            st.error("⚠️ High Risk of Heart Disease")
        elif prob > 0.4:
            st.warning("⚠️ Medium Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")