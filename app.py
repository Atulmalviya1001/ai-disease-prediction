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

    sex_option = st.selectbox("Sex", ["Male", "Female"])
    sex = 1 if sex_option == "Male" else 0

    cp = st.selectbox("Chest Pain Type", [0,1,2,3])
    trestbps = st.number_input("Blood Pressure")
    chol = st.number_input("Cholesterol")

    data = [Age, sex, cp, trestbps, chol]

    if st.button("Predict Heart Disease"):
        prob = h_model.predict_proba([data])[0][1]

        st.write("Probability:", round(prob, 2))

        if prob > 0.5:
            st.error("High Risk")
        else:
            st.success("Low Risk")