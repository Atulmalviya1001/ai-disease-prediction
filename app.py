import streamlit as st
import joblib
import numpy as np

# =========================
# LOAD MODELS
# =========================

d_model = joblib.load(
    "diabetes_model.pkl"
)

d_scaler = joblib.load(
    "diabetes_scaler.pkl"
)

h_model = joblib.load(
    "Heart_model.pkl"
)

h_scaler = joblib.load(
    "Heart_scaler.pkl"
)

# =========================
# PAGE
# =========================

st.set_page_config(
    page_title="AI Health System",
    layout="centered"
)

st.title("🧠 AI Disease Prediction System")

st.warning(
    "⚠️ AI prediction only. Consult doctor for medical advice."
)

option = st.sidebar.selectbox(
    "Select Disease",
    ["Diabetes", "Heart Disease"]
)

# =====================================================
# DIABETES
# =====================================================

if option == "Diabetes":

    st.header("Diabetes Prediction")

    glucose = st.slider(
        "Glucose Level",
        50,
        250,
        110
    )

    weight = st.number_input(
        "Weight (kg)",
        20.0,
        200.0,
        60.0
    )

    height = st.number_input(
        "Height (feet)",
        3.0,
        8.0,
        5.5
    )

    height_m = height * 0.3048

    bmi = weight / (height_m ** 2)

    st.write(f"Calculated BMI: {bmi:.2f}")

    age = st.slider(
        "Age",
        1,
        100,
        25
    )

    pregnancies = st.number_input(
        "Pregnancies",
        0,
        20,
        0
    )

    if st.button("Predict Diabetes"):

        data = np.array([[
            glucose,
            bmi,
            age,
            pregnancies
        ]])

        data = d_scaler.transform(data)

        prob = d_model.predict_proba(data)[0][1]

        risk = prob * 100

        st.subheader(f"Risk Score: {risk:.1f}%")

        st.progress(int(risk))

        if risk > 70:

            st.error(
                "⚠️ High Risk of Diabetes"
            )

        elif risk > 40:

            st.warning(
                "⚠️ Medium Risk of Diabetes"
            )

        else:

            st.success(
                "✅ Low Risk of Diabetes"
            )

# =====================================================
# HEART
# =====================================================

elif option == "Heart Disease":

    st.header("Heart Disease Prediction")

    age = st.slider(
        "Age",
        1,
        100,
        30
    )

    sex_option = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    sex = 1 if sex_option == "Male" else 0

    cp_option = st.selectbox(
        "Chest Pain Type",
        [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "No Chest Pain"
        ]
    )

    cp_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "No Chest Pain": 3
    }

    cp = cp_map[cp_option]

    trestbps = st.slider(
        "Blood Pressure",
        80,
        220,
        120
    )

    chol = st.slider(
        "Cholesterol",
        100,
        400,
        200
    )

    fbs_option = st.selectbox(
        "High Fasting Blood Sugar?",
        ["No", "Yes"]
    )

    fbs = 1 if fbs_option == "Yes" else 0

    thalach = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    exang_option = st.selectbox(
        "Exercise Induced Chest Pain?",
        ["No", "Yes"]
    )

    exang = 1 if exang_option == "Yes" else 0

    if st.button("Predict Heart Disease"):

        data = np.array([[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            thalach,
            exang
        ]])

        data = h_scaler.transform(data)

        prob = h_model.predict_proba(data)[0][1]

        risk = prob * 100

        st.subheader(f"Risk Score: {risk:.1f}%")

        st.progress(int(risk))

        if risk > 70:

            st.error(
                "⚠️ High Risk of Heart Disease"
            )

        elif risk > 40:

            st.warning(
                "⚠️ Medium Risk of Heart Disease"
            )

        else:

            st.success(
                "✅ Low Risk of Heart Disease"
            )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.write(
    "Developed by Atul | AI ML Project"
)