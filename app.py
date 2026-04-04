import streamlit as st
import joblib

# load models
d_model = joblib.load("models/diabetes_model.pkl")
h_model = joblib.load("models/heart_model.pkl")

st.set_page_config(page_title="AI Health System")

st.title("🧠 AI Disease Prediction System")

st.warning("⚠️ This is an AI prediction tool. Consult a doctor for final diagnosis.")

option = st.selectbox("Select Disease", ["Diabetes", "Heart Disease"])

# ---------------- DIABETES ----------------
if option == "Diabetes":
    st.header("Diabetes Prediction")

    # Glucose
    glucose_option = st.selectbox("Glucose Level", ["Low", "Normal", "High"])
    if glucose_option == "Low":
        Glucose = 80
    elif glucose_option == "Normal":
        Glucose = 110
    else:
        Glucose = 160

    # BMI
    bmi_option = st.selectbox("Body Weight (BMI)", ["Underweight", "Normal", "Overweight"])
    if bmi_option == "Underweight":
        BMI = 18
    elif bmi_option == "Normal":
        BMI = 23
    else:
        BMI = 30

    Age = st.number_input("Age")
    Pregnancies = st.number_input("Pregnancies")

    data = [Glucose, BMI, Age, Pregnancies]

    if st.button("Predict Diabetes"):
        prob = d_model.predict_proba([data])[0][1]
        percentage = prob * 100

        st.write(f"Risk Score: {percentage:.1f}%")
        st.progress(int(percentage))

        if percentage > 70:
            st.error("⚠️ High Risk of Diabetes")
        elif percentage > 40:
            st.warning("⚠️ Medium Risk of Diabetes")
        else:
            st.success("✅ Low Risk of Diabetes")

# ---------------- HEART ----------------
elif option == "Heart Disease":
    st.header("Heart Disease Prediction")

    Age = st.number_input("Age")

    # Sex
    sex_option = st.selectbox("Sex", ["Male", "Female"])
    sex = 1 if sex_option == "Male" else 0

    # Chest Pain
    cp_option = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "No Pain"]
    )
    cp = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "No Pain"].index(cp_option)

    # Blood Pressure
    bp_option = st.selectbox("Blood Pressure Level", ["Low", "Normal", "High"])
    if bp_option == "Low":
        trestbps = 100
    elif bp_option == "Normal":
        trestbps = 120
    else:
        trestbps = 150

    # Cholesterol
    chol_option = st.selectbox("Cholesterol Level", ["Normal", "High"])
    chol = 180 if chol_option == "Normal" else 250

    data = [Age, sex, cp, trestbps, chol]

    if st.button("Predict Heart Disease"):
        prob = h_model.predict_proba([data])[0][1]
        percentage = prob * 100

        st.write(f"Risk Score: {percentage:.1f}%")
        st.progress(int(percentage))

        if percentage > 70:
            st.error("⚠️ High Risk of Heart Disease")
        elif percentage > 40:
            st.warning("⚠️ Medium Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

# ---------------- FOOTER ----------------
st.markdown("---")
st.write("Developed by: Atul malviya | AI ML Project")