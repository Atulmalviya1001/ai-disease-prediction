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

    # Weight & Height → BMI
    weight = st.number_input("Weight (kg)")
    height = st.number_input("Height (meters, e.g., 1.7)")

    if height > 0:
        BMI = weight / (height ** 2)
    else:
        BMI = 0

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

            st.subheader("🍽️ Diet Suggestions")
            st.write("""
            - Avoid sugar & sweets
            - Eat whole grains (roti, oats)
            - Increase vegetables & fruits
            - Drink plenty of water
            """)

            st.subheader("🏃 Lifestyle Tips")
            st.write("""
            - Daily walking (30 mins)
            - Reduce weight
            - Avoid junk food
            """)

        elif percentage > 40:
            st.warning("⚠️ Medium Risk of Diabetes")

            st.subheader("🍽️ Diet Suggestions")
            st.write("""
            - Reduce sugar intake
            - Balanced diet
            - Avoid fried food
            """)

        else:
            st.success("✅ Low Risk of Diabetes")

            st.subheader("🍽️ Maintain Healthy Diet")
            st.write("""
            - Continue balanced diet
            - Regular exercise
            """)

# ---------------- HEART ----------------
elif option == "Heart Disease":
    st.header("Heart Disease Prediction")

    Age = st.number_input("Age")

    # Sex
    sex_option = st.selectbox("Sex", ["Male", "Female"])
    sex = 1 if sex_option == "Male" else 0

    # Chest Pain (simple)
    cp_option = st.selectbox(
        "Do you feel chest pain?",
        ["No Pain", "Mild Pain", "Moderate Pain", "Severe Pain"]
    )

    if cp_option == "No Pain":
        cp = 3
    elif cp_option == "Mild Pain":
        cp = 2
    elif cp_option == "Moderate Pain":
        cp = 1
    else:
        cp = 0

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

            st.subheader("🍽️ Diet Suggestions")
            st.write("""
            - Reduce salt intake
            - Avoid oily & fried food
            - Eat fruits, nuts, green vegetables
            """)

            st.subheader("❤️ Lifestyle Tips")
            st.write("""
            - Regular exercise
            - Avoid smoking
            - Manage stress
            """)

        elif percentage > 40:
            st.warning("⚠️ Medium Risk of Heart Disease")

            st.subheader("🍽️ Diet Suggestions")
            st.write("""
            - Reduce cholesterol food
            - Eat healthy fats
            """)

        else:
            st.success("✅ Low Risk of Heart Disease")

            st.subheader("🍽️ Maintain Healthy Lifestyle")
            st.write("""
            - Keep active
            - Eat balanced diet
            """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.write("Developed by: Atul malviya  | AI ML Project")

