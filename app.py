import streamlit as st
import joblib
import numpy as np

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="AI Health System",
    layout="centered"
)

# ==========================
# MODERN CSS
# ==========================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#0f172a,#1e293b);
    color:white;
}

h1{
    text-align:center;
    color:#38bdf8;
    font-size:42px !important;
}

h2,h3{
    color:white;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#111827;
}

/* Modern Button */
.stButton>button{
    width:100%;
    border:none;
    border-radius:14px;
    padding:14px;
    font-size:17px;
    font-weight:bold;
    background:linear-gradient(90deg,#06b6d4,#2563eb);
    color:white;
    transition:0.3s;
}

.stButton>button:hover{
    transform:scale(1.04);
    box-shadow:0px 0px 20px rgba(59,130,246,0.6);
}

/* Inputs */
.stNumberInput, .stSlider, .stSelectbox{
    background:#1e293b;
    border-radius:10px;
}

/* Result Card */
.result-box{
    padding:20px;
    border-radius:15px;
    background:#1e293b;
    margin-top:15px;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODELS
# ==========================
d_model = joblib.load("models/diabetes_model.pkl")
d_scaler = joblib.load("models/diabetes_scaler.pkl")

h_model = joblib.load("models/Heart_model.pkl")
h_scaler = joblib.load("models/Heart_scaler.pkl")

# ==========================
# HEADER
# ==========================
st.title("🧠 AI Disease Prediction System")
st.warning("⚠️ AI prediction only. Consult a doctor for real diagnosis.")

option = st.sidebar.selectbox(
    "Select Disease",
    ["Diabetes", "Heart Disease"]
)

# =====================================================
# DIABETES
# =====================================================
if option == "Diabetes":

    st.header("🩸 Diabetes Prediction")

    glucose = st.slider("Glucose Level",50,250,110)

    col1,col2 = st.columns(2)

    with col1:
        weight = st.number_input("Weight (kg)",20.0,200.0,60.0)

    with col2:
        height = st.number_input("Height (feet)",3.0,8.0,5.5)

    height_m = height * 0.3048
    bmi = weight / (height_m**2)

    st.info(f"Calculated BMI: {bmi:.2f}")

    age = st.slider("Age",1,100,25)
    pregnancies = st.number_input("Pregnancies",0,20,0)

    if st.button("Predict Diabetes"):

        data = np.array([[glucose,bmi,age,pregnancies]])
        data = d_scaler.transform(data)

        prob = d_model.predict_proba(data)[0][1]
        risk = prob*100

        st.markdown(f"""
        <div class="result-box">
        <h3>Risk Score: {risk:.1f}%</h3>
        </div>
        """,unsafe_allow_html=True)

        st.progress(int(risk))

        if risk > 70:
            st.error("⚠️ High Risk of Diabetes")
            st.write("""
### Precautions:
- Reduce sugar intake
- Exercise daily
- Maintain healthy weight
- Drink enough water
- Monitor blood sugar regularly
- Consult doctor immediately
""")

        elif risk > 40:
            st.warning("⚠️ Medium Risk of Diabetes")
            st.write("""
### Precautions:
- Avoid junk food
- Walk 30 mins daily
- Reduce stress
- Monitor glucose monthly
""")

        else:
            st.success("✅ Low Risk of Diabetes")
            st.write("Maintain healthy lifestyle.")

# =====================================================
# HEART
# =====================================================
elif option == "Heart Disease":

    st.header("❤️ Heart Disease Prediction")

    age = st.slider("Age",1,100,30)

    sex_option = st.selectbox("Sex",["Male","Female"])
    sex = 1 if sex_option=="Male" else 0

    cp_option = st.selectbox("Chest Pain Type",
        ["Typical Angina","Atypical Angina","Non-anginal Pain","No Chest Pain"]
    )

    cp_map={
        "Typical Angina":0,
        "Atypical Angina":1,
        "Non-anginal Pain":2,
        "No Chest Pain":3
    }

    cp=cp_map[cp_option]

    trestbps = st.slider("Blood Pressure",80,220,120)
    chol = st.slider("Cholesterol",100,400,200)

    fbs = 1 if st.selectbox(
        "High Fasting Blood Sugar?",
        ["No","Yes"]
    )=="Yes" else 0

    thalach = st.slider("Maximum Heart Rate",60,220,150)

    exang = 1 if st.selectbox(
        "Exercise Induced Chest Pain?",
        ["No","Yes"]
    )=="Yes" else 0

    if st.button("Predict Heart Disease"):

        data=np.array([[age,sex,cp,trestbps,chol,fbs,thalach,exang]])
        data=h_scaler.transform(data)

        prob=h_model.predict_proba(data)[0][1]
        risk=prob*100

        st.markdown(f"""
        <div class="result-box">
        <h3>Risk Score: {risk:.1f}%</h3>
        </div>
        """,unsafe_allow_html=True)

        st.progress(int(risk))

        if risk>70:
            st.error("⚠️ High Risk of Heart Disease")
            st.write("""
### Precautions:
- Avoid oily food
- Reduce salt intake
- Stop smoking
- Exercise daily
- Check BP regularly
- Consult cardiologist immediately
""")

        elif risk>40:
            st.warning("⚠️ Medium Risk")
            st.write("""
### Precautions:
- Walk daily
- Avoid stress
- Eat fruits & vegetables
- Monitor cholesterol
""")

        else:
            st.success("✅ Low Risk")
            st.write("Maintain healthy habits.")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.write("Developed by Atul | AI ML Project")