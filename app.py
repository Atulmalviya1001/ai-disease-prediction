import streamlit as st
import joblib
import numpy as np

from database import conn, cursor

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

/* Background */
.stApp{
    background: linear-gradient(135deg,#0f172a,#1e293b);
    color:white;
}

/* Title */
h1{
    text-align:center;
    color:#38bdf8;
    font-size:42px !important;
    font-weight:bold;
}

/* Headers */
h2,h3{
    color:white;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background-color:#111827;
}

/* Modern Buttons */
.stButton > button{
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

.stButton > button:hover{
    transform:scale(1.03);
    box-shadow:0px 0px 20px rgba(59,130,246,0.5);
}

/* Input Fields */
div[data-baseweb="input"]{
    background-color:#1e293b !important;
    border-radius:12px !important;
    border:1px solid #334155 !important;
}

input{
    color:white !important;
    font-size:16px !important;
}

/* Selectbox */
div[data-baseweb="select"]{
    background-color:#1e293b !important;
    border-radius:12px !important;
}

/* Result Card */
.result-box{
    background:#1e293b;
    padding:20px;
    border-radius:16px;
    margin-top:20px;
    text-align:center;
    border:1px solid #334155;
}

/* Footer */
footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODELS
# ==========================
d_model = joblib.load("models/diabetes_model.pkl")
d_scaler = joblib.load("models/diabetes_scaler.pkl")

h_model = joblib.load("models/heart_model.pkl")
h_scaler = joblib.load("models/heart_scaler.pkl")

# ==========================
# TITLE
# ==========================
st.title("🧠 AI Disease Prediction System")

st.warning(
    "⚠️ AI prediction only. Consult doctor for medical advice."
)

# ==========================
# LOGIN SYSTEM
# ==========================
st.sidebar.title("Account")

menu = st.sidebar.selectbox(
    "Menu",
    ["Login", "Signup"]
)

# --------------------------
# SIGNUP
# --------------------------
if menu == "Signup":

    st.subheader("Create Account")

    new_user = st.text_input("Username")

    new_pass = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Signup"):

        try:

            cursor.execute(
                "INSERT INTO users(username,password) VALUES(?,?)",
                (new_user, new_pass)
            )

            conn.commit()

            st.success("Account Created Successfully")

        except:

            st.error("Username already exists")

# --------------------------
# LOGIN
# --------------------------
elif menu == "Login":

    st.subheader("Login")

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        cursor.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        )

        data = cursor.fetchone()

        if data:

            st.session_state["logged_in"] = True
            st.session_state["username"] = username

            st.success("Login Successful")

        else:

            st.error("Invalid Username or Password")

# ==========================
# STOP IF NOT LOGGED IN
# ==========================
if "logged_in" not in st.session_state:

    st.warning("Please Login First")

    st.stop()

# ==========================
# SIDEBAR
# ==========================
st.sidebar.success(
    f"Welcome {st.session_state['username']}"
)

# Logout
if st.sidebar.button("Logout"):

    st.session_state.clear()

    st.rerun()

# View History
if st.sidebar.button("View History"):

    cursor.execute(
        "SELECT disease,risk,result FROM reports WHERE username=?",
        (st.session_state["username"],)
    )

    reports = cursor.fetchall()

    st.subheader("📜 Prediction History")

    for report in reports:

        st.write(
            f"Disease: {report[0]} | Risk: {report[1]:.1f}% | Result: {report[2]}"
        )

# Disease Selection
option = st.sidebar.selectbox(
    "Select Disease",
    ["Diabetes", "Heart Disease"]
)

# =====================================================
# DIABETES
# =====================================================
if option == "Diabetes":

    st.header("🩸 Diabetes Prediction")

    glucose = st.number_input(
        "Glucose Level",
        min_value=50,
        max_value=250,
        value=110
    )

    col1, col2 = st.columns(2)

    with col1:

        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=200.0,
            value=60.0
        )

    with col2:

        height = st.number_input(
            "Height (feet)",
            min_value=3.0,
            max_value=8.0,
            value=5.5
        )

    height_m = height * 0.3048

    bmi = weight / (height_m ** 2)

    st.info(f"Calculated BMI: {bmi:.2f}")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=25
    )

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=0
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

        st.markdown(f"""
        <div class="result-box">
            <h2>Risk Score</h2>
            <h1>{risk:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(risk))

        # Risk Level
        if risk > 70:

            result = "High Risk"

        elif risk > 40:

            result = "Medium Risk"

        else:

            result = "Low Risk"

        # Save Report
        cursor.execute(
            "INSERT INTO reports(username,disease,risk,result) VALUES(?,?,?,?)",
            (
                st.session_state["username"],
                "Diabetes",
                risk,
                result
            )
        )

        conn.commit()

        # Show Result
        if risk > 70:

            st.error("⚠️ High Risk of Diabetes")

            st.write("""
### Precautions
- Reduce sugar intake
- Exercise daily
- Drink enough water
- Avoid junk food
- Maintain healthy weight
- Consult doctor immediately
""")

        elif risk > 40:

            st.warning("⚠️ Medium Risk of Diabetes")

            st.write("""
### Precautions
- Walk daily
- Avoid stress
- Monitor glucose regularly
- Improve diet quality
""")

        else:

            st.success("✅ Low Risk of Diabetes")

            st.write("""
### Healthy Habits
- Continue balanced diet
- Exercise regularly
- Sleep properly
""")

# =====================================================
# HEART DISEASE
# =====================================================
elif option == "Heart Disease":

    st.header("❤️ Heart Disease Prediction")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30
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

    trestbps = st.number_input(
        "Blood Pressure",
        min_value=80,
        max_value=220,
        value=120
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=100,
        max_value=400,
        value=200
    )

    fbs_option = st.selectbox(
        "High Fasting Blood Sugar?",
        ["No", "Yes"]
    )

    fbs = 1 if fbs_option == "Yes" else 0

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=60,
        max_value=220,
        value=150
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

        st.markdown(f"""
        <div class="result-box">
            <h2>Risk Score</h2>
            <h1>{risk:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(risk))

        # Risk Level
        if risk > 70:

            result = "High Risk"

        elif risk > 40:

            result = "Medium Risk"

        else:

            result = "Low Risk"

        # Save Report
        cursor.execute(
            "INSERT INTO reports(username,disease,risk,result) VALUES(?,?,?,?)",
            (
                st.session_state["username"],
                "Heart Disease",
                risk,
                result
            )
        )

        conn.commit()

        # Show Result
        if risk > 70:

            st.error("⚠️ High Risk of Heart Disease")

            st.write("""
### Precautions
- Reduce oily food
- Reduce salt intake
- Exercise regularly
- Stop smoking
- Monitor blood pressure
- Consult cardiologist immediately
""")

        elif risk > 40:

            st.warning("⚠️ Medium Risk of Heart Disease")

            st.write("""
### Precautions
- Walk daily
- Avoid stress
- Eat healthy foods
- Monitor cholesterol
""")

        else:

            st.success("✅ Low Risk of Heart Disease")

            st.write("""
### Healthy Habits
- Maintain balanced diet
- Exercise regularly
- Sleep properly
""")

# ==========================
# FOOTER
# ==========================
st.markdown("---")

st.write("Developed by Atul | AI ML Project")
