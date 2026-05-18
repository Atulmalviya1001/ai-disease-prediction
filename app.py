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
# LOGIN SYSTEM
# ==========================
st.sidebar.title("Account")

menu = st.sidebar.selectbox("Menu", ["Login", "Signup"])

# SIGNUP
if menu == "Signup":

    st.subheader("Create Account")

    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")

    if st.button("Signup"):

        try:
            cursor.execute(
                "INSERT INTO users(username,password) VALUES(?,?)",
                (new_user, new_pass)
            )
            conn.commit()
            st.success("Account Created")

        except:
            st.error("Username already exists")

# LOGIN
elif menu == "Login":

    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        cursor.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        )

        user = cursor.fetchone()

        if user:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success("Login Successful")

        else:
            st.error("Invalid Credentials")

# STOP IF NOT LOGGED IN
if "logged_in" not in st.session_state:
    st.warning("Please Login First")
    st.stop()

# ==========================
# LOAD MODELS
# ==========================
d_model = joblib.load("models/diabetes_model.pkl")
d_scaler = joblib.load("models/diabetes_scaler.pkl")

h_model = joblib.load("models/Heart_model.pkl")
h_scaler = joblib.load("models/Heart_scaler.pkl")

# ==========================
# SIDEBAR
# ==========================
st.sidebar.success(f"Welcome {st.session_state['username']}")

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

# HISTORY
if st.sidebar.button("View History"):

    cursor.execute("""
        SELECT disease,risk,result,precautions
        FROM reports
        WHERE username=?
    """, (st.session_state["username"],))

    data = cursor.fetchall()

    st.subheader("📜 History")

    for r in data:
        st.write(f"""
        🩺 Disease: {r[0]}
        💊 Risk: {r[1]:.2f}%
        📌 Result: {r[2]}
        ⚠️ Precautions: {r[3]}
        """)

# OPTION
option = st.sidebar.selectbox(
    "Select Disease",
    ["Diabetes", "Heart Disease"]
)

# ==========================
# DIABETES
# ==========================
if option == "Diabetes":

    st.header("🩸 Diabetes Prediction")

    glucose = st.number_input("Glucose",50,250,110)

    weight = st.number_input("Weight",20.0,200.0,60.0)
    height = st.number_input("Height (feet)",3.0,8.0,5.5)

    bmi = weight / ((height * 0.3048) ** 2)

    st.info(f"BMI: {bmi:.2f}")

    age = st.number_input("Age",1,100,25)
    pregnancies = st.number_input("Pregnancies",0,20,0)

    if st.button("Predict Diabetes"):

        data = np.array([[glucose,bmi,age,pregnancies]])
        data = d_scaler.transform(data)

        prob = d_model.predict_proba(data)[0][1]
        risk = prob * 100

        # RESULT + PRECAUTIONS
        if risk > 70:
            result = "High Risk"
            precautions = "Reduce sugar, Exercise daily, Consult doctor"

        elif risk > 40:
            result = "Medium Risk"
            precautions = "Walk daily, Healthy diet, Monitor sugar"

        else:
            result = "Low Risk"
            precautions = "Maintain healthy lifestyle"

        # SAVE TO DB
        cursor.execute("""
            INSERT INTO reports(username,disease,risk,result,precautions)
            VALUES (?,?,?,?,?)
        """, (
            st.session_state["username"],
            "Diabetes",
            risk,
            result,
            precautions
        ))

        conn.commit()

        # SHOW RESULT
        st.subheader(f"Risk: {risk:.2f}%")

        st.write(result)
        st.write("Precautions:", precautions)

# ==========================
# HEART DISEASE
# ==========================
elif option == "Heart Disease":

    st.header("❤️ Heart Disease Prediction")

    age = st.number_input("Age",1,100,30)

    sex = 1 if st.selectbox("Sex",["Male","Female"])=="Male" else 0

    cp = st.selectbox(
        "Chest Pain",
        ["Typical","Atypical","Non-anginal","None"]
    )

    cp_map = {"Typical":0,"Atypical":1,"Non-anginal":2,"None":3}

    trestbps = st.number_input("BP",80,220,120)
    chol = st.number_input("Cholesterol",100,400,200)

    fbs = 1 if st.selectbox("FBS",["No","Yes"])=="Yes" else 0
    thalach = st.number_input("Max Heart Rate",60,220,150)
    exang = 1 if st.selectbox("Exercise Pain",["No","Yes"])=="Yes" else 0

    if st.button("Predict Heart Disease"):

        data = np.array([[age,sex,cp_map[cp],trestbps,chol,fbs,thalach,exang]])
        data = h_scaler.transform(data)

        prob = h_model.predict_proba(data)[0][1]
        risk = prob * 100

        if risk > 70:
            result = "High Risk"
            precautions = "Avoid oily food, Exercise, Stop smoking"

        elif risk > 40:
            result = "Medium Risk"
            precautions = "Walk daily, Healthy diet"

        else:
            result = "Low Risk"
            precautions = "Maintain fitness"

        cursor.execute(
    """
    INSERT INTO reports(username, disease, risk, result, precautions)
    VALUES (?, ?, ?, ?, ?)
    """,
    (
        st.session_state["username"],
        "Diabetes",
        risk,
        result,
        precautions
    )
)

        conn.commit()

        st.subheader(f"Risk: {risk:.2f}%")
        st.write(result)
        st.write("Precautions:", precautions)
