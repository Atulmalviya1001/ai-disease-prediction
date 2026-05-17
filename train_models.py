import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# =========================
# DIABETES MODEL
# =========================

df = pd.read_csv("diabetes.csv")

# Features
X = df[[
    "Glucose",
    "BMI",
    "Age",
    "Pregnancies"
]]

# Target
y = df["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Scaling
d_scaler = StandardScaler()

X_train = d_scaler.fit_transform(X_train)
X_test = d_scaler.transform(X_test)

# Model
d_model = LogisticRegression(max_iter=1000)

d_model.fit(X_train, y_train)

# Accuracy
pred = d_model.predict(X_test)

print("Diabetes Accuracy:",
      accuracy_score(y_test, pred))

# Save
joblib.dump(d_model,
            "models/diabetes_model.pkl")

joblib.dump(d_scaler,
            "diabetes_scaler.pkl")


# =========================
# HEART MODEL
# =========================

heart = pd.read_csv("Heart.csv")

X = heart[[
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "thalach",
    "exang"
]]

y = heart["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Scaling
h_scaler = StandardScaler()

X_train = h_scaler.fit_transform(X_train)
X_test = h_scaler.transform(X_test)

# Model
h_model = LogisticRegression(max_iter=1000)

h_model.fit(X_train, y_train)

pred = h_model.predict(X_test)

print("Heart Accuracy:",
      accuracy_score(y_test, pred))

# Save
joblib.dump(h_model,
            "models/Heart_model.pkl")

joblib.dump(h_scaler,
            "Heart_scaler.pkl")

print("Models Trained Successfully")
