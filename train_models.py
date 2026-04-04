import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# create models folder
os.makedirs("models", exist_ok=True)

# -------- DIABETES --------
df = pd.read_csv("diabetes.csv")

X = df[["Glucose", "BMI", "Age", "Pregnancies"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "models/diabetes_model.pkl")

# -------- HEART --------
df = pd.read_csv("Heart.csv")

X = df[["age", "sex", "cp", "trestbps", "chol"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "models/heart_model.pkl")

print("DONE")