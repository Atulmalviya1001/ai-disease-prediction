# =========================================
# IMPORT LIBRARIES
# =========================================

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =========================================
# CREATE MODELS DIRECTORY
# =========================================

print("📁 Creating models directory...")
os.makedirs("models", exist_ok=True)


# =========================================
# FUNCTION: PRINT BASIC INFO
# =========================================

def explore_data(df, name):
    print(f"\n📊 Exploring {name} Dataset")
    print("-" * 40)
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())


# =========================================
# FUNCTION: HANDLE MISSING VALUES
# =========================================

def handle_missing_values(df, cols):
    print("\n🔧 Handling missing values...")

    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    df.fillna(df.mean(numeric_only=True), inplace=True)

    print("✅ Missing values handled")
    return df


# =========================================
# FUNCTION: SCALE DATA
# =========================================

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("📏 Data scaling completed")
    return X_train_scaled, X_test_scaled, scaler


# =========================================
# FUNCTION: TRAIN MODEL
# =========================================

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("🤖 Model training completed")
    return model


# =========================================
# FUNCTION: EVALUATE MODEL
# =========================================

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n📈 Model Evaluation")
    print("-" * 40)
    print("Accuracy:", round(acc, 3))
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return acc


# =========================================
# FUNCTION: FEATURE IMPORTANCE
# =========================================

def show_feature_importance(model, feature_names):
    print("\n🔍 Feature Importance:")
    importances = model.feature_importances_

    for name, score in zip(feature_names, importances):
        print(f"{name}: {round(score, 3)}")


# =========================================
# DIABETES MODEL PIPELINE
# =========================================

print("\n==============================")
print("🩺 DIABETES MODEL TRAINING")
print("==============================")

df_diabetes = pd.read_csv("diabetes.csv")

explore_data(df_diabetes, "Diabetes")

df_diabetes = handle_missing_values(df_diabetes, ["Glucose", "BMI"])

features_d = ["Glucose", "BMI", "Age", "Pregnancies"]
target_d = "Outcome"

X_d = df_diabetes[features_d]
y_d = df_diabetes[target_d]

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_d, y_d, test_size=0.2, random_state=42
)

X_train_d, X_test_d, scaler_d = scale_data(X_train_d, X_test_d)

model_d = train_model(X_train_d, y_train_d)

evaluate_model(model_d, X_test_d, y_test_d)

show_feature_importance(model_d, features_d)

joblib.dump(model_d, "models/diabetes_model.pkl")
joblib.dump(scaler_d, "models/diabetes_scaler.pkl")

print("💾 Diabetes model & scaler saved")


# =========================================
# HEART MODEL PIPELINE
# =========================================

print("\n==============================")
print("❤️ HEART MODEL TRAINING")
print("==============================")

df_heart = pd.read_csv("Heart.csv")

explore_data(df_heart, "Heart")

df_heart = handle_missing_values(df_heart, [])

features_h = ["age", "sex", "cp", "trestbps", "chol"]
target_h = "target"

X_h = df_heart[features_h]
y_h = df_heart[target_h]

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_h, y_h, test_size=0.2, random_state=42
)

X_train_h, X_test_h, scaler_h = scale_data(X_train_h, X_test_h)

model_h = train_model(X_train_h, y_train_h)

evaluate_model(model_h, X_test_h, y_test_h)

show_feature_importance(model_h, features_h)

joblib.dump(model_h, "models/heart_model.pkl")
joblib.dump(scaler_h, "models/heart_scaler.pkl")

print("💾 Heart model & scaler saved")


# =========================================
# FINAL MESSAGE
# =========================================

print("\n====================================")
print("✅ ALL MODELS TRAINED SUCCESSFULLY")
print("====================================")