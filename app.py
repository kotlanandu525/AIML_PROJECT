import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Heart Stroke Prediction", layout="centered")

st.title("Heart Stroke Prediction App")
st.write("This app predicts the **risk of heart stroke** using Machine Learning.")

@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

try:
    df = load_data()
except:
    st.error("heart.csv not found. Upload dataset in the same folder.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.success(f"Accuracy: {accuracy * 100:.2f}%")

st.subheader(" Enter Patient Details")

age = st.number_input("Age", 1, 120, 45)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

if st.button(" Predict Stroke Risk"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error(" High Risk of Heart Stroke")
    else:
        st.success(" Low Risk of Heart Stroke")

st.markdown("---")
st.caption("Developed using Streamlit & Machine Learning")