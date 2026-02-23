import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------- Page Config ----------------
st.set_page_config(page_title="Heart Stroke Dashboard", page_icon="💜", layout="wide")

# ---------------- Custom CSS (FIXED VISIBILITY) ----------------
st.markdown("""
<style>
/* App background */
.main {
    background: linear-gradient(135deg, #ede9fe, #f5f3ff, #faf5ff);
}

/* Force dark text everywhere */
* {
    color: #0f172a !important;
}

/* Cards */
.card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 6px 16px rgba(0,0,0,0.1);
}

/* KPI accents */
.kpi1 { border-top: 5px solid #7c3aed; }
.kpi2 { border-top: 5px solid #22c55e; }
.kpi3 { border-top: 5px solid #f59e0b; }

/* Result boxes */
.result-high {
    background: #fee2e2;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    font-size: 20px;
    font-weight: 600;
    color: #7f1d1d !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.12);
}

.result-low {
    background: #dcfce7;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    font-size: 20px;
    font-weight: 600;
    color: #14532d !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.12);
}

/* Sidebar background + text */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ede9fe, #f5f3ff);
}

section[data-testid="stSidebar"] * {
    color: #0f172a !important;
}

/* Inputs text */
input, select, textarea {
    color: #0f172a !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.title("💜 Heart Stroke Prediction Dashboard")
st.write("AI-powered system to estimate heart stroke risk")

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

try:
    df = load_data()
except:
    st.error("❌ heart.csv not found. Keep it in the same folder as app.py")
    st.stop()

# ---------------- Train Model ----------------
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# ---------------- KPI Cards ----------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="card kpi1">
        <div>Dataset Size</div>
        <h2>{df.shape[0]}</h2>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card kpi2">
        <div>Features</div>
        <h2>{df.shape[1]-1}</h2>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card kpi3">
        <div>Model Accuracy</div>
        <h2>{accuracy*100:.2f}%</h2>
    </div>
    """, unsafe_allow_html=True)

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("🧍 Patient Details")

age = st.sidebar.number_input("Age", 1, 120, 45)
sex = st.sidebar.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.sidebar.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting BP", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar", [1, 0])
restecg = st.sidebar.selectbox("Rest ECG (0–2)", [0, 1, 2])
thalach = st.sidebar.number_input("Max Heart Rate", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Angina", [1, 0])
oldpeak = st.sidebar.number_input("ST Depression", 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox("Slope (0–2)", [0, 1, 2])
ca = st.sidebar.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

# ---------------- Prediction ----------------
if st.sidebar.button("🔍 Predict Risk"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.markdown("## 🧠 Prediction Result")

    if prediction[0] == 1:
        st.markdown("""
        <div class="result-high">
            🚨 High Risk of Heart Stroke<br>
            Please consult a doctor immediately.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-low">
            ✅ Low Risk of Heart Stroke<br>
            Keep maintaining a healthy lifestyle!
        </div>
        """, unsafe_allow_html=True)

# ---------------- Data Preview ----------------
with st.expander("📄 Dataset Preview"):
    st.dataframe(df.head())

st.caption("Developed with 💜 using Streamlit & Machine Learning")
