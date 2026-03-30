<div align="center">
  <h1>💜 Heart Stroke Prediction AI</h1>
  <p><strong>A stunning, AI-powered web dashboard to estimate heart stroke risk instantly.</strong></p>

  <!-- Badges -->
  <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Vite-B73BFE?style=for-the-badge&logo=vite&logoColor=FFD62E" />
</div>

<br />

## 🌟 Overview
The **Heart Stroke Prediction AI** is a state-of-the-art predictive healthcare application designed to evaluate a patient's risk of stroke using advanced artificial intelligence techniques. The system is built utilizing a robust `RandomForestClassifier` trained on historical health data. It integrates a high-performance Python/FastAPI backend with a visually striking, glassmorphism-styled React frontend dashboard, forming an end-to-end full-stack AI application.

## 📁 Project Architecture
This repository follows a clean, decoupled client-server microservices structure:

```text
AIML_PROJECT/
├── frontend/             # Stunning React UI application (Vite)
│   ├── src/
│   │   ├── components/   # Reusable UI components (Modals, Charts)
│   │   ├── App.jsx       # Main application layout & state logic
│   │   ├── index.css     # Global glassmorphism UI styles
│   │   └── main.jsx      # React DOM entry point
│   ├── package.json      # Frontend package metadata and scripts
│   └── vite.config.js    # Vite builder configuration
└── backend/              # FastAPI server & ML logic
    ├── api.py            # Main API controller/server instance
    ├── heart.csv         # Core dataset for model training
    └── requirements.txt  # Python backend dependencies
```

## ✨ Key Features & Technical Details

### Model Pipeline & Backend (`FastAPI`)
- **API Endpoints**:
  - `GET /stats`: Summarizes the loaded dataset properties and current model accuracy.
  - `GET /averages`: Computes runtime descriptive statistics for `high_risk` vs `low_risk` demographic groups.
  - `POST /predict`: Handles inbound JSON arrays mapping all 13 core clinical features and invokes the Random Forest `predict` routine securely.
- **In-Memory ML Strategy**: The backend caches the StandardScaler transform state and Random Forest model estimators in memory upon `FastAPI` application startup (`@app.on_event("startup")`) to ensure maximum inference speeds and to prevent constant I/O bottlenecks.
- **Data Preprocessing**: It performs immediate standard scaling on features using `scikit-learn`'s `StandardScaler`, splitting the subset into strict unskewed training and testing populations to measure internal `accuracy_score`.
- **CORS Handling**: Ready out of the box with `CORSMiddleware` interceptors configured to bridge standard browser origin mechanisms between React's dev server and Uvicorn.

### Interactive Application UI (`React & Vite`)
- **Vite & HMR**: Powered by `Vite.js` for ultra-fast bundling and immediate Hot Module Replacement.
- **Micro-Animations & Glassmorphism Design System**: Built with strict CSS standards utilizing frosted-glass effects (blur backdrop filters), gradients, precise typographic hierarchy, modern layouts (Grid/Flexbox), and layered drop-shadows to ensure high user engagement. 
- **Recharts Dynamic Rendering**: Incorporates the flexible `recharts` React data viz library. Real-time patient inputs organically update graphs showing "How You Compare" against backend averages.
- **Responsiveness**: Smoothly handles dynamic viewports with adaptive CSS boundaries.

## 🧠 Clinical Features Examined
To ensure high accuracy, the model ingests 13 standard physiological descriptors for evaluation:
1. `age` (Years)
2. `sex` (Binary index)
3. `cp` (Chest pain type grading)
4. `trestbps` (Resting blood pressure)
5. `chol` (Serum cholesterol in mg/dl)
6. `fbs` (Fasting blood sugar > 120 mg/dl indicator)
7. `restecg` (Resting electrocardiographic results)
8. `thalach` (Maximum heart rate achieved)
9. `exang` (Exercise induced angina variable)
10. `oldpeak` (ST depression relative to rest)
11. `slope` (Slope of the peak exercise ST segment)
12. `ca` (Number of major vessels colored by fluoroscopy)
13. `thal` (Thalassemia diagnostic factor)

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Node.js 18+

### 1. Start the Backend API
The backend dynamically learns distributions, establishes base conditions, calculates mathematical bounds and runs the AI instance.

```bash
# Navigate to the backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server natively via Uvicorn
python -m uvicorn api:app --reload --port 8000
```
> The auto-generated API swagger blueprint documentation will be instantly available at `http://localhost:8000/docs`

### 2. Start the Frontend Application
The React application maps all graphical forms to the local inference API.

```bash
# Navigate to the frontend directory
cd frontend

# Install exact node modules (only required the first time)
npm install

# Start the Vite development and Hot Reload server
npm run dev
```
> The dashboard portal UI will be provisioned on network localhost ports at `http://localhost:5173`

## 🤝 Contributing
Contributions, enhancements, bug-fixes, or feature requests are highly welcome! Feel free to checkout out the issues page or submit a standardized Pull Request.

---
<div align="center">
  <i>Developed with 💜 to assist in proactive health monitoring.</i>
</div>
