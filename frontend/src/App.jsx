import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import './index.css'

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [stats, setStats] = useState({ dataset_size: 0, features_count: 0, model_accuracy: 0 });
  const [loadingStats, setLoadingStats] = useState(true);
  
  const [formData, setFormData] = useState({
    age: 45, sex: 1, cp: 0, trestbps: 120, chol: 200,
    fbs: 0, restecg: 0, thalach: 150, exang: 0, oldpeak: 1.0,
    slope: 1, ca: 0, thal: 2
  });
  
  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState(null);
  const [averages, setAverages] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/stats`)
      .then(res => res.json())
      .then(data => {
        setStats(data);
        setLoadingStats(false);
      })
      .catch(err => {
        console.error("Failed to fetch stats:", err);
        setError("Could not connect to the backend API. Make sure FastAPI is running on port 8000.");
        setLoadingStats(false);
      });

    fetch(`${API_BASE_URL}/averages`)
      .then(res => res.json())
      .then(data => setAverages(data))
      .catch(err => console.error("Failed to fetch averages:", err));
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      // Parse floats for everything, HTML inputs return strings
      [name]: parseFloat(value) || 0
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setPredicting(true);
    setPrediction(null);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      
      if (!response.ok) throw new Error('Prediction request failed');
      
      const data = await response.json();
      if (data.error) throw new Error(data.error);
      
      // Artificial delay for smooth animated UX feeling
      setTimeout(() => {
        setPrediction(data);
        setPredicting(false);
        // Scroll to result softly
        window.scrollBy({ top: 300, behavior: 'smooth' });
      }, 600);
      
    } catch (err) {
      setError(err.message);
      setPredicting(false);
    }
  };

  let chartData = [];
  if (averages && prediction) {
    // Averages data structure is {"high_risk": {...}, "low_risk": {...}}
    chartData = [
      {
        name: 'Cholesterol',
        You: formData.chol,
        Healthy: Math.round(averages.low_risk.chol),
        'High Risk Avg': Math.round(averages.high_risk.chol),
      },
      {
        name: 'Max HR',
        You: formData.thalach,
        Healthy: Math.round(averages.low_risk.thalach),
        'High Risk Avg': Math.round(averages.high_risk.thalach),
      },
      {
        name: 'Resting BP',
        You: formData.trestbps,
        Healthy: Math.round(averages.low_risk.trestbps),
        'High Risk Avg': Math.round(averages.high_risk.trestbps),
      }
    ];
  }

  return (
    <div className="app-container">
      <h1 className="title">Heart Stroke Predictor</h1>
      <p className="subtitle">AI-powered system to estimate your heart stroke risk instantly.</p>

      {error && (
        <div style={{ background: 'var(--danger-light)', color: 'var(--danger)', padding: '1rem', borderRadius: '12px', marginBottom: '2rem', textAlign: 'center' }}>
          <strong>Error: </strong> {error}
        </div>
      )}

      {/* KPI Dashboard */}
      <div className="dashboard-grid">
        <div className="glass-card kpi-card purple">
          <div className="kpi-label">Dataset Size</div>
          <div className="kpi-value">{loadingStats ? '...' : stats.dataset_size}</div>
        </div>
        <div className="glass-card kpi-card green">
          <div className="kpi-label">Analyzed Features</div>
          <div className="kpi-value">{loadingStats ? '...' : stats.features_count}</div>
        </div>
        <div className="glass-card kpi-card orange">
          <div className="kpi-label">AI Accuracy</div>
          <div className="kpi-value">
            {loadingStats ? '...' : `${(stats.model_accuracy * 100).toFixed(2)}%`}
          </div>
        </div>
      </div>

      {/* Prediction Form */}
      <div className="glass-card">
        <h2 style={{ marginBottom: '1.5rem', textAlign: 'center' }}>Patient Health Profile</h2>
        
        <form onSubmit={handleSubmit} className="form-grid">
          <div className="input-group">
            <label className="input-label">Age</label>
            <input type="number" name="age" className="input-field" value={formData.age} onChange={handleInputChange} min="1" max="120" required />
          </div>
          
          <div className="input-group">
            <label className="input-label">Sex</label>
            <select name="sex" className="input-field" value={formData.sex} onChange={handleInputChange}>
              <option value={1}>Male</option>
              <option value={0}>Female</option>
            </select>
          </div>

          <div className="input-group">
            <label className="input-label">Chest Pain Type</label>
            <select name="cp" className="input-field" value={formData.cp} onChange={handleInputChange}>
              <option value={0}>Typical Angina</option>
              <option value={1}>Atypical Angina</option>
              <option value={2}>Non-anginal Pain</option>
              <option value={3}>Asymptomatic</option>
            </select>
          </div>

          <div className="input-group">
            <label className="input-label">Resting BP (mm Hg)</label>
            <input type="number" name="trestbps" className="input-field" value={formData.trestbps} onChange={handleInputChange} min="80" max="250" required />
          </div>

          <div className="input-group">
            <label className="input-label">Cholesterol (mg/dl)</label>
            <input type="number" name="chol" className="input-field" value={formData.chol} onChange={handleInputChange} min="100" max="600" required />
          </div>

          <div className="input-group">
            <label className="input-label">Fasting Blood Sugar &gt; 120</label>
            <select name="fbs" className="input-field" value={formData.fbs} onChange={handleInputChange}>
              <option value={1}>Yes (True)</option>
              <option value={0}>No (False)</option>
            </select>
          </div>

          <div className="input-group">
            <label className="input-label">Resting ECG</label>
            <select name="restecg" className="input-field" value={formData.restecg} onChange={handleInputChange}>
              <option value={0}>Normal</option>
              <option value={1}>ST-T Wave Abnormality</option>
              <option value={2}>Left Ventricular Hypertrophy</option>
            </select>
          </div>

          <div className="input-group">
            <label className="input-label">Max Heart Rate</label>
            <input type="number" name="thalach" className="input-field" value={formData.thalach} onChange={handleInputChange} min="60" max="220" required />
          </div>

          <div className="input-group">
            <label className="input-label">Exercise Induced Angina</label>
            <select name="exang" className="input-field" value={formData.exang} onChange={handleInputChange}>
              <option value={1}>Yes</option>
              <option value={0}>No</option>
            </select>
          </div>

          <div className="input-group">
            <label className="input-label">ST Depression</label>
            <input type="number" step="0.1" name="oldpeak" className="input-field" value={formData.oldpeak} onChange={handleInputChange} min="0" max="10" required />
          </div>

          <div className="input-group">
            <label className="input-label">Slope</label>
            <select name="slope" className="input-field" value={formData.slope} onChange={handleInputChange}>
              <option value={0}>Upsloping</option>
              <option value={1}>Flat</option>
              <option value={2}>Downsloping</option>
            </select>
          </div>

          <div className="input-group">
            <label className="input-label">Major Vessels</label>
            <select name="ca" className="input-field" value={formData.ca} onChange={handleInputChange}>
              <option value={0}>0</option>
              <option value={1}>1</option>
              <option value={2}>2</option>
              <option value={3}>3</option>
            </select>
          </div>

          <div className="input-group">
            <label className="input-label">Thalassemia</label>
            <select name="thal" className="input-field" value={formData.thal} onChange={handleInputChange}>
              <option value={0}>Normal</option>
              <option value={1}>Fixed Defect</option>
              <option value={2}>Reversable Defect</option>
              <option value={3}>Unknown</option>
            </select>
          </div>

          <button type="submit" className="btn-submit" disabled={predicting}>
            {predicting ? (
              <><span className="loader"></span> Analyzing health data...</>
            ) : (
              '🔍 Predict Risk'
            )}
          </button>
        </form>
      </div>

      {/* Result Display */}
      {prediction && (
        <div className="result-container">
          <div className={`result-card ${prediction.risk_level === 'High' ? 'result-high' : 'result-low'}`}>
            <h2 className="result-title">
              {prediction.risk_level === 'High' 
                ? '🚨 High Risk Detected' 
                : '✅ Low Risk'}
            </h2>
            <p className="result-message">
              {prediction.risk_level === 'High'
                ? 'Our AI model indicates a high risk for heart stroke given these parameters. Please consult a healthcare professional immediately.'
                : 'Your profile looks stable. Keep maintaining a healthy lifestyle, regular exercise, and balanced diet!'}
            </p>
          </div>

          {chartData.length > 0 && (
            <div className="glass-card" style={{ marginTop: '2rem' }}>
              <h3 style={{ textAlign: 'center', marginBottom: '1.5rem', color: '#fff' }}>How You Compare</h3>
              <div style={{ width: '100%', height: 350 }}>
                <ResponsiveContainer>
                  <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                    <XAxis dataKey="name" stroke="#cbd5e1" tickLine={false} axisLine={false} />
                    <YAxis stroke="#cbd5e1" tickLine={false} axisLine={false} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '12px' }}
                      itemStyle={{ color: '#fff', fontWeight: 500 }}
                      cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                    />
                    <Legend wrapperStyle={{ paddingTop: '20px' }} />
                    <Bar dataKey="You" fill="#8b5cf6" radius={[6, 6, 0, 0]} />
                    <Bar dataKey="Healthy" fill="#10b981" radius={[6, 6, 0, 0]} />
                    <Bar dataKey="High Risk Avg" fill="#ef4444" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default App
