# ⚡ Dynamic Line Rating (DLR) Ampacity Forecaster

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boosting-orange.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)

An industrial-grade Machine Learning pipeline that forecasts power grid transmission capacity (Ampacity) up to 72 hours in advance. By combining Numerical Weather Prediction (NWP) data with the **IEEE-738 thermodynamic physics standard** and **LightGBM Quantile Regression**, this tool safely unlocks hidden transmission capacity over conservative static grid ratings.

## 🚀 Key Results
* **Capacity Unlocked:** +481 to +522 Amps average gain over traditional Static Line Ratings (a >120% capacity increase).
* **High Accuracy:** Day-Ahead (24h) accuracy of 91.7% (8.2% MAPE). 72-hour accuracy of 89.0% (11.0% MAPE).
* **Grid Safety Limits:** Conformal-calibrated P10 quantile regression strictly enforces a ~80% confidence interval, keeping unsafe capacity overpredictions below 12.5%.

---

## 🏗️ Architecture

1. **Data Ingestion:** Automated fetching of ERA5 Reanalysis (Ground Truth) and Open-Meteo Previous-Runs NWP data.
2. **Physics Engine:** Heavily vectorized NumPy implementation of the IEEE-738 thermal model to compute baseline Ampacity.
3. **Feature Engineering:** Leave-one-out climatology (to prevent leakage) and cyclical trigonometric transformations (`sin/cos` + `arctan2`) to resolve circular wind-direction forecasting errors.
4. **Machine Learning:** LightGBM regressors trained for L1 point forecasting and P10/P50/P90 probabilistic bounds.
5. **Deployment:** Dockerized Streamlit application for real-time visualization.

---

## 💻 Installation & Usage

### Method 1: Fast Docker Deployment (Recommended)
Make sure Docker Desktop is running, then simply execute:
```bash
docker-compose up -d --build
```
Navigate to **http://localhost:8501** in your browser to view the interactive dashboard.

### Method 2: Local Python Environment
```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Or .\.venv\Scripts\Activate.ps1 on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the ML Pipeline (Downloads data, trains models, outputs to /results)
python -m dlr_planb.run

# 4. Launch the Streamlit Dashboard
streamlit run streamlit_app.py
```

---

## 🧪 Testing
The physics engine is strictly validated against textbook regression constraints.
```bash
pytest tests/
```

## 📁 Directory Structure
```text
DLR_PlanB/
│
├── dlr_planb/             # Core ML and Physics Python Package
│   ├── audit.py           # Data leakage prevention checks
│   ├── config.py          # Environment & Pipeline configurations
│   ├── data.py            # Open-Meteo API ETL pipelines
│   ├── evaluate.py        # Safety and accuracy metrics
│   ├── features.py        # Cyclical feature engineering & Climatology
│   ├── models.py          # LightGBM Quantile regressors
│   ├── physics.py         # Vectorized IEEE-738 Ampacity equations
│   └── run.py             # Main execution script
│
├── tests/                 # Pytest physics regression suite
├── Dockerfile             # Docker image configuration
├── docker-compose.yml     # Fast-deployment composition
├── streamlit_app.py       # Interactive web dashboard
└── requirements.txt       # Dependencies
```
