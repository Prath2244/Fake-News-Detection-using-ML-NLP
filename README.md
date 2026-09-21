# 🧠 NeuroCare – EEG-Based Depression Classification & Risk Assessment

[![Python](https://img.shields.io/badge/Python-3.14-3776ab?style=flat-square&logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.3-000000?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0.36-CC2927?style=flat-square&logo=database)](https://www.sqlalchemy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-3.0-150458?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![ReportLab](https://img.shields.io/badge/ReportLab-4.2.5-339999?style=flat-square)](https://www.reportlab.com/)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.x-7952B3?style=flat-square&logo=bootstrap)](https://getbootstrap.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

> A machine learning-powered web application for EEG-based depression severity classification. Uses Gradient Boosting and 56 EEG features to classify depression levels into 4 categories (Normal, Mild Risk, Moderate, Severe) with clinical-grade PDF report generation.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [Core Workflows](#-core-workflows)
- [API Endpoints](#-api-endpoints)
- [Classification Severity Levels](#-classification-severity-levels)
- [Model Performance](#-model-performance)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧾 Overview

**NeuroCare** is a clinical-grade EEG analysis platform that automatically classifies depression severity based on neurophysiological patterns. It combines machine learning with a user-friendly web interface to provide instant risk assessment, confidence scoring, and downloadable clinical reports.

The system analyzes **56 EEG features** across 14 brain electrodes (Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4) plus Heart Rate Variability (HRV), training a Gradient Boosting classifier to distinguish between Normal, Mild Risk, Moderate, and Severe depression levels with **99.75% accuracy**.

---

## ✨ Features

### 🔬 EEG Analysis & Classification
- **56 EEG Features** – Alpha/beta waves, mean amplitude, and standard deviation per electrode.
- **4-Class Classification** – Normal (0-30%), Mild Risk (40-65%), Moderate (65-85%), Severe (85-100%).
- **Multi-Electrode Support** – Comprehensive brain coverage with Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4.
- **HRV Integration** – Heart Rate Variability as an additional depression biomarker.
- **Real-Time Predictions** – Instant classification with confidence scoring.

### 📊 User Dashboard
- **Prediction History** – View all past EEG analyses with timestamps and severity levels.
- **Statistics Dashboard** – Track depression prevalence, level distribution, and HRV trends.
- **Confidence Visualization** – Risk probability bars with color-coded severity levels.
- **Level Breakdown** – Pie charts and statistics by depression category.

### 📋 Data Input Methods
- **Manual Form Entry** – Enter 56 EEG features + HRV directly in the web interface.
- **CSV File Upload** – Batch process EEG data from CSV files with auto-validation.
- **Auto-Fill from CSV** – Upload CSV and auto-populate form fields.
- **Validation** – Real-time error detection and helpful guidance.

### 📄 Clinical Reports
- **PDF Generation** – Download clinical-grade reports with prediction results.
- **Report Contents** – Patient info, EEG data summary, classification result, confidence score, and wellness recommendations.
- **Professional Formatting** – ReportLab-powered PDF with charts and structured layout.

### 🔐 User Management & Security
- **User Authentication** – Secure login/signup with password hashing.
- **Session Management** – Flask-Login for secure user sessions.
- **Role-Based Access** – User-specific prediction history and profile.
- **Data Privacy** – All predictions tied to authenticated users.

### 📈 Analytics & Insights
- **Prediction Logs** – Automatically log depression vs. non-depression predictions.
- **Export Options** – Download full prediction history as CSV.
- **Trend Analysis** – Track patient patterns over time.

---

## 🛠️ Tech Stack

| Category          | Technology                        |
|-------------------|-----------------------------------|
| Backend           | Python 3.14, Flask 3.0.3          |
| ORM               | SQLAlchemy 2.0.36, Flask-SQLAlchemy 3.1.1 |
| Authentication    | Flask-Login 0.6.3                 |
| ML/AI             | scikit-learn 1.9.1, joblib 1.4.2  |
| Data Processing   | Pandas 3.0.6, NumPy 2.5.3         |
| Report Generation | ReportLab 4.2.5                   |
| Database          | SQLite (default)                  |
| Frontend          | Bootstrap 5, HTML5, CSS3, JavaScript |
| Web Server        | Werkzeug 3.0.3                    |

---

## 🧱 Architecture

- **Backend** – Flask with modular route handlers; Gradient Boosting Classifier for multi-class prediction; StandardScaler for feature normalization.
- **Database** – SQLite with SQLAlchemy ORM; User model with hashed passwords; Prediction model for history tracking.
- **ML Pipeline** – Scikit-learn pipeline combining StandardScaler + GradientBoostingClassifier; trained on balanced 4-class synthetic EEG data.
- **Data Processing** – Pandas for CSV parsing and feature extraction; NumPy for numerical operations.
- **Report Generation** – ReportLab for PDF creation with formatted clinical reports.
- **Frontend** – Bootstrap 5 responsive design; form validation; real-time error feedback; color-coded severity indicators.

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.14+
- pip (Python package manager)
- ~500 MB disk space

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/neurocare.git
cd neurocare
```

### 2. Create virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate synthetic training data (first time only)
```bash
python create_synthetic_training_data.py
```
This creates `EEG_4class_training.csv` with 2000 balanced samples across 4 depression classes.

### 5. Train the ML model
```bash
python retrain_model.py
```
Output: `best_EEG_model.pkl` and `model.pkl` with 99.75% accuracy.

---

## 🚀 Running the Application

### 1. Start the Flask server
```bash
python app.py
```
Server runs on `http://127.0.0.1:5000` in development mode.

### 2. Access the application
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### 3. Create an account
- Click **Sign Up**
- Enter name, email, and password
- Click **Create Account**

### 4. Login
- Enter your email and password
- Click **Login**

### 5. Make predictions
**Option A – Manual Entry:**
- Click **Predict**
- Enter EEG values for all 14 electrodes (alpha, beta, mean, std) + HRV
- Click **Analyze**

**Option B – CSV Upload:**
- Click **Upload CSV**
- Select a CSV file from `sample_files/`
- Review auto-populated values
- Click **Analyze**

### 6. View results
- See prediction: **Normal | Mild Risk | Moderate | Severe**
- Check confidence: **0-30% | 40-65% | 65-85% | 85-100%**
- Review severity color: **Green | Yellow | Orange | Red**
- Download PDF report
- View prediction history

---

## 🔄 Core Workflows

| Actor | Action           | Description                                                   |
|-------|------------------|---------------------------------------------------------------|
| User  | Sign Up          | Creates account with name, email, password.                   |
| User  | Login            | Authenticates with email/password; redirects to dashboard.    |
| User  | Manual Predict   | Enters 56 EEG features + HRV; model returns 4-class result.   |
| User  | CSV Predict      | Uploads CSV file; auto-fills fields; submits for prediction.  |
| User  | View History     | Lists all past predictions with timestamps and severity.       |
| User  | Download Report  | Generates and downloads clinical PDF for a prediction.        |
| User  | Export CSV       | Downloads full prediction history as CSV file.                |
| User  | View Dashboard   | Sees statistics: total predictions, level breakdown, trends.  |
| User  | Update Profile   | Changes name or password.                                      |
| Admin | View Logs        | Accesses depression/non-depression prediction logs.           |
| Admin | Retrain Model    | Runs `retrain_model.py` to update classifier.                 |

---

## 📡 API Endpoints

### Authentication

| Method | Endpoint       | Description      | Access  |
|--------|----------------|------------------|---------|
| POST   | `/signup`      | Register user    | Public  |
| POST   | `/login`       | Login user       | Public  |
| GET    | `/logout`      | Logout user      | Authed  |

### Predictions

| Method | Endpoint                 | Description                     | Access  |
|--------|--------------------------|----------------------------------|---------|
| GET    | `/predict`               | Get prediction form              | Authed  |
| POST   | `/predict`               | Submit EEG data for analysis     | Authed  |
| POST   | `/upload_csv_predict`    | Upload CSV for auto-fill         | Authed  |

### User Dashboard

| Method | Endpoint       | Description                  | Access  |
|--------|----------------|------------------------------|---------|
| GET    | `/dashboard`   | View user dashboard + stats  | Authed  |
| GET    | `/history`     | View prediction history      | Authed  |

### Reports & Export

| Method | Endpoint                      | Description                    | Access  |
|--------|-------------------------------|--------------------------------|---------|
| GET    | `/download_report/<pred_id>`  | Download PDF clinical report   | Authed  |
| GET    | `/export_csv`                 | Export all predictions as CSV   | Authed  |

### Logs & Admin

| Method | Endpoint  | Description                 | Access  |
|--------|-----------|-----------------------------|----|
| GET    | `/logs`   | View depression/non-depression logs | Authed  |

---

## 🎯 Classification Severity Levels

| Level      | Probability | Color  | EEG Pattern                          | Recommendation              |
|------------|-------------|--------|--------------------------------------|---------------------------|
| Normal     | 0-30%       | 🟢     | High alpha/beta, low theta/delta    | Maintain healthy lifestyle |
| Mild Risk  | 40-65%      | 🟡     | Moderate alpha/beta reduction       | Monitor mood, light exercise |
| Moderate   | 65-85%      | 🟠     | Significant alpha/beta reduction    | Consult healthcare provider |
| Severe     | 85-100%     | 🔴     | Very low alpha/beta, high theta     | Seek immediate professional help |

**Note:** This tool is for screening purposes only and should not replace professional medical diagnosis.

---

## 📊 Model Performance

- **Accuracy:** 99.75% on test set
- **Training Samples:** 2000 (500 per class, balanced)
- **Test Samples:** 400 (100 per class)
- **Algorithm:** Gradient Boosting Classifier
  - n_estimators: 150
  - max_depth: 4
  - learning_rate: 0.1
- **Features:** 56 EEG features + 1 HRV feature
- **Preprocessing:** StandardScaler normalization

---

## 🗺️ Roadmap

- [x] 4-class depression severity classification
- [x] EEG feature extraction (56 features × 14 electrodes)
- [x] Manual form entry with validation
- [x] CSV file upload and auto-fill
- [x] Prediction history tracking
- [x] Clinical PDF report generation
- [x] User authentication and session management
- [x] Dashboard with statistics and visualizations
- [x] Prediction logging (depression vs. non-depression)
- [x] CSV export functionality
- [x] Color-coded severity indicators
- [x] Confidence probability mapping per class
- [ ] Real-time EEG stream processing
- [ ] Multi-language support
- [ ] Mobile app (React Native)
- [ ] Advanced time-series analysis
- [ ] Longitudinal trend prediction
- [ ] Integration with wearable EEG devices (Muse, Emotiv)
- [ ] Email report delivery
- [ ] Patient portal for clinicians
- [ ] API for third-party integrations

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Open a Pull Request.

Please ensure your code follows PEP 8 style guidelines and includes appropriate comments.

---

## ⚠️ Disclaimer

**NeuroCare is a research and screening tool only.** It is not intended to diagnose, treat, or prevent any medical condition. Always consult with a qualified healthcare professional for medical advice. The model's predictions should be used as supplementary information, not a replacement for professional psychiatric evaluation.

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Prathmesh** – [GitHub](https://github.com/yourusername) · [LinkedIn](https://linkedin.com/in/yourusername)

---

**Made with ❤️ for mental health research and screening.**
