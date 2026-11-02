# 🚇 MetroGuard AI: Transit Infrastructure Monitoring

![GitHub CI](https://github.com/your-username/MetroGuardAI/actions/workflows/ci.yml/badge.svg)
![SonarCloud Quality Gate](https://sonarcloud.io/github/your-username/MetroGuardAI/badge.svg)

**MetroGuard AI** is a professional-grade anomaly detection and predictive maintenance system designed for transit infrastructure. By combining deep learning (LSTM Autoencoders) and ensemble machine learning (XGBoost), the system identifies subtle patterns of degradation in sensor data before they lead to critical failures.

## 🌟 Key Features
- **Hybrid AI Engine**: Dual-model approach using an LSTM Autoencoder for reconstruction error and XGBoost for binary anomaly classification.
- **Health Scoring**: Real-time calculation of system health (0-100%) based on anomaly trends.
- **Root Cause Analysis**: Identifies which specific sensors are contributing most to a detected anomaly.
- **Persistence Filtering**: Reduces false positives by requiring consecutive anomalies before triggering alerts.
- **Professional Pipeline**: Integrated CI/CD with linting, security scanning, and data validation.

## 🛠️ Technical Architecture
The project follows a modular software engineering pattern:
- `src/`: Core logic (Data loading, Preprocessing, Model Inference).
- `training/`: Model training and evaluation pipelines.
- `pages/`: Multi-page Streamlit dashboard.
- `models/`: Versioned model artifacts and configuration.
- `tests/`: Unit and integration tests ensuring pipeline stability.

### Model Stack
- **LSTM Autoencoder**: Captures temporal dependencies in sensor sequences.
- **XGBoost**: Provides high-precision anomaly classification.
- **Pandera**: Ensures strict input data validation.
- **StandardScaler**: Normalizes sensor data for neural network stability.

## 🚀 Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/MetroGuardAI.git
   cd MetroGuardAI
   ```
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the System
1. **Evaluate Models** (Generates metrics and score files):
   ```bash
   PYTHONPATH=. python training/evaluate.py
   ```
2. **Launch Dashboard**:
   ```bash
   PYTHONPATH=. streamlit run app.py
   ```

## 📈 Quality Assurance
This project implements industry-standard MLOps practices:
- **CI Pipeline**: Automated testing via GitHub Actions (`ruff`, `pytest`, `bandit`).
- **Data Validation**: Strict schema enforcement using `pandera`.
- **Security**: Static analysis for security vulnerabilities.
- **Maintainability**: Fully type-hinted codebase following PEP 8 standards.

---
Developed for the future of smart transit safety.
