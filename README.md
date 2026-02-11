# 🛠️ Machine Failure Predictor

A machine learning application that predicts potential equipment failures using sensor data. This project includes a trained model, a data processing pipeline, and a web-based dashboard built with Streamlit.



## 🚀 Features
* **Predictive Modeling:** Uses XGBoost to identify failure patterns with high precision.
* **Interactive Dashboard:** A Streamlit web app for real-time manual data input and prediction.
* **Data Visualization:** Built-in charts for feature importance and model performance.

## 📁 Project Structure
* `app.py`: The main Streamlit web application.
* `model_building.ipynb`: Data exploration and training logic.
* `machine_model.json`: The serialized trained model.
* `requirements.txt`: List of dependencies needed to run the project.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/athiyataj9-arch/machine_failure_predictor.git](https://github.com/athiyataj9-arch/machine_failure_predictor.git)
   cd machine_failure_predictor

2. **Create and activate a virtual environment:**
# Create environment
python -m venv venv

# Activate on Windows:
.\venv\Scripts\activate

3. **Install dependencies:**
pip install -r requirements.txt

4. **Launch the app:**
streamlit run app.py

## 📊 Results
The model was evaluated using a confusion matrix and achieved high accuracy in identifying specific failure modes, particularly:

Heat Dissipation Failures

Overstrain Failures
