from setuptools import setup, find_packages

setup(
    name="machine_anomaly_detection",
    version="1.0.0",
    author="Athiya Taj",
    description="An AI-powered dashboard to predict industrial machine failures using XGBoost.",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "streamlit",
        "joblib",
        "matplotlib",
        "seaborn"
    ],
    python_requires=">=3.8",
)