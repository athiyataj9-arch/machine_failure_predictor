import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # 1. Load Data
    print("Loading Dataset...")
    df = pd.read_csv('factory_sensor_simulator_2040.csv')

    # 2. Preprocessing & Feature Selection
    target = 'Failure_Within_7_Days'
    X = df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
    y = df[target].astype(int)
    feature_names = X.columns.tolist()

    # 3. Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Model
    print("Training XGBoost AI Brain...")
    model = XGBClassifier(eval_metric='logloss', n_estimators=100, max_depth=6)
    model.fit(X_train_scaled, y_train)

    # 5. --- EVALUATION & SAVING GRAPHS AS FILES ---
    print("\n Generating and Saving Model Insights...")
    y_pred = model.predict(X_test_scaled)
    
    # Print Text Report to Terminal
    print(classification_report(y_test, y_pred))

    # GRAPH A: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Prediction Accuracy: Actual vs Predicted')
    plt.ylabel('Real Status')
    plt.xlabel('AI Guess')
    plt.savefig('confusion_matrix.png') # Saves the file to your folder
    plt.close() # Clears the plot for the next one
    print("Saved: confusion_matrix.png")

    # GRAPH B: Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette="viridis")
    plt.title('Which Sensors Predict Failure Best?')
    plt.xlabel('Relative Importance')
    plt.tight_layout() # Ensures labels aren't cut off
    plt.savefig('feature_importance.png') # Saves the file to your folder
    plt.close()
    print("Saved: feature_importance.png")

    # 6. Export Model Files
    print("Saving AI Assets...")
    model.save_model("machine_model.json")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(feature_names, "model_features.pkl")
    
    print("✅ Success! All files (Model and Graphs) are in your folder.")

if __name__ == "__main__":
    main()