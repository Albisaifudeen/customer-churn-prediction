import joblib
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def predict_single(input_dict: dict):
    model = joblib.load(os.path.join(BASE_DIR, "models", "xgb_churn.pkl"))
    encoders = joblib.load(os.path.join(BASE_DIR, "models", "encoders.pkl"))

    df = pd.DataFrame([input_dict])
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    proba = float(model.predict_proba(df)[0][1])

    return {
        "churn_probability": round(proba, 4),
        "will_churn": bool(proba > 0.5),
        "verdict": "⚠️ Customer WILL CHURN" if proba > 0.5 else "✅ Customer will STAY"
    }