from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from app.schemas import CustomerInput


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "model.joblib"
)


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Model not found. Run python scripts/train.py"
        )

    # Only load artifacts created by trusted code.
    return joblib.load(MODEL_PATH)


def predict_churn(customer: CustomerInput) -> dict:
    model = load_model()

    data = pd.DataFrame([customer.model_dump()])

    probability = float(
        model.predict_proba(data)[0][1]
    )

    prediction = int(probability >= 0.5)

    if probability >= 0.7:
        risk = "high"
    elif probability >= 0.4:
        risk = "medium"
    else:
        risk = "low"

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_level": risk,
    }