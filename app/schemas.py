from typing import Literal

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=100000)
    support_calls: int = Field(ge=0, le=100)
    contract_type: Literal["monthly", "annual"]


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str