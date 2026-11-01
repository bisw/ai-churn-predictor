from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.predictor import load_model, predict_churn
from app.schemas import (
    CustomerInput,
    PredictionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail at startup if the trained model is unavailable.
    load_model()
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/")
def home():
    return {
        "message": "Welcome to AI Churn Prediction API",
        "status": "running",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(customer: CustomerInput):
    return predict_churn(customer)