from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from predict import predict_single

app = FastAPI(title="Churn Prediction API")

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict(customer: CustomerData):
    try:
        result = predict_single(customer.dict())
        return result                   
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
def health():
    return {"status": "ok"}
