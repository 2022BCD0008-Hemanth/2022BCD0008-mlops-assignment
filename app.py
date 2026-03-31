from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from typing import List

app = FastAPI()
model = joblib.load("models/model.pkl")

# Define input schema
class InputData(BaseModel):
    features: List[float]

@app.get("/health")
def health():
    return {
        "name": "Hemanth Sai Manikanta Appari",
        "roll_no": "2022BCD0008"
    }

@app.post("/predict")
def predict(data: InputData):
    pred = model.predict([data.features]).tolist()
    return {
        "prediction": pred,
        "name": "Hemanth Sai Manikanta Appari",
        "roll_no": "2022BCD0008"
    }