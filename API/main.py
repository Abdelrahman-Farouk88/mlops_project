import mlflow
import dagshub
import pandas as pd
import os

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Water Potability Prediction API",
    description="API to predict whether water is potable or not.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dagshub.init(
    repo_owner="Abdelrahman-Farouk88",
    repo_name="mlops_project",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/Abdelrahman-Farouk88/mlops_project.mlflow"
)

def load_model():
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(
        "water_potability_model",
        stages=["Production"]
    )

    if not versions:
        raise Exception("No Production model found.")

    run_id = versions[0].run_id
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    return model


model = load_model()



class Water(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float


@app.get("/")
def home():
    return {"message": "Water Potability Prediction API is running."}


@app.post("/predict")
def predict(water: Water):

    sample = pd.DataFrame({
        "ph": [water.ph],
        "Hardness": [water.Hardness],
        "Solids": [water.Solids],
        "Chloramines": [water.Chloramines],
        "Sulfate": [water.Sulfate],
        "Conductivity": [water.Conductivity],
        "Organic_carbon": [water.Organic_carbon],
        "Trihalomethanes": [water.Trihalomethanes],
        "Turbidity": [water.Turbidity]
    })

    prediction = model.predict(sample)

    result = int(prediction[0])

    if result == 1:
        return {
            "prediction": result,
            "result": "Water is Potable (Safe to drink)"
        }
    else:
        return {
            "prediction": result,
            "result": "Water is NOT Potable (Not safe to drink)"
        }