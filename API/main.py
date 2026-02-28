import os
import mlflow
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

repo_owner = os.getenv("DAGSHUB_USERNAME")
repo_name = "mlops_project"

if not repo_owner:
    raise RuntimeError("DAGSHUB_USERNAME not set")

mlflow.set_tracking_uri(
    f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
)

print("MLflow tracking URI configured.")


try:
    model = mlflow.pyfunc.load_model(
        "models:/water_potability_model/Production"
    )
    print("Production model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None


app = FastAPI(
    title="Water Potability Prediction API",
    description="API to predict whether water is potable or not.",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Water Potability API is running "}


@app.get("/health")
def health():
    return {"status": "healthy"}

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

@app.post("/predict")
def predict(water: Water):

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
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

        return {
            "prediction": result,
            "result": (
                "Water is Potable (Safe to drink)"
                if result == 1
                else "Water is NOT Potable (Not safe to drink)"
            )
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))