import os
import pandas as pd
import mlflow
import threading
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

ENV = os.getenv("ENV", "development")

repo_owner = os.getenv("DAGSHUB_USERNAME")
repo_name = "mlops_project"
token = os.getenv("DAGSHUB_TOKEN")


if ENV == "development" and repo_owner and token:
    try:
        import dagshub

        dagshub.auth.add_app_token(token)
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

        tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)

        print("MLflow tracking (dev):", tracking_uri)

    except Exception as e:
        print(f"MLflow init failed (dev): {e}")
else:
    print("Production mode — MLflow disabled")


app = FastAPI(
    title="Water Potability Prediction API",
    description="API to predict whether water is potable or not.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None


def load_model():
    print("Starting model loading..................................................................")
    global model

    try:
        if ENV == "development":
            client = mlflow.tracking.MlflowClient()

            versions = client.get_latest_versions(
                name="water_potability_model",
                stages=["Production"]
            )

            if not versions:
                print("No Production model found in registry.")
                return

            run_id = versions[0].run_id
            model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

            print("Model loaded from MLflow (dev)")

        else:
            if os.path.exists("model"):
                model = mlflow.pyfunc.load_model("model")
                print("Model loaded from local artifact (prod)")
            else:
                print("No local model artifact found in production")

    except Exception as e:
        print(f"Error loading model: {e}")
    
    print("🚀 Starting background model loading............................................................")
    


@app.on_event("startup")
def startup():
    print("🚀 Starting background model loading")
    threading.Thread(target=load_model).start()


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
    return {"message": "Water Potability Prediction API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def predict(water: Water):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        sample = pd.DataFrame([water.dict()])

        prediction = model.predict(sample)
        result = int(prediction[0])

        return {
            "prediction": result,
            "result": "Water is Potable (Safe to drink)"
            if result == 1
            else "Water is NOT Potable (Not safe to drink)"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))