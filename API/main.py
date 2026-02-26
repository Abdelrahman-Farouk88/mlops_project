import os
import mlflow
import dagshub
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ==============================
# Initialize DagsHub + MLflow (Token-based)
# ==============================

repo_owner = os.getenv("DAGSHUB_USERNAME")
repo_name = "mlops_project"
token = os.getenv("DAGSHUB_TOKEN")

if repo_owner and token:
    try:
        import dagshub

        # token-based auth (supported way)
        dagshub.auth.add_app_token(token)

        dagshub.init(
            repo_owner=repo_owner,
            repo_name=repo_name,
            mlflow=True
        )

        mlflow.set_tracking_uri(
            f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
        )

        print("✅ Dagshub initialized successfully.")

    except Exception as e:
        print(f"❌ Dagshub init failed: {e}")
else:
    print("⚠️ DAGSHUB credentials not found — running offline mode")


# ==============================
# FastAPI App
# ==============================

app = FastAPI(
    title="Water Potability Prediction API",
    description="API to predict whether water is potable or not.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# Load Production Model (once)
# ==============================

model = None

def load_model():
    global model

    try:
        client = mlflow.tracking.MlflowClient()

        versions = client.get_latest_versions(
            name="water_potability_model",
            stages=["Production"]
        )

        if not versions:
            print("⚠️ No Production model found in MLflow.")
            return None

        run_id = versions[0].run_id

        model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        print("✅ Production model loaded successfully.")
        return model

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


# load at startup (but failure won’t crash API)
load_model()


# ==============================
# Request Schema
# ==============================

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


# ==============================
# Routes
# ==============================

@app.get("/")
def home():
    return {"message": "Water Potability Prediction API is running 🚀"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def predict(water: Water):

    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

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
            "result": "Water is Potable (Safe to drink)" if result == 1
            else "Water is NOT Potable (Not safe to drink)"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))