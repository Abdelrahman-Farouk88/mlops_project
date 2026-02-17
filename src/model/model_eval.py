import os
from dotenv import load_dotenv
import json
import pickle
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- MLflow tracking configuration (DagsHub when credentials available, local fallback otherwise) ---
REPO_OWNER = "Abdelrahman-Farouk88"
REPO_NAME = "mlops_project"

dagshub_token = os.getenv("DAGSHUB_TOKEN")

# Load environment variables from .env if present
load_dotenv()
dagshub_user = os.getenv("DAGSHUB_USERNAME")
dagshub_token = os.getenv("DAGSHUB_TOKEN")

if dagshub_user and dagshub_token:
    # Use DagsHub remote with embedded basic auth
    mlflow_tracking_uri = f"https://{dagshub_user}:{dagshub_token}@dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
    print("Using DagsHub MLflow tracking URI (with auth).")
else:
    # Local dev fallback -> keep experiments in ./mlruns
    local_mlruns = os.path.abspath("mlruns")
    os.makedirs(local_mlruns, exist_ok=True)
    if os.name == "nt":
        # On Windows use a filesystem path (no file://) to avoid MLflow rejecting it as a remote URI
        mlflow_tracking_uri = local_mlruns
    else:
        mlflow_tracking_uri = f"file://{local_mlruns}"
    print(f"DAGSHUB credentials not found. Falling back to local tracking at {local_mlruns}")

print("DEBUG: DAGSHUB_USERNAME present:", bool(dagshub_user))
print("DEBUG: DAGSHUB_TOKEN present:", bool(dagshub_token))

mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("DVC PIPELINE")
# --- end MLflow config ---

def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")


def load_model(filepath: str):
    try:
        with open(filepath, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {filepath}: {e}")


def evaluation_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
    try:
        params = yaml.safe_load(open("params.yaml", "r"))
        test_size = params["data_collection"]["test_size"]
        n_estimators = params["model_building"]["n_estimators"]

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_param("Test_size", test_size)
        mlflow.log_param("n_estimators", n_estimators)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Ensure reports/figures directory exists
        figures_dir = os.path.join("reports", "figures")
        os.makedirs(figures_dir, exist_ok=True)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {model_name}")
        cm_path = os.path.join(figures_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png")
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)

        metrics_dict = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error evaluating model: {e}")


def save_metrics(metrics: dict, metrics_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as file:
            json.dump(metrics, file, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {metrics_path}: {e}")


def main():
    try:
        test_data_path = "./data/processed/test_processed.csv"
        model_path = "models/model.pkl"
        metrics_path = "reports/metrics.json"
        # Use a single consistent artifact name (no spaces)
        model_name = "Best_Model"

        test_data = load_data(test_data_path)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_path)

        with mlflow.start_run() as run:
            metrics = evaluation_model(model, X_test, y_test, model_name)
            save_metrics(metrics, metrics_path)

            # log artifacts
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(metrics_path)
            mlflow.log_artifact(__file__)

            # infer signature and log model under the same artifact name
            signature = infer_signature(X_test, model.predict(X_test))
            mlflow.sklearn.log_model(
                sk_model=model,
                name=model_name,
                signature=signature
            )

            run_info = {'run_id': run.info.run_id, 'model_name': model_name}
            reports_path = "reports/run_info.json"
            os.makedirs(os.path.dirname(reports_path), exist_ok=True)
            with open(reports_path, 'w') as file:
                json.dump(run_info, file, indent=4)

    except Exception as e:
        # Raise a clearer exception for troubleshooting
        raise Exception(f"An Error occurred: {e}")


if __name__ == "__main__":
    main()