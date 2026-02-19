import os
import json
import pickle
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dotenv import load_dotenv

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

REPO_OWNER = "Abdelrahman-Farouk88"
REPO_NAME = "mlops_project"

EXPERIMENT_NAME = "DVC PIPELINE"

TEST_DATA_PATH = Path("data/processed/test_processed.csv")
MODEL_PATH = Path("models/model.pkl")

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"
RUN_INFO_PATH = REPORTS_DIR / "run_info.json"

MLFLOW_MODEL_ARTIFACT_PATH = "model"


def load_params(params_path: str = "params.yaml") -> dict:
    try:
        with open(params_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read params from {params_path}: {e}")


def setup_mlflow_tracking() -> str:
    load_dotenv()

    dagshub_user = os.getenv("DAGSHUB_USERNAME")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")

    if dagshub_user and dagshub_token:
        tracking_uri = (
            f"https://{dagshub_user}:{dagshub_token}"
            f"@dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
        )
        print("Using DagsHub MLflow tracking URI (authenticated).")
    else:
        local_mlruns = Path("mlruns").resolve()
        local_mlruns.mkdir(parents=True, exist_ok=True)

        if os.name == "nt":
            tracking_uri = str(local_mlruns)
        else:
            tracking_uri = f"file://{local_mlruns}"

        print(f"DagsHub credentials not found. Using local tracking at {local_mlruns}")

    print("DEBUG: DAGSHUB_USERNAME present:", bool(dagshub_user))
    print("DEBUG: DAGSHUB_TOKEN present:", bool(dagshub_token))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    return tracking_uri


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Test data not found: {path}")
    return pd.read_csv(path)


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "Potability" not in df.columns:
        raise ValueError("Target column 'Potability' not found in test dataset.")
    X = df.drop(columns=["Potability"])
    y = df["Potability"]
    return X, y


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def plot_confusion_matrix(y_true, y_pred, out_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    setup_mlflow_tracking()

    params = load_params("params.yaml")

    test_size = params.get("data_collection", {}).get("test_size", None)
    n_estimators = params.get("model_building", {}).get("n_estimators", None)

    df_test = load_data(TEST_DATA_PATH)
    X_test, y_test = prepare_data(df_test)

    model = load_model(MODEL_PATH)

    with mlflow.start_run() as run:
        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)

        if test_size is not None:
            mlflow.log_param("test_size", test_size)
        if n_estimators is not None:
            mlflow.log_param("n_estimators", n_estimators)

        mlflow.log_metrics(metrics)

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        cm_path = FIGURES_DIR / "confusion_matrix.png"
        plot_confusion_matrix(
            y_test, y_pred, cm_path, title="Confusion Matrix (Test Set)"
        )
        mlflow.log_artifact(str(cm_path))

        save_json(metrics, METRICS_PATH)
        mlflow.log_artifact(str(METRICS_PATH))

        mlflow.log_artifact(__file__)

        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=MLFLOW_MODEL_ARTIFACT_PATH,
            signature=signature,
        )

        run_info = {
            "run_id": run.info.run_id,
            "mlflow_model_artifact_path": MLFLOW_MODEL_ARTIFACT_PATH,
            "registered_model_name": "water_potability_model",
        }
        save_json(run_info, RUN_INFO_PATH)

        print("Evaluation finished.")
        print("Run ID:", run.info.run_id)
        print("Saved run info to:", RUN_INFO_PATH)


if __name__ == "__main__":
    main()
