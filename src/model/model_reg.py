import os
import sys
import json
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv

import mlflow
from mlflow.tracking import MlflowClient
import mlflow.exceptions

REPO_OWNER = "Abdelrahman-Farouk88"
REPO_NAME = "mlops_project"

RUN_INFO_PATH = Path("reports/run_info.json")

def setup_mlflow_tracking() -> str:
    """
    Sets tracking URI to DagsHub if credentials exist.
    Otherwise uses public DagsHub endpoint (read-only).
    """
    load_dotenv()

    dagshub_user = os.getenv("DAGSHUB_USERNAME")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")

    if dagshub_user and dagshub_token:
        tracking_uri = (
            f"https://{dagshub_user}:{dagshub_token}"
            f"@dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
        )
    else:
        tracking_uri = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"

    mlflow.set_tracking_uri(tracking_uri)

    try:
        parsed = urlparse(mlflow.get_tracking_uri())
        host_only = urlunparse(parsed._replace(netloc=parsed.hostname or ""))
    except Exception:
        host_only = mlflow.get_tracking_uri()

    print("DEBUG: DAGSHUB_USERNAME present:", bool(dagshub_user))
    print("DEBUG: DAGSHUB_TOKEN present:", bool(dagshub_token))
    print("DEBUG: MLflow tracking host:", host_only)

    return tracking_uri


def load_run_info(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run evaluation stage first.")
    with open(path, "r") as f:
        return json.load(f)


def ensure_registered_model(client: MlflowClient, name: str) -> None:
    try:
        client.create_registered_model(name)
        print(f"Created registered model: {name}")
    except mlflow.exceptions.MlflowException as e:
        msg = str(e)
        if "RESOURCE_ALREADY_EXISTS" in msg or "ALREADY_EXISTS" in msg:
            print(f"Registered model already exists: {name}")
        else:
            raise


def main():
    setup_mlflow_tracking()

    run_info = load_run_info(RUN_INFO_PATH)

    run_id = run_info.get("run_id")
    artifact_path = run_info.get("mlflow_model_artifact_path", "model")
    registered_name = run_info.get("registered_model_name", "water_potability_model")

    if not run_id:
        print("ERROR: run_id missing in reports/run_info.json")
        sys.exit(1)

    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())

    try:
        client.get_run(run_id)
    except Exception:
        print(f"ERROR: Run ID does not exist or is not accessible: {run_id}")
        sys.exit(1)

    model_uri = f"runs:/{run_id}/{artifact_path}"

    print(f"Registering model from: {model_uri}")
    print(f"Registry name: {registered_name}")

    ensure_registered_model(client, registered_name)

    try:
        mv = client.create_model_version(
            name=registered_name,
            source=model_uri,
            run_id=run_id,
        )
    except Exception as e:
        print("ERROR creating model version:", e)
        print("Common causes:")
        print(" - Wrong artifact path (should be 'model')")
        print(" - Authentication failed (check DAGSHUB_USERNAME and DAGSHUB_TOKEN)")
        print(" - Registry not supported in this tracking backend")
        raise

    version = mv.version
    print(f"Created model version: {version}")

    try:
        client.transition_model_version_stage(
            name=registered_name,
            version=version,
            stage="Staging",
            archive_existing_versions=True,
        )
        print(f"Model {registered_name} v{version} moved to Staging.")
    except Exception as e:
        print("ERROR transitioning stage:", e)
        raise


if __name__ == "__main__":
    main()
