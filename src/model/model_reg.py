import json
from dotenv import load_dotenv
import os
import sys
from urllib.parse import urlparse, urlunparse

import mlflow
from mlflow.tracking import MlflowClient
import mlflow.exceptions

load_dotenv()
REPO_OWNER = "Abdelrahman-Farouk88"
REPO_NAME = "mlops_project"

dagshub_user = os.getenv("DAGSHUB_USERNAME")
dagshub_token = os.getenv("DAGSHUB_TOKEN")

if dagshub_user and dagshub_token:
    tracking_uri = f"https://{dagshub_user}:{dagshub_token}@dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
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

reports_path = "reports/run_info.json"
if not os.path.exists(reports_path):
    print(f"ERROR: {reports_path} not found")
    sys.exit(1)

with open(reports_path, 'r') as file:
    run_info = json.load(file)

run_id = run_info.get('run_id')
model_name = run_info.get('model_name', "Best_Model")  # fallback

if not run_id:
    print("ERROR: run_id not found in run_info.json")
    sys.exit(1)

registered_name = model_name.replace(" ", "_")

client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())

model_uri = f"runs:/{run_id}/{model_name}"

print(f"Attempting to register model from: {model_uri} as registry name: {registered_name}")

try:
    client.create_registered_model(registered_name)
    print(f"Successfully created registered model '{registered_name}'.")
except mlflow.exceptions.MlflowException as e:
    msg = str(e)
    if "RESOURCE_ALREADY_EXISTS" in msg or "ALREADY_EXISTS" in msg:
        print(f"Registered model '{registered_name}' already exists. Creating a new version...")
    else:
        print("ERROR creating registered model:", msg)
        raise

try:
    reg = client.create_model_version(
        name=registered_name,
        source=model_uri,
        run_id=run_id,
    )
except Exception as e:
    print("ERROR creating model version:", e)
    print(" - Check that the Run ID exists and that the artifact path is correct.")
    print(" - If authentication failed, ensure DAGSHUB_USERNAME and DAGSHUB_TOKEN are set in the runner.")
    raise

model_version = reg.version
new_stage = "Staging"

try:
    client.transition_model_version_stage(
        name=registered_name,
        version=model_version,
        stage=new_stage,
        archive_existing_versions=True
    )
    print(f"Model {registered_name} version {model_version} transitioned to {new_stage} stage.")
except Exception as e:
    print("ERROR transitioning model stage:", e)
    raise