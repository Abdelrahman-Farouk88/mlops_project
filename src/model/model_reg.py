import json
from dotenv import load_dotenv
import os
import sys
from urllib.parse import urlparse, urlunparse

import mlflow
from mlflow.tracking import MlflowClient
import mlflow.exceptions

load_dotenv()
# Configure tracking URI using DagsHub credentials when available
REPO_OWNER = "Abdelrahman-Farouk88"
REPO_NAME = "mlops_project"

dagshub_user = os.getenv("DAGSHUB_USERNAME")
dagshub_token = os.getenv("DAGSHUB_TOKEN")

if dagshub_user and dagshub_token:
    # embed credentials into the MLflow URI so mlflow operations authenticate
    tracking_uri = f"https://{dagshub_user}:{dagshub_token}@dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
else:
    # fallback (no auth)
    tracking_uri = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"

mlflow.set_tracking_uri(tracking_uri)

# Safe print of tracking URI without credentials for debug
try:
    parsed = urlparse(mlflow.get_tracking_uri())
    host_only = urlunparse(parsed._replace(netloc=parsed.hostname or ""))
except Exception:
    host_only = mlflow.get_tracking_uri()

print("DEBUG: DAGSHUB_USERNAME present:", bool(dagshub_user))
print("DEBUG: DAGSHUB_TOKEN present:", bool(dagshub_token))
print("DEBUG: MLflow tracking host:", host_only)

# Load run info
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

# Normalize registry model name (avoid spaces)
registered_name = model_name.replace(" ", "_")

# Initialize Mlflow client (uses the mlflow tracking URI set above)
client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())

# The artifact path used when the model was logged with mlflow.sklearn.log_model(name=...)
# Use the same model_name that was used when logging (run_info.model_name)
model_uri = f"runs:/{run_id}/{model_name}"

print(f"Attempting to register model from: {model_uri} as registry name: {registered_name}")

try:
    client.create_registered_model(registered_name)
    print(f"Successfully created registered model '{registered_name}'.")
except mlflow.exceptions.MlflowException as e:
    # If the model already exists, that's fine — we'll create a new version
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
    # Common cause: auth failure or wrong model_uri/artifact path
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