import json
from mlflow.tracking import MlflowClient
import mlflow

import dagshub
dagshub.init(repo_owner='Abdelrahman-Farouk88', repo_name='mlops_project', mlflow=True)

mlflow.set_experiment("Final_Model")
mlflow.set_tracking_uri("https://dagshub.com/Abdelrahman-Farouk88/mlops_project.mlflow")

reports_path = "reports/run_info.json"
with open(reports_path, 'r') as file:
    run_info = json.load(file)

run_id = run_info['run_id']
model_name = run_info['model_name']  

client = MlflowClient()

model_uri = f"runs:/{run_id}/model.pkl"

print(f"Attempting to register model from: {model_uri}")

try:
    client.create_registered_model("Best Model")
    print("Successfully registered model 'Best Model'.")
except mlflow.exceptions.MlflowException as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e) or "ALREADY_EXISTS" in str(e):
        print("Registered model 'Best Model' already exists. Creating a new version...")
    else:
        raise e

reg = client.create_model_version(
    name="Best Model",
    source=model_uri,
    run_id=run_id,
)

model_version = reg.version

new_stage = "Staging"

client.transition_model_version_stage(
    name="Best Model",
    version=model_version,
    stage=new_stage,
    archive_existing_versions=True
)

print(f"Model Best Model version {model_version} transitioned to {new_stage} stage.")