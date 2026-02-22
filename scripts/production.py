import mlflow
from mlflow.tracking import MlflowCliennt
import os

dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Abdelrahman-Farouk88"
repo_name = "mlops_project"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

model_name = "water_potability_model"

def promote_model_to_production():
    client = MlflowCliennt()

    staging_versions = client.get_latest_versions(model_name, stages = ["Staging"])
    if not staging_versions:
        print("No model found in the 'Staging' stage.")
        return
    
    latest_staging_version = staging_versions[0]
    staging_version_number = latest_staging_version.version

    production_versions = client.get_latest_versions(model_name, stages = ["Production"])
    
    if  production_versions:
        current_production_version = production_versions[0]
        production_version_number = current_production_version.version

        client.tranisition_model_version_stage(
            name = model_name,
            version = production_version_number,
            stage = "Archived",
            archive_existing_versions = False
        )

        print(f"Archived model version {production_version_number} in 'Production'.")

    else:
        print("No model currently in 'Production'")

    client.tranisition_model_version_stage(
            name = model_name,
            version = staging_version_number,
            stage = "Production",
            archive_existing_versions = False
        )
    print(f"Promoted model version {staging_version_number} to 'Production'.")

if __name__ == "__main__":
    promote_model_to_production()
    