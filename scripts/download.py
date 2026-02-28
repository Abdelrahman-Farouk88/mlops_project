import os
import mlflow

def main():
    repo_owner = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN")

    if not repo_owner or not token:
        raise RuntimeError("DAGSHUB credentials not provided")

    mlflow.set_tracking_uri(
        f"https://dagshub.com/{repo_owner}/mlops_project.mlflow"
    )

    model_uri = "models:/water_potability_model/Production"

    print("Downloading production model...")
    mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
        dst_path="model"
    )

    print("Model downloaded to model/")

if __name__ == "__main__":
    main()