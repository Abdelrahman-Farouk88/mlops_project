import mlflow
import dagshub
import pandas as pd
import pickle
import os

dagshub.init(repo_owner='Abdelrahman-Farouk88', repo_name='mlops_project', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Abdelrahman-Farouk88/mlops_project.mlflow")

model_name = "Best Model"

try:
    client = mlflow.tracking.MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    prod_versions = [v for v in versions if v.current_stage == "Production"]

    if prod_versions:
        latest_version = prod_versions[0].version
        run_id = prod_versions[0].run_id
        print(f'Latest version in Production: {latest_version}, Run ID: {run_id}')

        artifact_path = client.download_artifacts(run_id, "model.pkl")
        print("Downloaded model to:", artifact_path)

        with open(artifact_path, "rb") as f:
            model = pickle.load(f)

        data = pd.DataFrame({
                'ph': 3.7,
                'Hardness': 204.9,
                'Solids': 20791.3,
                'Chloramines': 7.3,
                'Sulfate': 368.5,
                'Conductivity': 564.3,
                'Organic_carbon': 10.37,
                'Trihalomethanes': 86.99,
                'Turbidity': 2.9
            }, index=[0])

        prediction = model.predict(data)
        print(f"Prediction: {prediction}")
        print("Water is Potable" if prediction[0] == 1 else "Water is Not Potable")
    else:
        print('No model found in the "Production" stage.')
except Exception as e:
    print(f'Error fetching model: {e}')



