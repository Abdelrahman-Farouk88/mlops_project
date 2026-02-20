import unittest
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.tracking import MlflowClient
import os
import pandas as pd

# Use the name defined in your src scripts
MODEL_NAME = "water_potability_model"
ARTIFACT_PATH = "model" 

class TestModelLoading(unittest.TestCase):
    def setUp(self):
        # Tracking setup
        repo_owner = "Abdelrahman-Farouk88"
        repo_name = "mlops_project"
        dagshub_url = "https://dagshub.com"
        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
        self.client = MlflowClient()

    def test_model_in_staging(self):
        # Updated to catch the deprecation warning context if needed
        versions = self.client.get_latest_versions(MODEL_NAME, stages=["Staging"])
        self.assertGreater(len(versions), 0, f"No model found for '{MODEL_NAME}' in Staging.")

    def test_model_loading(self):
        versions = self.client.get_latest_versions(MODEL_NAME, stages=["Staging"])
        if not versions:
            self.fail(f"No model found for '{MODEL_NAME}' in Staging.")

        run_id = versions[0].run_id
        # IMPORTANT: Use the ARTIFACT_PATH "model", not the MODEL_NAME
        model_uri = f'runs:/{run_id}/{ARTIFACT_PATH}'
        
        try:
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            self.assertIsNotNone(loaded_model, "The loaded model is None.")
        except Exception as e:
            self.fail(f"Failed to load the model from {model_uri}: {e}")

    def test_model_performance(self):
        versions = self.client.get_latest_versions(MODEL_NAME, stages=["Staging"])
        if not versions:
            self.fail("No model found in Staging, skipping performance test.")
        
        run_id = versions[0].run_id
        model_uri = f'runs:/{run_id}/{ARTIFACT_PATH}'
        loaded_model = mlflow.pyfunc.load_model(model_uri)

        test_data_path = "./data/processed/test_processed.csv"
        if not os.path.exists(test_data_path):
            self.fail(f"Test data not found at {test_data_path}")

        test_data = pd.read_csv(test_data_path)
        X_test = test_data.drop(columns=["Potability"])
        y_test = test_data["Potability"]

        predictions = loaded_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"Test Accuracy: {accuracy}")
        self.assertGreaterEqual(accuracy, 0.3, f"Accuracy {accuracy} is below threshold.")

if __name__ == "__main__":
    unittest.main()