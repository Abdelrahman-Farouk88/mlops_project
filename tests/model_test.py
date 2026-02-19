import unittest
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.tracking import MlflowClient
import os
import pandas as pd

dagshub_token = os.getenv("DAGSHUB_TOKEN")

if not dagshub_token:

    raise EncodingWarning("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token

dagshub_url = "https://dagshub.com"

repo_owner = "Abdelrahman-Farouk88"
repo_name = "mlops_project"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

model_name = "Best Model"

class TestModelLoading(unittest.TestCase):

    def test_model_in_staging(self):

        client = MlflowClient()

        versions = client.get_latest_versions(model_name, stages = ["Staging"])

        self.assertGreater(len(versions), 0, "No model found in the 'Staging' stage.")

    def test_model_loading(self):

        client = MlflowClient()

        versions = client.get_latest_versions(model_name, stages = ["Staging"])

        if not versions:
            self.fail("No model found in the 'Staging' stage.")

        latest_version = versions[0].version

        run_id = versions[0].run_id

        logged_model = f'runs://{run_id}/{model_name}'
        

        try:
            logged_model = mlflow.pyfunc.load_model(logged_model)

        except Exception as e:

            self.fail(f"Failed to load the model: {e}")

        self.assertIsNotNone(logged_model, "the loaded model is None.")

        print(f"Model successfully loaded from {logged_model}.")


    def test_model_performance(self):

        client = MlflowClient()

        versions = client.get_latest_versions(model_name, stages = ["Staging"])

        if not versions:
            self.fail("No model found in the 'Staging' stage, skipping performance test.")

            latest_version = versions[0].run_id
            logged_model = f'runs://{latest_version}/{model_name}'
            loaded_model = mlflow.pyfunc.load_model(logged_model)

            test_data_path = "./data/processed/test_processed.csv"

            if not os.path.exists(test_data_path):
                self.fail(f"Test data not found at {test_data_path}")

            test_data = pd.read_csv(test_data_path)
            X_test = test_data.drop(columns=["Potability"])
            y_test = test_data["Potability"]

            predictions = loaded_model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)

            precision = precision_score(y_test, predictions, average="binary")
            recall = recall_score(y_test, predictions, average="binary")
            f1 = f1_score(y_test, predictions, average="binary")

            print(f"Accuracy: {accuracy}")

            print(f"Precision) {precision}")

            print(f"Recall: {recall}")

            print(f"F1 Score: {f1}")

            self.assertGreaterEqual(accuracy, 0.3, "Accuracy is below threshold.")
            self.assertGreaterEqual (precision, 0.3, "Precision is below threshold.")
            self.assertGreaterEqual(recall, 0.3, "Recall is below threshold.")
            self.assertGreaterEqual(f1, 0.3, "F1 Score is below threshold.")


if __name__ == "__main__":
    
    unittest.main()