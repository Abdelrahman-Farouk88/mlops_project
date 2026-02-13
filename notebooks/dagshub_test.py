import mlflow
import dagshub


mlflow.set_tracking_uri('https://dagshub.com/Abdelrahman-Farouk88/mlops_project.mlflow')


dagshub.init(repo_owner='Abdelrahman-Farouk88', repo_name='mlops_project', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)