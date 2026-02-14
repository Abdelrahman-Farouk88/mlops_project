import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from mlflow.models import infer_signature
import dagshub
import os

dagshub.init(repo_owner='Abdelrahman-Farouk88', repo_name='mlops_project', mlflow=True)


mlflow.set_experiment("Experiment 4")

mlflow.set_tracking_uri("https://dagshub.com/Abdelrahman-Farouk88/mlops_project.mlflow")  

base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "..", "data", "external", "water_potability.csv")
data = pd.read_csv(csv_path)

train_data, test_data = train_test_split(data, test_size=0.20, random_state=22)

def fill_missing_with_mean(df):
    for column in df.columns:
        if df[column].isnull().any():  
            mean_value = df[column].mean()  
            df[column].fillna(mean_value, inplace=True)  
    return df

train_processed_data = fill_missing_with_mean(train_data)
test_processed_data = fill_missing_with_mean(test_data)

X_train = train_processed_data.drop(columns=["Potability"], axis=1)  
y_train = train_processed_data["Potability"]  

rf = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators': [100, 200, 300, 500, 1000],  
    'max_depth': [None, 4, 5, 6, 10],  
}

random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, cv=5, n_jobs=-1, verbose=2, random_state=42)

with mlflow.start_run(run_name="Random Forest Tuning") as parent_run:

    random_search.fit(X_train, y_train)

    for i in range(len(random_search.cv_results_['params'])):
        with mlflow.start_run(run_name=f"Combination{i+1}", nested=True) as child_run:
            mlflow.log_params(random_search.cv_results_['params'][i])  
            mlflow.log_metric("mean_test_score", random_search.cv_results_['mean_test_score'][i])  

    print("Best parameters found: ", random_search.best_params_)

    mlflow.log_params(random_search.best_params_)

    best_rf = random_search.best_estimator_
    best_rf.fit(X_train, y_train)

    pickle.dump(best_rf, open("model.pkl", "wb"))

    X_test = test_processed_data.drop(columns=["Potability"], axis=1)  
    y_test = test_processed_data["Potability"]  

    model = pickle.load(open('model.pkl', "rb"))

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1 score", f1)

    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)
    
    mlflow.log_input(train_df, "train")  
    mlflow.log_input(test_df, "test")  

    mlflow.log_artifact(__file__)

    sign = infer_signature(X_test, random_search.best_estimator_.predict(X_test))
    
    mlflow.sklearn.log_model(random_search.best_estimator_, "Best Model", signature=sign)

    print("Accuracy: ", acc)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", f1)