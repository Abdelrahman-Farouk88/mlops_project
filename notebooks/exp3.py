import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import dagshub
from sklearn.model_selection import train_test_split
import os

import dagshub
dagshub.init(repo_owner='Abdelrahman-Farouk88', repo_name='mlops_project', mlflow=True)
mlflow.set_experiment("Experiment 3")
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

X_train = train_processed_data.drop(columns=["Potability"])
y_train = train_processed_data["Potability"]
X_test = test_processed_data.drop(columns=["Potability"])
y_test = test_processed_data["Potability"]

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Classifier": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
     "XG Boost" : XGBClassifier()
 }

with mlflow.start_run(run_name="Water Potability Models Experiment"):
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name, nested=True):  
            model.fit(X_train, y_train)
            
            model_filename = f"{model_name.replace(' ', '_')}.pkl"
            pickle.dump(model, open(model_filename, "wb"))
            
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix for {model_name}")
            plt.savefig(f"confusion_matrix_{model_name.replace(' ', '_')}.png")
            
            mlflow.log_artifact(f"confusion_matrix_{model_name.replace(' ', '_')}.png")
            
            mlflow.sklearn.log_model(model, model_name.replace(' ', '_'))
    
    mlflow.log_artifact(__file__)
    
    mlflow.set_tag("author", "Abdelrahman Farouk")
    
    print("All models have been trained and logged as child runs successfully.")