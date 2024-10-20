
import hydra
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import os
import dagshub

dagshub.init(repo_owner='as183789043', repo_name='MLOps-End-to-End-Project', mlflow=True)

def load_data(path: DictConfig):
    X_test = pd.read_csv(abspath(path.X_test.path))
    y_test = pd.read_csv(abspath(path.y_test.path))
    return X_test, y_test


def load_model(model_path: str):
    return joblib.load(model_path)


def predict(model: XGBClassifier, X_test: pd.DataFrame):
    return model.predict(X_test)

def log_params(model: XGBClassifier, features: list):
    mlflow.log_params({"model_class": type(model).__name__})
    model_params = model.get_params()

    for arg, value in model_params.items():
        mlflow.log_params({arg: value})

    mlflow.log_params({"features": features})


def log_metrics(**metrics: dict):
    mlflow.log_metrics(metrics)

@hydra.main( config_path="../../config", config_name="main")
def evaluate(config: DictConfig):
    mlflow.set_experiment('Employee-Churn')
    with mlflow.start_run():

        # Load data and model
        X_test, y_test = load_data(config.processed)

        model = load_model(abspath(config.model.path))

        # Get predictions
        prediction = predict(model, X_test)

        # Get metrics
        f1 = f1_score(y_test, prediction)
        print(f"F1 Score of this model is {f1}.")

        accuracy = accuracy_score(y_test, prediction)
        print(f"Accuracy Score of this model is {accuracy}.")

        # Log metrics
        log_params(model, config.process.features)
        log_metrics(f1_score=f1, accuracy_score=accuracy)



if __name__ == "__main__":
    evaluate()