import bentoml
import numpy as np
import pandas as pd
from bentoml.io import JSON, NumpyNdarray
from hydra import compose, initialize
from pydantic import BaseModel

with initialize(version_base=None, config_path="../../config"):
    config = compose(config_name="main")
    FEATURES = config.process.features
    MODEL_NAME = config.model.name

class Employee(BaseModel):
    Education:str = 'Bachelors'
    JoiningYear:int = 2017
    City: str = "Pune"
    PaymentTier: int = 1
    Age: int = 25
    Gender: str = "Female"
    EverBenched: str = "No"
    ExperienceInCurrentDomain: int = 1


def add_dummy_data(df: pd.DataFrame):
    """Add dummy rows so that patsy can create features similar to the train dataset"""
    rows = {
        "Education": ['Bachelors','Masters','PHD'],
        "JoiningYear":[2016,2015,2017],
        "City": ["Bangalore", "New Delhi", "Pune"],
        "Gender": ["Male", "Female", "Female"],
        "EverBenched": ["Yes", "Yes", "No"],
        "PaymentTier": [0, 0, 0],
        "Age": [0, 0, 0],
        "ExperienceInCurrentDomain": [0, 0, 0],
    }
    dummy_df = pd.DataFrame(rows)
    return pd.concat([df, dummy_df])

def get_features(features:list,data:pd.DataFrame):
    X = pd.get_dummies(data[features],drop_first=False).astype("Int64")
    return X




@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class employee_churn:
    # Retrieve the latest version of the model from the BentoML Model Store
    bento_model = bentoml.models.get(f"{MODEL_NAME}:latest")

    def __init__(self):
        self.model = bentoml.xgboost.load_model(self.bento_model)


    @bentoml.api()
    def predict(self,employee: Employee) -> np.ndarray:
        """Transform the data then make predictions"""
        df = pd.DataFrame(employee.dict(), index=[0])
        df = add_dummy_data(df)

        df = get_features(FEATURES,df)
        return  self.model.predict(df)[0]
