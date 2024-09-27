import pandas as pd
from pandera import Check,Column,DataFrameSchema
from pytest_steps import test_steps

from training.src.process import get_features
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

@test_steps('get_features_step')
def test_process_suite(test_step,steps_data):
    if test_step == 'get_features_step':
        get_features_step(steps_data)


def get_features_step(steps_data):
    data = pd.DataFrame(
        {
            "Education":["Bachelors","Masters"],
            "JoiningYear":[2017,2017],
            "City":['Bangalore',"Pune"],
            "PaymentTier":[3,3],
            "Age":[34,24],
            "Gender":['Male','Male'],
            "EverBenched":['No','Yes'],
            "ExperienceInCurrentDomain":[0,2],
            "LeaveOrNot":[0,1]
        }
    )
    features=[
            "Education",
            "JoiningYear",
            "City",
            "PaymentTier",
            "Age",
            "Gender",
            "EverBenched",
            "ExperienceInCurrentDomain",
    ]
    target="LeaveOrNot"

    X,y=get_features(target,features,data)
    schema=DataFrameSchema(
        {
        "PaymentTier":Column(int,Check.isin([1,2,3])),
        "Age":Column(int,Check.greater_than(10)),
        "ExperienceInCurrentDomain":Column(int,Check.greater_than_or_equal_to(0)),
        "JoiningYear":Column(int,Check.greater_than(0)),
        "City_Bangalore":Column(int,Check.isin([0,1])),
        "City_Pune":Column(int,Check.isin([0,1])),
        "Gender_Male":Column(int,Check.isin([0,1])),
        "EverBenched_No":Column(int,Check.isin([0,1])),
        "EverBenched_Yes":Column(int,Check.isin([0,1])),
        "Education_Bachelors":Column(int,Check.isin([0,1])),
        "Education_Masters":Column(int,Check.isin([0,1])),
        }
    )


    schema.validate(X)