import joblib 
import pandas as pd
from training.src.train_model import load_data
from hydra import compose,initialize
from hydra.utils import to_absolute_path as abspath

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import PredictionDrift

def test_xgboost():
    with initialize(version_base=None,config_path="../../config"):
        config = compose(config_name='main')

    model_path = abspath(config.model.path)
    model = joblib.load(model_path)
    
    X_train,X_test,y_train,y_test  = load_data(config.processed)

    train_df = pd.concat([X_train,y_train],axis=1)
    test_df = pd.concat([X_test,y_test],axis=1)

    # Get all columns as a list
    train_df_list = [col for col in train_df.columns if col != 'LeaveOrNot']
    test_df_list = [col for col in test_df.columns if col != 'LeaveOrNot']


    train_ds = Dataset(train_df,label='LeaveOrNot',cat_features=train_df_list)
    vaildation_ds = Dataset(test_df,label='LeaveOrNot',cat_features=test_df_list)

    check = PredictionDrift(drift_mode='prediction')
    result=check.run(train_ds, vaildation_ds, model=model)
    result.save_as_html('predict-diff.html')

