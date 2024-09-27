import hydra
import pandas as pd 
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

def get_data(raw_path:str):
    data = pd.read_csv(raw_path)
    return data

def get_features(target:str,features:list,data:pd.DataFrame):
    X = pd.get_dummies(data[features],drop_first=False).astype("Int64")
    print(X.head(5))
    y = data[target]
    return X,y



@hydra.main(version_base="1.1",config_path="../../config",config_name="main")
def process_data(config:DictConfig):
    
    data =get_data(abspath(config.raw.path))

    X,y = get_features(config.process.target, config.process.features,data)
    

    X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.2,random_state=42)

    #Save Data
    X_train.to_csv(abspath(config.processed.X_train.path),index=False)
    X_test.to_csv(abspath(config.processed.X_test.path),index=False)
    y_train.to_csv(abspath(config.processed.y_train.path),index=False)
    y_test.to_csv(abspath(config.processed.y_test.path),index=False)

if __name__ == "__main__":
    process_data()
