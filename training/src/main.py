from omegaconf import DictConfig, OmegaConf
import hydra
from evaluate_model import evaluate
from train_model import train
from process import process_data

@hydra.main(config_path="../../config",config_name="main")
def main(config: DictConfig) -> None:
    process_data(config)
    train(config)
    evaluate(config)

if __name__ == "__main__":
    main()