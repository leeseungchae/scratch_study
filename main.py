import os

import hydra
from omegaconf import DictConfig

from core.train import Trainer


@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig) -> None:
    print("Working directory : {}".format(os.getcwd()))
    trainer = Trainer(cfg)

    trainer.train()


if __name__ == "__main__":
    print("git_test")
    train()
    # export HYDRA_FULL_ERROR = 1 python main.py hydra.job.chdir=False
