import os.path
from typing import Dict

import hydra
from hydra.utils import get_original_cwd
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from src_pl.data_helper import TranslationData
from src_pl.train_pi import TranslationModel


def make_config(cfg: DictConfig) -> Dict:
    result = {}
    result.update(dict(cfg.data))
    result.update(dict(cfg.model))
    result.update(dict(cfg.trainer))
    return result


@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig) -> None:
    model = TranslationModel(cfg)
    data_module = TranslationData(
        arg_data=cfg.data,
        src_vocab=model.src_vocab,
        trg_vocab=model.trg_vocab,
        max_seq_size=cfg.model.max_sequence_size,
        batch_size=cfg.trainer.batch_size,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(get_original_cwd(), "./SavedModel/"),
        filename=cfg.data.folder_name,
        save_top_k=True,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience="cfg.trainer.early_stopping",
        verbose=False,
        mode="min",
    )
    wandb_logger = WandbLogger(project=cfg.data.project_name, name=cfg.data.folder_name)
    wandb_logger.log_hyperparams(make_config(cfg))
    trainer = Trainer(devices="auto", accelerator="auto", max_epochs=cfg.trainer.epochs)
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    train()
    # export HYDRA_FULL_ERROR = 1 python main_pl.py hydra.job.chdir=False