import torch.optim.optimizer
from omegaconf import DictConfig
from torch import nn

from core.base import ABstracTools
from nlp.utils.utils import count_parameters
from nlp.utils.weight_initialization import select_weight_initialize_method


class Trainer(ABstracTools):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.optimizer = None

    def train(self):
        model = self.get_model()
        model.train()
        print(f"The model {count_parameters(model)}")

        select_weight_initialize_method(
            method=self.arg.model.weight_init,
            distribution=self.arg.model_weight_distribtion,
            model=model
        )

    def init_optimizer(self, model:nn.Module) ->None:
        optimizer_type = self.arg.trainer.init_optimize
        if optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr = self.arg.trainer.learning_rate,
                betas=(self.arg.trainer.optimizer_b1,self.arg.trainer.optimizer_b2),
                eps = self.arg.trainer.optimizer_e,
                weight_decay = self.arg.weight_decay
            )

        elif optimizer_type == 'AdamW':
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr = self.arg.trainer.learning_rate,
                betas=(self.arg.trainer.optimizer_b1,self.arg.trainer.optimizer_b2),
                eps = self.arg.trainer.optimizer_e,
                weight_decay = self.arg.weight_decay
            )
        else:
            raise ValueError("trainer param 'optimizer' must be one of [Adam ,AdamW]")

