import os.path

import torch.optim.optimizer
from omegaconf import DictConfig
from torch import Tensor, nn

import wandb
from core.base import ABstracTools
from nlp.utils.utils import count_parameters
from nlp.utils.weight_initialization import select_weight_initialize_method


class Trainer(ABstracTools):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.model = self.get_model()
        self.model.train()
        self.init_optimizer()
        print(f"The model {count_parameters(self.model)}")

        select_weight_initialize_method(
            method=self.arg.model.weight_init,
            distribution=self.arg.model.weight_distribution,
            model=self.model,
        )
        self.train_loader, self.valid_loader = self.get_loader()
        self.loss_function = nn.CrossEntropyLoss(
            ignore_index=self.arg.data.pad_id,
            label_smoothing=self.arg.trainer.label_smoothing_value,
        )

        wandb.init(config=self.arg)

    def train(self):
        print(f"The model{count_parameters(self.model)} trainerble parameters")
        wandb.watch(self.model)

        epoch_step = len(self.train_loader) + 1
        total_step = self.arg.trainer.epochs * epoch_step
        step = 0
        for epoch in range(self.arg.trainer.epochs):
            for idx, data in enumerate(self.train_loader, 1):
                try:
                    self.optimizer.zero_grad()
                    src_input, trg_input, trg_output = data
                    output = self.model(src_input, trg_input)
                    loss = self.calculate_loss(output, trg_output)

                    if step % self.arg.trainer.print_train_step == 0:
                        wandb.log({"Train_loss": loss.item()})
                        print(
                            "[Train] epoch: {0:2d} iter: {1:4d}/{2:4e} step: {3:6d}/{4:6d} =>"
                            "loss: {5:10f}".format(
                                epoch, idx, epoch_step, step, total_step, loss.item()
                            )
                        )
                    if step % self.arg.trainer.print_valid_step == 0:
                        val_loss = self.valid()
                        wandb.log({"Train_loss": val_loss})
                        print(
                            "[Train] epoch: {0:2d} iter: {1:4d}/{2:4e} step: {3:6d}/{4:6d} =>"
                            "loss: {5:10f}".format(
                                epoch, idx, epoch_step, step, total_step, loss.item()
                            )
                        )

                    if step % self.arg.save_steps == 0:
                        self.save_model(epoch, step)

                    loss.backward()
                    self.optimizer.step()
                    step += 1
                except Exception as e:
                    self.save_model(epoch, step)
                    raise e

    def save_model(self, epoch: int, step: int) -> None:
        model_name = f"{str(step).zfill(6)}_{self.arg.model.model_type}.pth"
        model_path = os.path.join(self.arg.data.model_path, model_name)
        # if not os.path.isdir(self.arg.data.model_path):
        os.makedirs(self.arg.data.model_path, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "step": step,
                "data": self.arg.data,
                "model": self.arg.model,
                "trainer": self.arg.trainer,
                "model_state_dict": self.model.state_dict(),
            },
            model_path,
        )

    def valid(self) -> float:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in self.valid_loader:
                src_input, trg_input, trg_output = data
                output = self.model(src_input, trg_input)
                loss = self.calculate_loss(output, trg_output)
                total_loss += loss.item()

        input_sentence = self.tensor2sentence(src_input[0].tolist(), self.src_vocab)
        predict_sentence = self.tensor2sentence(
            output.topk(1)[1].squeeze()[0, :].tolist(), self.trg_vocab
        )
        target_sentence = self.tensor2sentence(trg_input[0].tolist(), self.trg_vocab)
        return total_loss / len(self.valid_loader)

    def calculate_loss(self, predict: Tensor, target: Tensor) -> Tensor:
        """

        :rtype: object
        """
        # print('predict', predict)
        predict = predict.transpose(1, 2)
        # print('predict_size',predict.size())
        # print('target_size',target.size())

        return self.loss_function(predict, target)

    def valid(self) -> float:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in self.valid_loader:
                src_input, trg_input, trg_output = data
                output = self.model(src_input, trg_input)
                loss = self.calculate_loss(output, trg_output)
                total_loss += loss.item()

    def init_optimizer(self) -> None:
        optimizer_type = self.arg.trainer.optimizer
        if optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.arg.trainer.learning_rate,
                betas=(self.arg.trainer.optimizer_b1, self.arg.trainer.optimizer_b2),
                eps=self.arg.trainer.optimizer_e,
                weight_decay=self.arg.trainer.weight_decay,
            )

        elif optimizer_type == "AdamW":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.arg.trainer.learning_rate,
                betas=(self.arg.trainer.optimizer_b1, self.arg.trainer.optimizer_b2),
                eps=self.arg.trainer.optimizer_e,
                weight_decay=self.arg.weight_decay,
            )
        else:
            raise ValueError("trainer param 'optimizer' must be one of [Adam ,AdamW]")
