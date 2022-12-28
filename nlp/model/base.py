import math

import torch.nn as nn


class RNNBase(nn.Module):
    """Base class for advanced RNN Module"""

    __constants__ = ["input_size", "output_size", "bias"]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        num_chunks: int,
        nonlinearity: str = "tanh",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_chunks = num_chunks
        self.nonlinearity = nonlinearity
        if self.nonlinearity not in ["tanh", "relu"]:
            raise ValueError(
                "Invalid nonlinearity selected for RNN. Can be either  ``tanh`` or ``relu``"
            )

        self.ih = nn.Linear(
            in_features=input_size,
            out_features=num_chunks * hidden_size,
            bias=bias,
            **factory_kwargs
        )
        self.hh = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            **factory_kwargs
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            nn.init.uniform(weight, -stdv, stdv)
