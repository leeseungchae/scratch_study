from typing import Optional

import torch
from torch import Tensor

from .base import RNNBase


class RNNCell(RNNBase):
    """An Elman RNN cell with tanh or ReLU non-linearity.

    Args:
        input_size: The number of expected features in the input 'x'
        hidden_size: The number of fetures in the hidden state 'h'
        bias: If `False` then the layer doen not use bias weight bias. but no bias code here.
        nonlinearity: The non-linearity function to use. Can be either `tanh` or `relu`. Defaults to 'tanh'.

    Inputs: input, hidden
        - input: tensor containing the input features
        - hidden: tensor containing the initial hidden state
    """

    def __init__(
        self, input_size: int, hidden_size: int, bias: bool, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            num_chunks=1,
            **factory_kwargs
        )

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        # Make zero tensor if tensor is not initialized
        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )

        # forward
        hy = self.ih(input) * self.hh(hx)

        # function
        if self.nonlinearity == "tanh":
            ret = torch.tanh(hy)

        else:
            ret = torch.relu(hy)

        return ret
