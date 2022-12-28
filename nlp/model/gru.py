from typing import Optional

import torch
from torch import Tensor

from .base import RNNBase


class GRUCell(RNNBase):
    def __init__(
        self, input_size: int, hidden_size: int, bias: bool, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(GRUCell, self).__init__(
            input_size, hidden_size, bias=bias, num_chunks=3, **factory_kwargs
        )

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if hx is None:
            hx = torch.zeros(
                input.size(0),
                self.num_chunks * self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )

        x_t = self.ih(input)
        h_t = self.hh(hx)

        x_reset, x_update, x_new = x_t.chunk(3, 1)
        h_reset, h_update, h_new = h_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_update + h_update)
        new_gate = torch.tanh(reset_gate * h_new + x_new)

        h_t = (1 - update_gate) * new_gate + update_gate * hx

        return h_t
