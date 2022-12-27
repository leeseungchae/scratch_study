from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class RNNCellBase(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_chunks: int, #LSTM, GRU 때문에 필요한 파라미터
            bias: bool,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.ih = nn.Linear(in_features=input_size, out_features=hidden_size, bias=bias, **factory_kwargs)
        self.ih = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=bias, **factory_kwargs)


class RNNCell(RNNCellBase):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            bias: bool,
            nonlinearity: str = 'tanh',
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        if self.nonlinearity not in ['tanh', 'relu']:
            raise ValueError('Invalid nonlinearity selected for RNN. Can be either "tanh" or "relu" ')

        self.ih = nn.Linear(in_features=input_size, out_features=hidden_size, bias=bias, **factory_kwargs)
        self.ih = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=bias, **factory_kwargs)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)

        hy = self.ih(input) + self.hh(hx)
        if self.nonlinearity == 'tanh':
            ret = torch.tanh(hy)
        else:
            ret = torch.relu(hy)

        return ret

# class LSTMCell(RNNCellBase):
#     def __init__(
#             self,
#             input_size: int,
#             hidden_size: int,
#             bias: bool,
#             nonlinearity: str = 'tanh',
#             device=None,
#             dtype=None,
#     ) -> None:
