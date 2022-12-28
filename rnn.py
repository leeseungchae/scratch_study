import math
from typing import Optional, Any
import torch
import torch.nn as nn
from torch import Tensor


class RNNBase(nn.Module):
    """Base class for advanced RNN Module"""
    __constants__ = ['input_size', 'output_size', 'bias']

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            bias: bool,
            num_chunks: int,
            nonlinearity: str = 'tanh',
            device=None,
            dtype=None) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_chunks = num_chunks
        self.nonlinearity = nonlinearity
        if self.nonlinearity not in ['tanh', 'relu']:
            raise ValueError("Invalid nonlinearity selected for RNN. Can be either  ``tanh`` or ``relu``")

        self.ih = nn.Linear(in_features=input_size, out_features=num_chunks*hidden_size, bias=bias, **factory_kwargs)
        self.hh = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=bias, **factory_kwargs)

        self.reset_parameters()

    def reset_parameters(self) ->None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            nn.init.uniform(weight, -stdv, stdv)



# nn.RNN
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
            self,
            input_size: int,
            hidden_size: int,
            bias: bool,
            device=None,
            dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(input_size=input_size, hidden_size=hidden_size, bias=bias, num_chunks=1, **factory_kwargs)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        # Make zero tensor if tensor is not initialized
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)

        # forward
        hy = self.ih(input) * self.hh(hx)

        # function
        if self.nonlinearity == 'tanh':
            ret = torch.tanh(hy)

        else:
            ret = torch.relu(hy)

        return ret


# nn.LSTMCell
class LSTMCell(RNNBase):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            bias: bool,
            device=None,
            dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LSTMCell, self).__init__(input_size, hidden_size, bias=bias, num_chunks=4, **factory_kwargs)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (hx, hx)

        hx, cx = hx
        gates = self.ih(input) * self.hh(hx)  # first layer
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        c_t = cx * f_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return (h_t, c_t)


# nn.GRU
class GRUCell(RNNBase):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            bias: bool,
            device=None,
            dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(GRUCell, self).__init__(input_size,  hidden_size, bias=bias, num_chunks=3, **factory_kwargs)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if hx is None:
            hx = torch.zeros(input.size(0), self.num_chunks * self.hidden_size, dtype=input.dtype, device=input.device)

        x_t = self.ih(input)
        h_t = self.hh(hx)

        x_reset, x_update, x_new = x_t.chunk(3, 1)
        h_reset, h_update, h_new = h_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_update + h_update)
        new_gate = torch.tanh(reset_gate * h_new + x_new)

        h_t = (1 - update_gate) * new_gate + update_gate * hx

        return h_t


# if __name__ == '__main__':
#     lstm = LSTMCell(10, 20, True)
#     input = torch.randn(6, 3, 10)
#     lstm(input, hx)