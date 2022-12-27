from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import GRUCell


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
        self.num_chunks =num_chunks

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
        super(RNNCell)
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

class LSTMCell(RNNCellBase):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            bias: bool,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LSTMCell, self).__init__(input_size, 4*hidden_size,bias,num_chunks=4 , **factory_kwargs)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tuple[Any, Any]:
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (hx,hx)

        print(input.size())
        print(self.ih.size())
        print(self.hh.size())

        hx, cx = hx
        gates = self.ih(input) + self.hh(hx)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4,1)

        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        c_t = cx * f_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return (h_t, c_t)

    class GRUCell(RNNCellBase):
        def __init__(
                self,
                input_size: int,
                hidden_size: int,
                bias: bool,
                device=None,
                dtype=None,
        ) -> None:
            factory_kwargs = {"device": device, "dtype": dtype}
            super(GRUCell, self).__init__(input_size, 3 * hidden_size, bias, num_chunks=3, **factory_kwargs)

            def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tuple[Any, Any]:
                if hx is None:
                    hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)

                x_t = self.ih(input)
                h_t = self.hh(hx)

                x_reset, x_update, x_new = x_t.chunk(3,1)
                h_reset, h_update, h_new = h_t.chunk(3,1)

                reset_gate = torch.sigmoid(x_reset+h_reset)
                update_gate = torch.sigmoid(x_reset+h_update)
                new_gate = torch.tanh(reset_gate * h_new + x_new)

                h_y = (1- update_gate) * new_gate + update_gate * hx

                return h_y



