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
            **factory_kwargs,
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


class RNN(RNNBase):
    def __init__(self, *args, **kwargs) -> None:
        self.nonlinearity = kwargs.pop("nonlinearity", "tanh")
        if self.nonlinearity == "tanh":
            mode = "RNN_TAHN"
        elif self.nonlinearity == "relu":
            mode = "RNN_RELU"
        else:
            raise ValueError(f"UnKnown nonlinearity{self.nonlinearity}")
        super(RNN, self).__init__(mode, *args, **kwargs)
        self.forward_rnn = self.init_layers()

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:

        """

        N :N batch_size
        L : sequence length
        D : 2 if bidrectional=True otherwise 1
        H_in = input_size
        H_out = hidden_size

        :args
            input :(Tensor) : The input can also be a packed variable length sequence.
                -(L,H_in) for unbatched input
                -(L,M,H_in) when ''batch_first'' = False
                -(M,L,H_in) when ''batch_first'' = True
            hx (Optional[Tensor], optional): _description_. Defaults to None.
                -(num
                -(num_layers, batch_size, H_out)
        Returns
            Tensor: _description_
        """
        batch_dim = 0 if self.batch_first else 1
        is_batch = input.dim() == 3
        if not is_batch:
            input = input.unsqueeze(batch_dim)  # -> [1,L,H_in] or [L,1,H_in]
            if hx is not None:
                if hx.dim() != 2:
                    raise RuntimeError(
                        f"for batched 3-D input, hx should also be 3-D but got{hx.dim()}-D tensor"
                    )
                hx = hx.unsqueeze(1)  # -> [num_layers, 1, H_out]
        else:
            if (
                hx is not None and hx.dim() != 3
            ):  # hidden state를 넣었는데, dimension이 3개가 아닐 때
                raise RuntimeError(
                    f"for batched 3-D input, hx should also be 3-D but got{hx.dim()}-D tensor"
                )

            # input -> [N,L,H_in] or [L,H,H_in]
            # hx -> [num_layers, batch_size, H_out]

        batch_size = input.size(0) if self.batch_first else input.size(1)
        sequence_size = input.size(1) if self.batch_first else input.size(1)

        if hx is None:
            hx = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )

        # hx -> [num_layers, batch_size, H_out]

        hidden_state = []
        next_hidden = []
        for layer_idx, rnn_cell in enumerate(self.forward_rnn):
            input_state = input  # Todo 여기가 if else 바꿔야지 layer 2부터 동작 가능
            h_i = hx[
                layer_idx, :, :
            ]  # -> [layer_idx번째, batch_szie, H_out] layer_idx번째의 hidden_state

            for i in range(sequence_size):
                input_split = (
                    input_state[:, i, :] if self.batch_first else input_state[i, :, :]
                )
                # -> [N,1,H_in] or [1,N,H_in]
                h_i = rnn_cell(input_split, h_i)
                # Todo : dropout
                next_hidden.append(h_i)
            hidden_state.append(torch.stack(next_hidden, dim=1))

    def init_layers(self):
        layers = []
        for layer_idx in range(self.num_layers):
            input_size = self.input_size if layer_idx == 0 else self.hidden_size
            layers.append(
                RNNCell(
                    input_size=input_size,
                    hidden_size=self.hidden_size,
                    bias=self.bias,
                    nonlinearity=self.nonlinearity,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
