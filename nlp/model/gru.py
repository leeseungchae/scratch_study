from typing import Optional

import torch
from torch import _VF, Tensor
from torch.nn.utils.rnn import PackedSequence

from .base import RNNBase, RNNCellBase


class GRUCell(RNNCellBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """A gated recurrent unit (GRU) cell

        .. math::

            \begin{array}{ll}
            r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
            z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
            n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
            h' = (1 - z) * n + z * h
            \end{array}

        where :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.

        Args:
            input_size: The number of expected features in the input `x`
            hidden_size: The number of features in the hidden state `h`
            bias: If ``False``, then the layer does not use bias weights `b_ih` and
                `b_hh`. Default: ``True``
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super(GRUCell, self).__init__(
            input_size, hidden_size, bias, num_chunks=3, **factory_kwargs
        )

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            input (Tensor): tensor containing input features
                - shape : (batch_size, input_size) or (input_size)
            hx (Optional[Tensor], optional): tensor containing the initial hidden state.
                - Defaults to None.
                - shape : (batch_size, hidden_size) or (hidden_size)

        Returns:
            Tensor: tensor containing the next hidden state for each element in the batch
                - shape : (batch_size, hidden_size)
        """

        assert input.dim() in (
            1,
            2,
        ), f"GRUCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"

        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        x_t = self.ih(input)
        h_t = self.hh(hx)

        x_reset, x_update, x_new = x_t.chunk(3, 1)
        h_reset, h_update, h_new = h_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_update + h_update)
        new_gate = torch.tanh(x_new + (reset_gate + h_new))

        h_y = update_gate * hx + (1 - update_gate) * new_gate

        return h_y


class GRU(RNNBase):
    def __init__(self, *args, **kwargs) -> None:
        super(GRU, self).__init__("GRU", *args, **kwargs)
        self.forward_gru = self.init_layers()

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        batch_dim = 0 if self.batch_first else 1
        sequence_dim = 1 if self.batch_first else 0
        batch_size = input.size(batch_dim)
        sequence_size = input.size(sequence_dim)
        is_batch = input.dim() == 3
        if not is_batch:
            input = input.unsqueeze(batch_dim)  # -> [1, L, H_in] or [L, 1, H_in]

        if hx is None:
            h_zeros = torch.zeros(  # hidden state 초기화
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )

            c_zeros = torch.zeros(  # cell state 초기화
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )
            hx = (h_zeros, c_zeros)

        elif is_batch:
            if hx[0].dim() != 3 or hx[1].dim() != 3:
                msg = (
                    "For batched 3-D input, hx and cx should "
                    f"also be 3-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors"
                )
                raise RuntimeError(msg)

        else:
            if hx[0].dim() != 2 or hx[1].dim() != 2:
                msg = (
                    "For unbatched 2-D input, hx and cx should "
                    f"also be 2-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors"
                )
                raise RuntimeError(msg)
            hx = (hx[0].unsqueeze(1), hx[1].unsqueeze(1))

        next_hidden, next_cell = [], []

        for layer_idx, lstm_cell in enumerate(self.forward_gru):
            if layer_idx == 0:
                input_state = input

            else:
                input_state = torch.stack(next_hidden, dim=sequence_dim)
                next_hidden = []
                next_cell = []

            h_i = hx[0][layer_idx, :, :]
            c_i = hx[1][layer_idx, :, :]
