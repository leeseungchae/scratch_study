from typing import Optional, List

import torch
from torch import Tensor, _VF

from torch.nn.utils.rnn import PackedSequence

from .base import RNNCellBase,RNNBase


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

        assert input.dim() in (1, 2), \
            f"GRUCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"

        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx
        hx = hx[0]

        #Todo hx 가 두개의 튜플로 나온다.

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
        if self.bidirectional:
            self.backward_gru = self.init_layers()

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


        hidden_state = []
        cell_state = []


        if self.bidirectional:
            next_hidden_f, next_hidden_b = [], []
            next_cell_f, next_cell_b = [], []
            for layer_idx, (forward_cell, backward_cell) in enumerate(
                    zip(self.forward_gru, self.backward_gru)
            ):
                if layer_idx == 0:
                    input_f_state = input
                    input_b_state = input
                else:
                    input_f_state = torch.stack(next_hidden_f, dim=sequence_dim)
                    input_b_state = torch.stack(next_hidden_b, dim=sequence_dim)
                    next_hidden_f, next_hidden_b = [], []
                    next_cell_f, next_cell_b = [], []

                h_f_i = hx[0][2 * layer_idx, :, :]
                h_b_i = hx[0][2 * layer_idx + 1, :, :]
                c_f_i = hx[1][2 * layer_idx, :, :]
                c_b_i = hx[1][2 * layer_idx + 1, :, :]


                for i in range(sequence_size):
                    input_f_i = (
                        input_f_state[:, i, :]
                        if self.batch_first
                        else input_f_state[i, :, :]
                    )

                    input_b_i = (
                        input_b_state[:, -(i + 1), :]
                        if self.batch_first
                        else input_b_state[-(i + 1), :, :]
                    )

                    h_f_i, c_f_i = forward_cell(input_f_i, (h_f_i, c_f_i))
                    h_b_i, c_b_i = backward_cell(input_b_i, (h_b_i, c_b_i))

                    if self.dropout:
                        h_f_i = self.dropout(h_f_i)
                        h_b_i = self.dropout(h_b_i)
                        c_f_i = self.dropout(c_f_i)
                        c_b_i = self.dropout(c_b_i)

                    next_hidden_f.append(h_f_i)
                    next_hidden_b.append(h_b_i)
                    next_cell_f.append(c_f_i)
                    next_cell_b.append(c_b_i)

                hidden_state.append(torch.stack(next_hidden_f, dim=sequence_dim))
                hidden_state.append(torch.stack(next_hidden_b[::-1], dim=sequence_dim))
                cell_state.append(torch.stack(next_cell_f, dim=sequence_dim))
                cell_state.append(torch.stack(next_cell_b[::-1], dim=sequence_dim))

            hidden_states = torch.stack(hidden_state, dim=0)
            cell_states = torch.stack(cell_state, dim=0)

            output_f_state = hidden_states[-2, :, :, :]
            output_b_state = hidden_states[-1, :, :, :]
            output = torch.cat([output_f_state, output_b_state], dim=2)

        else:
            next_hidden, next_cell = [], []
            for layer_idx, gru_cell in enumerate(self.forward_gru):
                if layer_idx == 0:
                    input_state = input

                else:
                    input_state = torch.stack(next_hidden, dim=sequence_dim)
                    next_hidden = []
                    next_cell = []

                h_i = hx[0][layer_idx, :, :]
                c_i = hx[1][layer_idx, :, :]


                for i in range(sequence_size):
                    input_i = (
                        input_state[:, i, :]
                        if self.batch_first
                        else input_state[i, :, :]
                    )

                    h_i, c_i = gru_cell(input_i, (h_i, c_i))
                    if self.dropout:
                        h_i = self.dropout(h_i)
                        c_i = self.dropout(c_i)

                    next_hidden.append(h_i)
                    next_cell.append(c_i)

                hidden_state.append(torch.stack(next_hidden, dim=sequence_dim))
                cell_state.append(torch.stack(next_cell, dim=sequence_dim))

            hidden_states = torch.stack(hidden_state, dim=0)
            cell_states = torch.stack(cell_state, dim=0)

            output = hidden_states[-1, :, :, :]  # -> [L, N, H_out]

        hn = (
            hidden_states[:, :, -1, :]
            if self.batch_first
            else hidden_states[:, -1, :, :]
        )
        cn = cell_states[:, :, -1, :] if self.batch_first else cell_states[:, -1, :, :]
        return output, (hn, cn)




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


    def init_layers(self) -> List[GRUCell]:
        layers = []
        for layer_idx in range(self.num_layers):
            input_size = self.input_size if layer_idx == 0 else self.hidden_size
            layers.append(
                GRUCell(
                    input_size=input_size,
                    hidden_size=self.hidden_size,
                    bias=self.bias,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        return layers