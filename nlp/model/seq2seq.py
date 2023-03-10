import random
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Encoder(nn.Module):
    def __init__(
        self,
        output_size: int,
        hidden_size: int,
        n_layers: int,
        dropout: float,
        mode: str = "lstm",
        batch_first: bool = True,
        bias: bool = True,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.layers = self.select_mode(
            mode=mode,
            hidden_size=hidden_size,
            n_layers=n_layers,
            bidirectional=bidirectional,
            bias=bias,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(
        self,
        dnc_input: Tensor,
        hidden: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
    ):
        embeded = self.embedding(dnc_input)
        relu_embeded = F.relu(embeded)
        output, hidden = self.layers(embeded, relu_embeded)
        output = self.softmax(self.linear(output))
        return output, hidden

    def select_mode(
        self,
        mode: str,
        hidden_size: int,
        n_layers: int,
        dropout: float,
        batch_first: bool = True,
        bias: bool = True,
        bidirectional: bool = False,
    ):
        if mode == "lstm":
            return nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers=n_layers,
                bidirectional=bidirectional,
                bias=bias,
                dropout=dropout,
                batch_first=batch_first,
            )

        elif mode == "rnn":
            return nn.RNN(
                hidden_size,
                hidden_size,
                num_layers=n_layers,
                bidirectional=bidirectional,
                bias=bias,
                dropout=dropout,
                batch_first=batch_first,
            )

        elif mode == "gru":
            return nn.GRU(
                hidden_size,
                hidden_size,
                num_layers=n_layers,
                bidirectional=bidirectional,
                bias=bias,
                dropout=dropout,
                batch_first=batch_first,
            )

        else:
            raise ValueError("param `mode` must be one of [rnn, lstm, gru]")


class Decoder(nn.Module):
    def __init__(
        self,
        output_size: int,
        hidden_size: int,
        n_layers: int,
        dropout: float,
        mode: str = "lstm",
        batch_first: bool = True,
        bias: bool = True,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(hidden_size, output_size)
        self.layers = self.select_mode(
            mode=mode,
            hidden_size=hidden_size,
            n_layers=n_layers,
            bidirectional=bidirectional,
            bias=bias,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(
        self, enc_input: Tensor, enc_hidden: Union[Tensor, Tuple[Tensor, Tensor]] = None
    ):
        embeded = self.embedding(enc_input)
        relu_embeded = F.relu(embeded)  # ???????????? ?????? ???
        output, hidden = self.layers(relu_embeded)
        output = self.softmax(self.linear(output[0]))
        return output, hidden

    def select_mode(
        self,
        mode: str,
        hidden_size: int,
        n_layers: int,
        dropout: float,
        batch_first: bool = True,
        bias: bool = True,
        bidirectional: bool = False,
    ):
        if mode == "lstm":
            return nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers=n_layers,
                bidirectional=bidirectional,
                bias=bias,
                dropout=dropout,
                batch_first=batch_first,
            )

        elif mode == "rnn":
            return nn.RNN(
                hidden_size,
                hidden_size,
                num_layers=n_layers,
                bidirectional=bidirectional,
                bias=bias,
                dropout=dropout,
                batch_first=batch_first,
            )

        elif mode == "gru":
            return nn.GRU(
                hidden_size,
                hidden_size,
                num_layers=n_layers,
                bidirectional=bidirectional,
                bias=bias,
                dropout=dropout,
                batch_first=batch_first,
            )

        else:
            raise ValueError("param `mode` must be one of [rnn, lstm, gru]")


class Seq2Seq(nn.Module):
    def __init__(
        self,
        enc_d_input: int,
        dec_d_input: int,
        d_hidden: int,
        n_layers: int,
        max_sequence_size: int,
        mode: str = "lstm",
        dropout_rate: float = 0.0,
        bidirectional: bool = True,
        bias: bool = True,
        batch_first: bool = True,
    ) -> None:

        super().__init__()
        self.encoder = Decoder(
            output_size=enc_d_input,
            hidden_size=d_hidden,
            n_layers=n_layers,
            dropout=dropout_rate,
            mode=mode,
            bidirectional=bidirectional,
            bias=bias,
            batch_first=batch_first,
        )

        self.max_seq_size = max_sequence_size
        self.vocab_size = dec_d_input

    def forward(
        self,
        enc_input: Tensor,
        dec_input: Tensor,
        teacher_focing_rate: Optional[float] = 1.0,
    ):
        enc_hidden = None
        for i in range(self.max_seq_size):
            enc_input_i = enc_input[:, i]
            _, enc_hidden = self.encoder(enc_input=enc_input_i, enc_hidden=enc_hidden)

        decoder_output = torch.zeros(
            dec_input.size(0), self.max_seq_size, self.vocab_size
        )
        dec_hidden = enc_hidden
        for i in range(self.max_seq_size):
            if teacher_focing_rate >= 1.0:
                dec_input_i = dec_input[:, i]
            else:
                if i == 0 or random.random() >= teacher_focing_rate:
                    dec_input_i = dec_input_i[:, i]
                    dec_input_i = (
                        dec_output_i.topk(1).seaueeze().detach()
                    )  # teacher focing
                else:
                    pass
            dec_output_i, dec_hidden = self.decoder(dec_input_i, dec_hidden)
            decoder_output[:, i, :] = dec_output_i
        return decoder_output
