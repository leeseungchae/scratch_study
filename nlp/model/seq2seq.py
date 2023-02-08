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
        for m in self.modules():
            if hasattr(m, "weight") and m.weight.dim() > 1:
                nn.init.xavier_uniform(m.weight.data)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight.data)

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

        for m in self.modules():
            if hasattr(m, "weight") and m.weight.dim() > 1:
                nn.init.xavier_uniform(m.weight.data)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight.data)

    def forward(
        self, enc_input: Tensor, enc_hidden: Union[Tensor, Tuple[Tensor, Tensor]] = None
    ):
        embeded = self.embedding(enc_input)
        relu_embeded = F.relu(embeded)  # 논문에는 없는 것
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
        self, encoder: Encoder, decoder: Decoder, max_seq_size: int, vocab_size: int
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_seq_size = max_seq_size
        self.vocab_size = vocab_size

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
