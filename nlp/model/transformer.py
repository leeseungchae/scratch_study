from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nlp.utils.utils import get_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_position_encoding_table(max_sequence_size: int, d_hidden: int) -> Tensor:
    def get_angle(position: int, i: int) -> float:
        return position / np.power(10000, 2 * (i // 2) / d_hidden)

    def get_angle_vector(position: int) -> List[float]:
        return [get_angle(position, hid_j) for hid_j in range(d_hidden)]

    pe_table = Tensor([get_angle_vector(pos_i) for pos_i in range(max_sequence_size)])
    pe_table[:, 0::2] = np.sin(pe_table[:, 0::2])  # dim 2i
    pe_table[:, 1::2] = np.cos(pe_table[:, 1::2])  # dim 2i +1
    return pe_table


def get_position(inputs: Tensor) -> Tensor:
    position = (
        torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype)
        .expand(inputs.size(0), inputs.size(1))
        .contiguous()
    )  # -> [bs, max_seq_size]
    return position


def get_padding_mask(inputs: Tensor, padding_id: int) -> Tensor:
    pad_attn_mask = inputs.data.eq(padding_id).unsqueeze(1)
    return pad_attn_mask.expand(
        inputs.size(0), inputs.size(1), inputs.size(1)
    ).contiguous()


def get_look_ahead_mask(inputs: Tensor) -> Tensor:
    look_ahead_mask = (
        torch.ones_like(inputs)
        .unsqueeze(-1)
        .expand(inputs.size(0), inputs.size(1), inputs.size(1))
    )  # => [batch_size, seq_len, seq_len]
    look_ahead_mask = look_ahead_mask.triu(
        diagonal=1
    )  # upper triangular part of a matrix(2-D) => [batch_size, seq_len, seq_len]
    return look_ahead_mask.eq(0)


# def get_attn_decoder_mask(dec_input: torch.Tensor) -> torch.Tensor:
#     batch_size, seq_len = dec_input.size(0), dec_input.size(1)
#     subsequent_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=dec_input.device), diagonal=1)
#     subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, -1, -1)  # Expand the mask to match the batch size
#     return subsequent_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, head_dim: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = head_dim**0.5

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, attion_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """dot product attention

        Args:
            query (Tensor): (bs, n_heads, mex_seq, d_hidden)
            key (Tensor): (bs, n_heads, mex_seq, d_hidden)
            value (Tensor): (bs, n_heads, mex_seq, d_hidden)

        Returns:
            _type_: _description_
        """
        # print('attion_mask', attion_mask.size())
        scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale
        # print('scores', scores.size())
        # => [bs, n_heads, len_q(max_seq_size), len_k(max_seq_size)]
        # Todo : score masking되는지 확인  -
        # attion_mask = attion_mask.unsqueeze(1).repeat(1, 8, 1, 1)
        # print('attention_mask',attion_mask)
        # print('scroes', scores.size())

        scores.masked_fill_(attion_mask, -1e9)
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        context = torch.matmul(attn_prob, value).squeeze()
        return context, attn_prob

    def get_attn_pad_mask(seq_q: Tensor, seq_k: Tensor, padding_id: int) -> Tensor:
        pad_attn_mask = seq_k.data.eq(padding_id).unsqueeze(1)
        return pad_attn_mask.expand(seq_k.size(0), seq_q.size(1), seq_k.size(1))


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_hidden: int, n_heads: int, head_dim: int, dropout: float = 0
    ) -> None:
        super().__init__()
        self.weight_q = nn.Linear(d_hidden, n_heads * head_dim)
        self.weight_k = nn.Linear(d_hidden, n_heads * head_dim)
        self.weight_v = nn.Linear(d_hidden, n_heads * head_dim)

        self.self_attention = ScaledDotProductAttention(
            head_dim=head_dim, dropout_rate=dropout
        )
        self.linear = nn.Linear(n_heads * head_dim, d_hidden)

        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.head_dim = head_dim

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor):
        """MultiHeadAttention

        :arg
            query (Tensor): input word vector (bs, max_seq_len, d_hidden)
            key (Tensor): input word vector (bs, max_seq_len, d_hidden)
            value (Tensor): input word vector (bs, max_seq_len, d_hidden)
        :returns:
            :type: _description
        """
        batch_size = query.size(0)

        # print("query: ", query.size())
        # print("weight_q(query): ", self.weight_q)

        q_s = (
            self.weight_q(query)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        # => [batch_size, n_heads, max_seq_len, head_dim]
        k_s = (
            self.weight_k(key)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        # => [batch_size, n_heads, max_seq_len, head_dim]
        v_s = (
            self.weight_v(value)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        # => [batch_size, n_heads, max_seq_len, head_dim]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, _ = self.self_attention(
            q_s, k_s, v_s, attn_mask
        )  # => [batch_size, n_heads, max_seq_len, head_dim]
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.head_dim)
        )  # =>[batch_size, max_seq_len, n_heads*head_dim]
        output = self.linear(context)

        output = self.dropout(output)
        # print(output)
        return output


class PoswiseFeedForwardNet(nn.Module):
    def __init__(
        self, d_hidden: int, ff_dim: int, dropout: float = 0, layer_type: str = "linear"
    ) -> None:
        super().__init__()
        if layer_type == "linear":
            self.layer1 = nn.Linear(d_hidden, ff_dim)
            self.layer2 = nn.Linear(ff_dim, d_hidden)
        # elif layer_type == 'cnn':
        #     self.conv1 = nn.Conv1d(in_channels=d_hidden, out_channels=ff_dim, kernel_size=1)
        #     self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=d_hidden, kernel_size=1)
        else:
            ValueError(f"")
        self.active = F.relu  # gelu
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.dropout(self.active(self.layer1(inputs)))
        output = self.dropout(self.layer2(output))
        return output


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_hidden: int,
        n_heads: int,
        head_dim: int,
        ff_dim: int,
        dropout: float = 0,
        layer_norm_epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        self.mh_attention = MultiHeadAttention(
            d_hidden=d_hidden, n_heads=n_heads, head_dim=head_dim, dropout=dropout
        )
        self.layer_norm1 = nn.LayerNorm(d_hidden, eps=layer_norm_epsilon)
        self.ffnn = PoswiseFeedForwardNet(
            d_hidden=d_hidden, ff_dim=ff_dim, dropout=dropout
        )
        self.layer_norm2 = nn.LayerNorm(d_hidden, eps=layer_norm_epsilon)

    def forward(self, enc_inputs: Tensor, enc_self_attn_mask: Tensor) -> Tensor:
        mh_output = self.mh_attention(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask
        )
        mh_output = self.layer_norm1(enc_inputs + mh_output)
        ffnn_output = self.ffnn(mh_output)
        ffnn_output = self.layer_norm2(ffnn_output + mh_output)
        return ffnn_output


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_hidden: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        ff_dim: int,
        max_sequence: int,
        padding_id: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.src_emb = nn.Embedding(input_dim, d_hidden)
        pe_table = get_position_encoding_table(max_sequence, d_hidden)
        self.pos_emb = nn.Embedding.from_pretrained(pe_table, freeze=True)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_hidden=d_hidden,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.padding_id = padding_id

    def forward(self, enc_inputs: Tensor):
        position = get_position(inputs=enc_inputs)
        # print(self.src_emb(enc_inputs))
        # print(self.pos_emb(position))
        # print('src_emb',self.src_emb(enc_inputs))
        conb_emb = self.src_emb(enc_inputs) + self.pos_emb(position)
        enc_self_attn_mask = get_padding_mask(
            enc_inputs, self.padding_id
        )  # =>[batch_size, max_seq_size, max_seq_size]
        enc_outputs = conb_emb
        for layer in self.layers:
            enc_outputs = layer(enc_outputs, enc_self_attn_mask)
            # enc_outputs => [batch_size, len_q, d_model]
        return enc_outputs

        # enc_self_attn_mask =


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_hidden: int,
        n_heads: int,
        head_dim: int,
        ff_dim: int,
        dropout: float = 0,
        layer_norm_epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        self.masked_mh = MultiHeadAttention(
            d_hidden=d_hidden, n_heads=n_heads, head_dim=head_dim, dropout=dropout
        )
        self.layer_norm1 = nn.LayerNorm(d_hidden, eps=layer_norm_epsilon)
        self.enc_dec_mh = MultiHeadAttention(
            d_hidden=d_hidden, n_heads=n_heads, head_dim=head_dim, dropout=dropout
        )
        self.layer_norm2 = nn.LayerNorm(d_hidden, eps=layer_norm_epsilon)
        self.ffnn = PoswiseFeedForwardNet(
            d_hidden=d_hidden, ff_dim=ff_dim, dropout=dropout
        )
        self.layer_norm3 = nn.LayerNorm(d_hidden, eps=layer_norm_epsilon)

    def forward(
        self,
        dec_inputs: Tensor,
        enc_outputs: Tensor,
        dec_self_attn_mask: Tensor,
        dec_enc_padding_mask,
    ) -> Tensor:
        masked_mh_output = self.masked_mh(
            dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask
        )
        masked_mh_output = self.layer_norm1(masked_mh_output + dec_inputs)

        enc_dec_mh_output = self.enc_dec_mh(
            query=masked_mh_output,
            key=enc_outputs,
            value=enc_outputs,
            attn_mask=dec_enc_padding_mask,
        )
        enc_dec_mh_output = self.layer_norm2(enc_dec_mh_output + masked_mh_output)

        ffnn_output = self.ffnn(enc_dec_mh_output)
        ffnn_output = self.layer_norm3(ffnn_output + enc_dec_mh_output)

        return ffnn_output


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_hidden: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        ff_dim: int,
        max_sequence: int,
        padding_id: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.src_emb = nn.Embedding(input_dim, d_hidden)
        pe_table = get_position_encoding_table(max_sequence, d_hidden)
        self.pos_emb = nn.Embedding.from_pretrained(pe_table, freeze=True)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_hidden=d_hidden,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.padding_id = padding_id
        self.classifier = nn.Linear(d_hidden, input_dim)

    def forward(self, dec_inputs: Tensor, enc_output: Tensor):
        position = get_position(inputs=dec_inputs)
        conb_emb = self.src_emb(dec_inputs) + self.pos_emb(position)
        # Embedding + pos_enbeding : [batch_size, max_seq_size, d_hidden]
        padding_mask = get_padding_mask(
            dec_inputs, self.padding_id
        )  # =>[batch_size, max_seq_size, max_seq_size]
        # print('dec_inputs_size', dec_inputs.size())
        look_ahead_mask = get_look_ahead_mask(dec_inputs)
        # print('dec_mask_size', look_ahead_mask.size())

        dec_self_attn_mask = padding_mask + look_ahead_mask
        dec_enc_padding_mask = get_padding_mask(dec_inputs, self.padding_id)
        dec_outputs = conb_emb
        for layer in self.layers:
            dec_outputs = layer(
                dec_outputs, enc_output, dec_self_attn_mask, dec_enc_padding_mask
            )
            # => [bs, max_seq_size, d_hidden]
        # print(dec_outputs.size())
        dec_outputs = self.classifier(dec_outputs)  # [bs, max_seq_size, input_dim]
        dec_outputs = F.log_softmax(
            dec_outputs, dim=1
        )  # => [bs, max_seq_size, input_dim]
        # print(dec_outputs.size())
        return dec_outputs

        # dec_self_attn_mask + dec_mask : torch.gt => 1번 레이어

        # dec_inputs, enc_output -> get_attn_pad_mask => mask 2번 레이어


class Transformer(nn.Module):
    def __init__(
        self,
        enc_d_input: int,
        enc_layers: int,
        enc_heads: int,
        enc_head_dim: int,
        enc_ff_dim: int,
        dec_d_input: int,
        dec_layers: int,
        dec_heads: int,
        dec_head_dim: int,
        dec_ff_dim: int,
        d_hidden: int,
        max_sequence_size: int,
        dropout_rate: float = 0.0,
        padding_id: int = 3,
    ) -> None:
        super().__init__()
        self.device = get_device()
        self.encoder = Encoder(
            input_dim=enc_d_input,
            d_hidden=d_hidden,
            n_layers=enc_layers,
            n_heads=enc_heads,
            head_dim=enc_head_dim,
            ff_dim=enc_ff_dim,
            max_sequence=max_sequence_size,
            padding_id=padding_id,
            dropout=dropout_rate,
        ).to(self.device)
        self.decoder = Decoder(
            input_dim=dec_d_input,
            d_hidden=d_hidden,
            n_layers=dec_layers,
            n_heads=dec_heads,
            head_dim=dec_head_dim,
            ff_dim=dec_ff_dim,
            max_sequence=max_sequence_size,
            padding_id=padding_id,
            dropout=dropout_rate,
        ).to(self.device)

    def forward(self, enc_inputs: Tensor, dec_input: Tensor) -> Tensor:
        """Transformer

        Args:
            enc_inputs (Tensor): source input tensor (batch_size, max_seq_len)
            dec_input (Tensor): target input tensor (batch_size, max_seq_len)

        Returns:
            Tensor: _description_
        """
        # print('enc_inputs_size', enc_inputs.size())
        # print('dec_input_size', dec_input.size())

        enc_outputs = self.encoder(enc_inputs)

        # print('enc_outputs_size', enc_outputs.size())
        dec_output = self.decoder(dec_input, enc_outputs)
        # print('dec_output_size', dec_output.size())
        return dec_output
