from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def get_position_encoding_table(mex_sequence_size: int, d_hidden: int):
    def get_angle(position: int, i: int) -> float:
        return position / np.power(10000, 2 * (i // 2) / d_hidden)

    def get_angle_vector(position: int) -> List[float]:
        return [get_angle(position, hid_j) for hid_j in range(d_hidden)]

    pe_table = torch.Tensor(
        [get_angle_vector(pos_i) for pos_i in range(mex_sequence_size)]
    )

    pe_table[:, 0::2] = np.sin(pe_table[:, 0::2])
    pe_table[:, 1::2] = np.cos(pe_table[:, 1::2])

    return pe_table


def get_attn_pad_mask(seq_q: Tensor, seq_k: Tensor, padding_id: int) -> Tensor:
    pad_attn_mask = seq_k.data.eq(padding_id).unsqueeze(1)
    return pad_attn_mask.expand(
        seq_k.size(0), seq_q.size(1), seq_k.size(1).contiguous()
    )


def get_attn_decoder_mask(dec_input: Tensor) -> Tensor:
    subsequent_mask = (
        torch.ones_like(dec_input)
        .unsqueeze(-1)
        .expand(dec_input.size(0), dec_input.size(1), dec_input.size(1))
    )
    # subsequent_mask = subsequent_mask.


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
        scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale
        # => [bs, n_heads, len_q(max_seq_size), len_k(max_seq_size)]
        scores.masked_fill(attion_mask, -1e9)
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        context = torch.matmul(attn_prob, value).squeeze()
        return context, attn_prob

    def get_attn_pad_mask(seq_q: Tensor, seq_k: Tensor, padding_id: int) -> Tensor:
        pad_attn_mask = seq_k.data.eq(padding_id).unsqueeze(1)
        # pad_attn_mask =
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
        print("query: ", query.size())
        print("weight_q(query): ", self.weight_q)

        q_s = (
            self.weight_q(query)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transport(1, 2)
        )
        # => [batch_size, n_heads, max_seq_len, head_dim]
        k_s = (
            self.weight_k(key)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transport(1, 2)
        )
        # => [batch_size, n_heads, max_seq_len, head_dim]
        v_s = (
            self.weight_v(value)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transport(1, 2)
        )
        # => [batch_size, n_heads, max_seq_len, head_dim]

        context, _ = self.self_attention(
            q_s, k_s, v_s, attn_mask
        )  # => [batch_size, n_heads, max_seq_len, head_dim]

        context(
            context.transpose(1, 2)
            .contiguous()
            .view()[batch_size, -1, self.n_heads * self.head_dim]
        )  # =>[batch_size, max_seq_len, n_heads*head_dim]
        output = self.linear(context)
        output = self.dropout(output)
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

    # def forward(self, enc_input):


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
            EncoderLayer(
                d_hidden=d_hidden,
                n_heads=n_heads,
                head_dim=head_dim,
                ff_dim=ff_dim,
                dropout=dropout,
                padding_id=padding_id,
            )
        )
        self.padding_id = padding_id

    def forward(self, enc_inputs: Tensor):
        position = self.get_position(enc_inputs=enc_inputs)
        conb_emb = self.src_emb(enc_inputs) + self.pos_emb(position)
        # enc_self_attn_mask =

    def get_position(self, enc_inputs: Tensor) -> Tensor:
        position = torch.arange(
            enc_inputs.size(1), device=enc_inputs.device, dtype=enc_inputs.dtype
        ).expand(enc_inputs.size(0), enc_inputs.size(1).contiguous())
        pos_mask = enc_inputs.eq(self.padding_id)
        position.masked_fill(pos_mask, 0)
        return position


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
        self.mh_attention = MultiHeadAttention(
            d_hidden=d_hidden, n_heads=n_heads, head_dim=head_dim, dropout=dropout
        )
        self.layer_norm1 = nn.LayerNorm(d_hidden, eps=layer_norm_epsilon)
        self.ffnn = PoswiseFeedForwardNet(
            d_hidden=d_hidden, ff_dim=ff_dim, dropout=dropout
        )
        self.layer_norm2 = nn.LayerNorm(d_hidden, eps=layer_norm_epsilon)


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
        )
        self.decoder = Decoder(
            input_dim=enc_d_input,
            d_hidden=d_hidden,
            n_layers=enc_layers,
            n_heads=enc_heads,
            head_dim=enc_head_dim,
            ff_dim=enc_ff_dim,
            max_sequence=max_sequence_size,
            padding_id=padding_id,
            dropout=dropout_rate,
        )

    def forward(
        self, dec_inputs: Tensor, enc_inputs: Tensor, enc_outputs: Tensor
    ) -> Tensor:
        position = self.get_position(enc_inputs=enc_inputs)
        conb_emb = self.src_emb(dec_inputs) + self.pos_emb(position)

        dec_self_attn_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.padding_id)
        dec_mask = get_attn_decoder_mask(dec_inputs)

        # dec_self.attn_mask  + dec_mask : torch.gt => 1번 레이어
        # dec_inputs, enc_ouput  => get_attn_pad_mask => 2번 레이어

    def get_position(self, enc_inputs: Tensor) -> Tensor:
        position = torch.arange(
            enc_inputs.size(1), device=enc_inputs.device, dtype=enc_inputs.dtype
        ).expand(enc_inputs.size(0), enc_inputs.size(1).contiguous())
        pos_mask = enc_inputs.eq(self.padding_id)


# class Scaled_dot_product_Attention(nn.Module):
#     def __init__(self, dropout_rate: float = 0.0) -> None:
#         super().__init__()
#         self.dropout = nn.Dropout(dropout_rate)
#
#     def forward(
#         self, query: Tensor, key: Tensor, value: Tensor, mask=None, dropout=None
#     ) -> Tuple[Tensor, Tensor]:
#         """dot product attention
#
#         Args:
#             query (Tensor): t 시점의 decoder cell에서의 hidden state
#             key (Tensor): 모든 시점의 encoder cell에서의 hidden state
#             value (Tensor): 모든 시점의 encoder cell에서의 hidden state
#
#         Returns:
#             _type_: _description_
#             :param mask:
#         """
#         # https://cpm0722.github.io/pytorch-implementation/transformer
#         # 벡터의 차원을 얻음
#         d_k = query.size(-1)
#         print(d_k)
#         scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
#             torch.tensor(d_k, dtype=torch.float32)
#         )
#
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)
#
#         attention_weights = F.softmax(scores, dim=-1)
#         print("test")
#         if dropout is not None:
#             attention_weights = dropout(attention_weights)
#
#         output = torch.matmul(attention_weights, value)
#         return output, attention_weights
#
#         scores = torch.matmul(query, key.transpose(-1, -2))
#         attn_prob = nn.Softmax(dim=-1)(scores)
#         attn_prob = self.dropout(attn_prob)
#         context = torch.matmul(attn_prob, value).squeeze()
#
#         print(f"scores {scores.size()}")
#         return context, attn_prob
#
#
# class PositionalEncoding(nn.Module):
#     def __init__(
#         self, hidden_dim: int = 512, dropout: float = 0.0, max_len: int = 5000
#     ):
#         # https://wikidocs.net/31379
#         # https://sirzzang.github.io/ai/AI-Transformer-02/
#         # mex_len 논문 기준
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#
#         # 벡터 초기화 (벡터간 거리 일정)
#         pe = torch.zeros(max_len, hidden_dim)  # torch.Size([5000, 512])
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
#             1
#         )  # torch.Size([5000, 1])
#
#         # torch.exp: 자연상수, log: 자연 로그, 홀수 인덱스의 경우 cos함수, 짝수 인덱스의 경우 sin함수 주기
#         div_term = torch.exp(
#             (torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim)).float()
#         )  # torch.Size([256])
#
#         # sin,cos 함수를 이용하기 때문에, 모두 -1 ~ 1 사이로 통일 벡터가 발산하지 않는다.
#         pe[:, 0::2] = torch.sin(position.float() * div_term)  # torch.Size([5000, 512])
#         pe[:, 1::2] = torch.cos(position.float() * div_term)  # torch.Size([5000, 512])
#
#         pe = pe.unsqueeze(0)  # torch.Size([1, 5000, 512])
#
#         # pe를 모델의 버퍼로 등록 (파타미터로 학습 안되게, 역전파에 의해 변경되지 않는다,)
#         self.register_buffer("pe", pe)
#
#     # 입력 데이터에 PositionalEncoding 추가하는 과정
#
#     # legacy pytorch 이전버전
#     # def forward(self, x):
#     #
#     #     x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
#     #     return self.dropout(x)
#     # 잘 모르겠음..
#     # https://github.com/dreamgonfly/transformer-pytorch/blob/master/embeddings.py
#     def forward(self, x, step=None):
#         x = self.embbedding(x)
#         x = x * math.sqrt(self.dim)
#         if step is None:
#             x = x + self.pe[:, : x.size(1)]
#         else:
#             x = x + self.pe[:, step]
#         x = self.dropout(x)
#         return x
#
