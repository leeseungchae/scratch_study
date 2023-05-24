import math
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
            query (Tensor): t 시점의 decoder cell에서의 hidden state
            key (Tensor): 모든 시점의 encoder cell에서의 hidden state
            value (Tensor): 모든 시점의 encoder cell에서의 hidden state

        Returns:
            _type_: _description_
        """
        scores = torch.matmul(query.unsqueeze(1), key.transpose(-1, -2)) / self.scale
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        context = torch.matmul(attn_prob, value).squeeze()
        return context, attn_prob


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
        self.padding_id = padding_id

    def forward(self, enc_inputs: Tensor):
        position = self.get_position(enc_inputs=enc_inputs)
        conb_emb = self.src_emb(enc_inputs) + self.pos_emb(position)

    def get_position(self, enc_inputs: Tensor) -> Tensor:
        position = torch.arange(
            enc_inputs.size(1), device=enc_inputs.device, dtype=enc_inputs.dtype
        ).expand(enc_inputs.size(0), enc_inputs.size(1).contiguous())
        pos_mask = enc_inputs.eq(self.padding_id)
        position.masked_fill(pos_mask, 0)
        return position


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

    def forward(self, enc_inputs: Tensor, dec_inputs: Tensor) -> Tensor:
        enc_outputs = self.encoder(enc_inputs)
        a = self.src_emb(enc_inputs)  # => [batch_size, mex_seq, d_hidden]

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
