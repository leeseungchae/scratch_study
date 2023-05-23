from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import Tensor
from torch.autograd import Variable


class Scaled_dot_product_Attention(nn.Module):
    def __init__(self, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor,mask=None ,dropout=None
    ) -> Tuple[Tensor, Tensor]:
        """dot product attention

        Args:
            query (Tensor): t 시점의 decoder cell에서의 hidden state
            key (Tensor): 모든 시점의 encoder cell에서의 hidden state
            value (Tensor): 모든 시점의 encoder cell에서의 hidden state

        Returns:
            _type_: _description_
            :param mask:
        """
        #https://cpm0722.github.io/pytorch-implementation/transformer
        # 벡터의 차원을 얻음
        d_k = query.size(-1)
        print(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)

        if dropout is not None:
            attention_weights = dropout(attention_weights)

        output = torch.matmul(attention_weights, value)
        return output, attention_weights

        scores = torch.matmul(query, key.transpose(-1, -2))
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        context = torch.matmul(attn_prob, value).squeeze()

        print(f"scores {scores.size()}")
        return context, attn_prob

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int = 512, dropout: float = 0.0, max_len: int = 5000):

        # https://wikidocs.net/31379
        # https://sirzzang.github.io/ai/AI-Transformer-02/
        # mex_len 논문 기준
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # 벡터 초기화 (벡터간 거리 일정)
        pe = torch.zeros(max_len, hidden_dim)  # torch.Size([5000, 512])
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # torch.Size([5000, 1])

        # torch.exp: 자연상수, log: 자연 로그, 홀수 인덱스의 경우 cos함수, 짝수 인덱스의 경우 sin함수 주기
        div_term = torch.exp((torch.arange(0, hidden_dim, 2) *
                              -(math.log(10000.0) / hidden_dim)).float())  # torch.Size([256])

        # sin,cos 함수를 이용하기 때문에, 모두 -1 ~ 1 사이로 통일 벡터가 발산하지 않는다.
        pe[:, 0::2] = torch.sin(position.float() * div_term)  # torch.Size([5000, 512])
        pe[:, 1::2] = torch.cos(position.float() * div_term)  # torch.Size([5000, 512])

        pe = pe.unsqueeze(0)  # torch.Size([1, 5000, 512])

        # pe를 모델의 버퍼로 등록 (파타미터로 학습 안되게, 역전파에 의해 변경되지 않는다,)
        self.register_buffer('pe', pe)

    # 입력 데이터에 PositionalEncoding 추가하는 과정

    # legacy pytorch 이전버전
    # def forward(self, x):
    #
    #     x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
    #     return self.dropout(x)
    # 잘 모르겠음..
    # https://github.com/dreamgonfly/transformer-pytorch/blob/master/embeddings.py
    def forward(self, x, step=None):
        x = self.embbedding(x)
        x = x * math.sqrt(self.dim)
        if step is None:
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:, step]
        x = self.dropout(x)
        return x

