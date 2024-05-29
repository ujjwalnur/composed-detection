import logging
from typing import Literal

import einops
import torch
from torch import nn

logger = logging.getLogger(__name__)


class DynamicallyComposedMultiHeadAttention(nn.Module):
    def __init__(self,
                 num_heads: int,
                 model_dim: int,
                 projection_rank: int = 2,
                 use_bias: bool = True,
                 dropout: float = 0.0,
                 add_zero_attn: bool = False,
                 ):
        super(DynamicallyComposedMultiHeadAttention, self).__init__()
        if model_dim % num_heads != 0:
            logger.error(f'model_dim {model_dim} must be divisible by num_heads {num_heads}.')
            raise ValueError
        self._num_heads = num_heads
        self._model_dim = model_dim
        self._head_dim = model_dim // num_heads
        self._projection_rank = projection_rank
        self._use_bias = use_bias
        self._W_query = nn.Linear(model_dim, model_dim, bias=self._use_bias)
        self._W_key = nn.Linear(model_dim, model_dim, bias=self._use_bias)
        self._W_value = nn.Linear(model_dim, model_dim, bias=self._use_bias)
        self._pre_compose_params = nn.ParameterDict(self._initialize_compose_params())
        self._post_compose_params = nn.ParameterDict(self._initialize_compose_params())
        self._dropout = nn.Dropout(dropout)
        self._add_zero_attn = add_zero_attn  # Initialize add_zero_attn

    def _initialize_compose_params(self):
        out = dict()
        out['q1'] = nn.Parameter(
            torch.empty(
                self._model_dim,
                2 * self._num_heads * self._projection_rank,
            )
        )

        nn.init.xavier_uniform_(out['q1'])

        out['q2'] = nn.Parameter(
            torch.empty(
                2 * self._num_heads * self._projection_rank,
                2 * self._num_heads * self._projection_rank,
            )
        )

        nn.init.xavier_uniform_(out['q2'])

        out['k1'] = nn.Parameter(
            torch.empty(
                self._model_dim,
                2 * self._num_heads * self._projection_rank,
            )
        )
        nn.init.xavier_uniform_(out['k1'])

        out['k2'] = nn.Parameter(
            torch.empty(
                2 * self._num_heads * self._projection_rank,
                2 * self._num_heads * self._projection_rank,
            )
        )

        nn.init.xavier_uniform_(out['k2'])

        out['qg'] = nn.Parameter(
            torch.empty(
                self._model_dim,
                self._num_heads
            )
        )

        nn.init.xavier_uniform_(out['qg'])

        out['kg'] = nn.Parameter(
            torch.empty(
                self._model_dim,
                self._num_heads
            )
        )

        nn.init.xavier_uniform_(out['kg'])
        return out

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask=None,
                attn_mask=None):
        query_projected = self._W_query(query)
        key_projected = self._W_key(key)
        value = self._W_value(value)
        if self._add_zero_attn:
            zero_pad = torch.zeros((key_projected.size(0), 1) + key_projected.size()[2:], dtype=key_projected.dtype,
                                   device=key_projected.device)
            key_projected = torch.cat((zero_pad, key_projected), dim=1)
            value = torch.cat((zero_pad, value), dim=1)

        attn_feature_matrix = self._compute_attention_logits(query=query_projected, key=key_projected)

        if key_padding_mask is not None:
            attn_feature_matrix.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_feature_matrix.masked_fill_(attn_mask.unsqueeze(1), float('-inf'))
            else:
                attn_feature_matrix += attn_mask.unsqueeze(1)

        attn_feature_matrix = self._compose(attn_information=attn_feature_matrix,
                                            query=query, key=key, projection_type='pre')
        attn_probs = attn_feature_matrix.softmax(dim=-1)
        attn_probs = self._dropout(attn_probs)
        attn_probs = self._compose(attn_information=attn_probs, query=query, key=key,
                                   projection_type='post')

        attn_probs = attn_probs.view(
            attn_probs.size(0),
            self._num_heads,
            -1,
            key.size(1)
        )
        value = value.view(
            value.size(0),
            self._num_heads,
            -1,
            self._head_dim
        )
        output = torch.einsum('B H T S, B H S D -> B H T D', attn_probs, value)
        output = einops.rearrange(output, 'B H S D -> B S (H D)')
        return output

    def _compose(self, attn_information: torch.Tensor, query: torch.Tensor, key: torch.Tensor,
                 projection_type: Literal['pre', 'post']):
        if projection_type == 'pre':
            projection_params = self._pre_compose_params
        else:
            projection_params = self._post_compose_params

        projection_wt_dict = self._dynamic_weight_projection(
            query=query, key=key, projection_params=projection_params
        )
        h = torch.einsum('B H T S, B T R H -> B R T S', attn_information, projection_wt_dict['query'][0])
        o_qp = torch.einsum('B R T S, B T R H -> B H T S', h, projection_wt_dict['query'][1])

        h = torch.einsum('B H T S, B S R H -> B R T S', attn_information, projection_wt_dict['key'][0])
        o_kp = torch.einsum('B R T S, B S R H -> B H T S', h, projection_wt_dict['key'][1])

        o_qg = torch.einsum('B H T S, B T H -> B H T S', attn_information,
                            nn.functional.tanh(query @ projection_params['qg']))
        o_kg = torch.einsum('B H T S, B S H -> B H T S', attn_information,
                            nn.functional.tanh(key @ projection_params['kg']))

        return attn_information + o_qp + o_kp + o_qg + o_kg

    def _compute_attention_logits(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """

        Args:
            query: Tensor of shape [batch_size, query_len, model_dim].
            key: Tensor of shape [batch_size, key_len, model_dim].

        Returns:

        """
        query = query.view(query.size(0), self._num_heads, -1, self._head_dim)
        key = key.view(key.size(0), self._num_heads, -1, self._head_dim)
        scale = self._head_dim ** 0.5  # Normalization factor
        _attn = torch.einsum('B H T D, B H S D -> B H T S', query, key) / scale
        return _attn

    def _dynamic_weight_projection(self, query: torch.Tensor, key: torch.Tensor,
                                   projection_params):
        dw_query = nn.functional.gelu(query @ projection_params['q1']) @ projection_params['q2']
        dw_key = nn.functional.gelu(key @ projection_params['k1']) @ projection_params['k2']
        dw_q1, dw_q2 = dw_query.chunk(2, dim=-1)
        dw_k1, dw_k2 = dw_key.chunk(2, dim=-1)
        dw_q1 = einops.rearrange(dw_q1, 'B T (R H) -> B T R H', R=self._projection_rank)
        dw_q2 = einops.rearrange(dw_q2, 'B T (R H) -> B T R H', R=self._projection_rank)

        dw_k1 = einops.rearrange(dw_k1, 'B S (R H) -> B S R H', R=self._projection_rank)
        dw_k2 = einops.rearrange(dw_k2, 'B S (R H) -> B S R H', R=self._projection_rank)
        return {
            'query': [dw_q1, dw_q2],
            'key': [dw_k1, dw_k2],
        }


if __name__ == "__main__":
    model = DynamicallyComposedMultiHeadAttention(
        num_heads=8,
        model_dim=2048,
        add_zero_attn=False
    )
    X_input = torch.randn((5, 49, 2048))

    output = model(X_input, X_input, X_input)
    print(output.size())
