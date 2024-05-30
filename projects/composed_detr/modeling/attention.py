# -*- coding: utf-8 -*-

"""
This file is authored by Ujjwal.
It was written at Universitat Hildesheim on May 29, 2024.
All rights reserved.
"""

import logging
from typing import Literal, Optional

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
                 add_bias_kv: bool = False,
                 ):
        """
        Args:
            num_heads (int): The number of attention heads.
            model_dim (int): The dimensionality of the model.
            projection_rank (int, optional): The rank of the projection tensor. Defaults to 2.
            use_bias (bool, optional): Whether to use bias in the linear layers. Defaults to True.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            add_zero_attn (bool, optional): Whether to add zero attention. Defaults to False.
            add_bias_kv (bool, optional): Whether to add bias to the key and value. Defaults to False.

        """
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
        self._add_bias_kv = add_bias_kv
        if self._add_bias_kv:
            self._bias_k = nn.Parameter(torch.zeros((1, 1, model_dim)))
            self._bias_v = nn.Parameter(torch.zeros((1, 1, model_dim)))

    def _initialize_compose_params(self):
        """
        Initializes the parameters required for composing tensors in the model.

        Returns:
            dict: A dictionary containing the initialized parameters.

        """
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

    # def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask=None,
    #             attn_mask=None):
    #     """
    #     Args:
    #         query: Tensor containing the query vectors. Shape: (batch_size, sequence_length, hidden_dim).
    #         key: Tensor containing the key vectors. Shape: (batch_size, sequence_length, hidden_dim).
    #         value: Tensor containing the value vectors. Shape: (batch_size, sequence_length, hidden_dim).
    #         key_padding_mask: Tensor containing a mask for padding elements in the key vectors. Shape: (batch_size, sequence_length).
    #         attn_mask: Tensor containing a mask for preventing certain connections. Shape: (batch_size, sequence_length, sequence_length).
    #
    #     Returns:
    #         output: Tensor containing the output of the forward pass. Shape: (batch_size, sequence_length, hidden_dim).
    #         attention_dict: Dictionary containing the attention scores at different stages. Dictionary keys:
    #             - 'uncomposed': Tensor containing the attention scores before composition. Shape: (batch_size, num_heads, sequence_length, sequence_length).
    #             - 'pre_composed': Tensor containing the attention scores after composition but before dropout. Shape: (batch_size, num_heads, sequence_length, sequence_length).
    #             - 'post_composed': Tensor containing the attention scores after composition and after dropout. Shape: (batch_size, num_heads, sequence_length, sequence_length).
    #     """
    #     if self._add_bias_kv:
    #         key = key + self._bias_k
    #         value = value + self._bias_v
    #     query_projected = self._W_query(query)
    #     key_projected = self._W_key(key)
    #     value = self._W_value(value)
    #     if self._add_zero_attn:
    #         zero_pad = torch.zeros((key_projected.size(0), 1) + key_projected.size()[2:], dtype=key_projected.dtype,
    #                                device=key_projected.device)
    #         key_projected = torch.cat((zero_pad, key_projected), dim=1)
    #         value = torch.cat((zero_pad, value), dim=1)
    #
    #     attn_feature_matrix = self._compute_attention_logits(query=query_projected, key=key_projected)
    #
    #     uncomposed_attention = attn_feature_matrix.softmax(dim=-1)
    #
    #     attn_feature_matrix = self._compose(attn_information=attn_feature_matrix,
    #                                         query=query, key=key, projection_type='pre')
    #     if key_padding_mask is not None:
    #         attn_feature_matrix.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
    #
    #     if attn_mask is not None:
    #         if attn_mask.dtype == torch.bool:
    #             attn_feature_matrix.masked_fill_(attn_mask.unsqueeze(1), float('-inf'))
    #         else:
    #             attn_feature_matrix += attn_mask.unsqueeze(1)
    #     attn_probs = attn_feature_matrix.softmax(dim=-1)
    #     pre_composed_attention = attn_probs
    #     attn_probs = self._dropout(attn_probs)
    #     attn_probs = self._compose(attn_information=attn_probs, query=query, key=key,
    #                                projection_type='post')
    #     post_composed_attention = attn_probs
    #     value = value.view(
    #         value.size(0),
    #         self._num_heads,
    #         -1,
    #         self._head_dim
    #     )
    #     output = torch.einsum('B H T S, B H S D -> B H T D', attn_probs, value)
    #     output = einops.rearrange(output, 'B H S D -> B S (H D)')
    #     return output, {
    #         'uncomposed': uncomposed_attention,
    #         'pre_composed': pre_composed_attention,
    #         'post_composed': post_composed_attention
    #     }

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask=None,
                attn_mask=None):
        if self._add_bias_kv:
            key = key.add_(self._bias_k)
            value = value.add_(self._bias_v)
        query_projected = self._W_query(query)
        key_projected = self._W_key(key)
        value = self._W_value(value)
        if self._add_zero_attn:
            zero_pad = torch.zeros((key_projected.size(0), 1) + key_projected.size()[2:], dtype=key_projected.dtype,
                                   device=key_projected.device)
            key_projected = torch.cat((zero_pad, key_projected), dim=1)
            value = torch.cat((zero_pad, value), dim=1)

        attn_feature_matrix = self._compute_attention_logits(query=query_projected, key=key_projected)

        uncomposed_attention = attn_feature_matrix.softmax(dim=-1)

        attn_feature_matrix = self._compose(attn_information=attn_feature_matrix,
                                            query=query, key=key, projection_type='pre')
        if key_padding_mask is not None:
            attn_feature_matrix.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_feature_matrix.masked_fill_(attn_mask.unsqueeze(1), float('-inf'))
            else:
                attn_feature_matrix.add_(attn_mask.unsqueeze(1))
        attn_probs = attn_feature_matrix.softmax(dim=-1)
        pre_composed_attention = attn_probs
        attn_probs = self._dropout(attn_probs)
        attn_probs = self._compose(attn_information=attn_probs, query=query, key=key,
                                   projection_type='post')
        post_composed_attention = attn_probs
        value = value.view(
            value.size(0),
            self._num_heads,
            -1,
            self._head_dim
        )
        output = torch.einsum('B H T S, B H S D -> B H T D', attn_probs, value)
        output = einops.rearrange(output, 'B H S D -> B S (H D)')
        del query_projected, key_projected, attn_feature_matrix, attn_probs, value  # delete unnecessary tensors
        torch.cuda.empty_cache()  # free up GPU memory
        return output, {
            'uncomposed': uncomposed_attention,
            'pre_composed': pre_composed_attention,
            'post_composed': post_composed_attention
        }

    def _compose(self, attn_information: torch.Tensor, query: torch.Tensor, key: torch.Tensor,
                 projection_type: Literal['pre', 'post']):
        """
        Args:
            attn_information: A tensor representing the attention information.
            query: A tensor representing the query.
            key: A tensor representing the key.
            projection_type: A string indicating the type of projection (either 'pre' or 'post').

        Returns:
            A tensor representing the composed attention information.

        Raises:
            None

        """
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
            query: A torch.Tensor representing the queries for computing attention logits. It has shape (batch_size, num_heads, sequence_length, head_dim).
            key: A torch.Tensor representing the keys for computing attention logits. It has shape (batch_size, num_heads, sequence_length, head_dim).

        Returns:
            A torch.Tensor representing the computed attention logits. It has shape (batch_size, num_heads, sequence_length, sequence_length).
        """
        query = query.view(query.size(0), self._num_heads, -1, self._head_dim)
        key = key.view(key.size(0), self._num_heads, -1, self._head_dim)
        scale = self._head_dim ** 0.5  # Normalization factor
        _attn = torch.einsum('B H T D, B H S D -> B H T S', query, key) / scale
        return _attn

    def _dynamic_weight_projection(self, query: torch.Tensor, key: torch.Tensor,
                                   projection_params):
        """
        Args:
            query: A torch.Tensor representing the input query.
            key: A torch.Tensor representing the input key.
            projection_params: A dictionary containing the projection parameters.

        Returns:
            A dictionary with keys 'query' and 'key', where the value for each key is a list of two torch.Tensors.

        Notes:
            - The method performs dynamic weight projection on the query and key tensors using the given projection parameters.
            - The projection parameters should contain the keys: 'q1', 'q2', 'k1', 'k2'.
            - The method applies the GELU activation function to the intermediate products in the projection.
            - The query and key tensors are divided into two halves, and each half is rearranged to have dimensions 'B T R H' or 'B S R H',
              where B is the batch size, T is the sequence length (for query), S is the sequence length (for key), R is the projection rank,
              and H is the hidden size.

        Example usage:
            # Assume projection_params is a dictionary with the required keys
            query_tensor = torch.tensor([1, 2, 3])
            key_tensor = torch.tensor([4, 5, 6])
            projection_result = _dynamic_weight_projection(query_tensor, key_tensor, projection_params)
        """
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


class DynamicallyComposedMultiHeadAttentionWrapper(nn.Module):
    """
    Class DynamicallyComposedMultiHeadAttentionWrapper

    This class provides a wrapper around the DynamicallyComposedMultiHeadAttention class,
    allowing for easier use and integration with other operations in a Transformer architecture.

    Args:
        embed_dim (int): The dimension of the input embeddings.
        num_heads (int): The number of attention heads.
        attn_drop (float, optional): Dropout probability for attention weights. Default: 0.0.
        proj_drop (float, optional): Dropout probability for projection weights. Default: 0.0.
        batch_first (bool, optional): If True, the input tensors have shape (batch_size, seq_len, embed_dim).
            If False, the input tensors have shape (seq_len, batch_size, embed_dim). Default: False.
        projection_rank (int, optional): The rank of the projection matrix. Default: 2.
        **kwargs: Additional keyword arguments.

    Attributes:
        embed_dim (int): The dimension of the input embeddings.
        num_heads (int): The number of attention heads.
        batch_first (bool): If True, the input tensors have shape (batch_size, seq_len, embed_dim).
            If False, the input tensors have shape (seq_len, batch_size, embed_dim).
        attn (DynamicallyComposedMultiHeadAttention): The internally used DynamicallyComposedMultiHeadAttention module.
        proj_drop (nn.Dropout): The dropout layer for projection.

    Methods:
        forward(query, key=None, value=None, identity=None, query_pos=None, key_pos=None, attn_mask=None,
                key_padding_mask=None, **kwargs) -> torch.Tensor:
            Forward function for the DynamicallyComposedMultiHeadAttentionWrapper.

    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            batch_first: bool = False,
            projection_rank: int = 2,
            **kwargs,
    ):
        super(DynamicallyComposedMultiHeadAttentionWrapper, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = DynamicallyComposedMultiHeadAttention(
            model_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            projection_rank=projection_rank
        )

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            query: torch.Tensor,
            key: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None,
            identity: Optional[torch.Tensor] = None,
            query_pos: Optional[torch.Tensor] = None,
            key_pos: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            query: The query tensor (batch_size, query_seq_len, embed_dim).
            key: The key tensor (batch_size, key_seq_len, embed_dim). If None, default to using query.
            value: The value tensor (batch_size, value_seq_len, embed_dim). If None, default to using key.
            identity: The identity tensor (batch_size, query_seq_len, embed_dim). If None, default to using query.
            query_pos: The positional encoding tensor for query (batch_size, query_seq_len, embed_dim). If None, positional encoding is not applied to query.
            key_pos: The positional encoding tensor for key (batch_size, key_seq_len, embed_dim). If None, and query_pos is not None, positional encoding is applied to key using query_pos.
            attn_mask: The attention mask tensor (batch_size, query_seq_len, key_seq_len). If None, no attention mask is applied.
            key_padding_mask: The padding mask tensor for key (batch_size, key_seq_len). If None, no padding mask is applied to key.
            **kwargs: Additional keyword arguments that are passed to the underlying attention mechanism.

        Returns:
            The output tensor after applying the attention mechanism and projection (batch_size, query_seq_len, embed_dim).
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    logger.warning(
                        f"position encoding of key is" f"missing in {self.__class__.__name__}."
                    )
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]
        out = identity + self.proj_drop(out)
        return out


if __name__ == "__main__":
    # model = DynamicallyComposedMultiHeadAttention(
    #     num_heads=8,
    #     model_dim=2048,
    #     add_bias_kv=True,
    #     dropout=0.2,
    #     add_zero_attn=False
    # )
    # X_input = torch.randn((5, 49, 2048))
    #
    # output, out2 = model(X_input, X_input, X_input)
    # print(output.size())
    # for key, value in out2.items():
    #     print(f"Shape of tensor at key '{key}': {value.shape}")
    #     print(f"Minimum value in tensor at key '{key}': {value.min()}")
    #     print(f"Maximum value in tensor at key '{key}': {value.max()}")
    #     print(f"Sum of elements in tensor at key '{key}' for index [0,0,0,:]: {value[0, 0, 0, :].sum()}")

    model = DynamicallyComposedMultiHeadAttentionWrapper(
        num_heads=8,
        embed_dim=2048,
    )
    X_input = torch.randn((5, 49, 2048))

    output = model(X_input, key_padding_mask=torch.rand(5, 49) > 0.5)
    print(output.size())
    print("PyTorch version: ", torch.__version__)
    print("CUDA version: ", torch.version.cuda)
    print("cuDNN version: ", torch.backends.cudnn.version())
    # for key, value in output.items():
    #     print(f"Shape of tensor at key '{key}': {value.shape}")
    #     print(f"Minimum value in tensor at key '{key}': {value.min()}")
    #     print(f"Maximum value in tensor at key '{key}': {value.max()}")
    #     print(f"Sum of elements in tensor at key '{key}' for index [0,0,0,:]: {value[0, 0, 0, :].sum()}")
