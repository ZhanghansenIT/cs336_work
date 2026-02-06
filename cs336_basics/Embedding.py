


from typing import Sequence
import torch 
import torch.nn as nn 
import numpy as np  


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Initialize an embedding layer.
        Args:
            num_embeddings: int, 词汇表的大小 ,有多少token 就有多大，根据BPE合并得到的
            embedding_dim: int, the dimension of the embeddings
            device: str, the device to use
            dtype: torch.dtype, the dtype to use
        """
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_embeddings = num_embeddings

        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(self.init_weights(num_embeddings, embedding_dim, factory_kwargs))

    def init_weights(self, vocab_size: int, d_model: int,  factory_kwargs: dict) -> torch.Tensor:

        W = torch.empty(vocab_size, d_model,**factory_kwargs)
        mean = 0 
        std = 1 
        nn.init.trunc_normal_(W, mean, std,-3,3)
        return W 

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size , sequence_length = token_ids.shape
        output = torch.empty(batch_size, sequence_length, self.embedding_dim)
        for i, seq in enumerate(token_ids):
            for j ,token_id in enumerate(seq):
                output[i, j, :] = self.weight[token_id]
        return output