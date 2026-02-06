
import torch 
import torch.nn as nn  

from .Attention import MultiheadSelfAttention
from .Positionwise_fd import PositionwiseFeedForward

from .RMSNorm import RMSNorm
from .Embedding import Embedding
from .Linear import Linear
class Transformer_block(nn.Module): 
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta:float):
        """

        Args:
            d_model: The dimension of the model
            num_heads: The number of heads
            d_ff: The dimension of the feedforward network
            max_seq_len: The maximum sequence length
            theta: The theta parameter for the RoPE
        """

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.norm_1 = RMSNorm(d_model)
        self.norm_2 = RMSNorm(d_model) 

        self.casual_multi_head_attention = MultiheadSelfAttention(d_model, 
        num_heads,use_rope=True,  
        max_seq_len=max_seq_len, 
        theta=theta, token_positions=None)

        self.positionwise_feed_forward = PositionwiseFeedForward(d_model, d_ff)



    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = x + self.casual_multi_head_attention(self.norm_1(x))
        output = y + self.positionwise_feed_forward(self.norm_2(y))
        return output 


class TransformerLM(nn.Module):

    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int,num_heads: int, 
        d_ff: int, theta:float,):

        super().__init__()

        self.vocab_size = vocab_size

        self.context_length = context_length

        
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [Transformer_block(
                d_model, 
                num_heads, 
                d_ff, 
                context_length,
                theta
                ) for _ in range(num_layers)])
        self.rms_norm = RMSNorm(d_model) 
        self.output_embeddings = Linear( d_model, vocab_size)

    def forward(self, in_indices: torch.Tensor):
        """
        Args:
            in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. 
            Shape is (batch_size, sequence_length), where `sequence_length` is at most `context_length`.

        Returns:
            Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
            next-word distribution for each token.
        """
        x = self.token_embeddings(in_indices)

        for layer in self.layers:
            x = layer(x)
        
        x_norm = self.rms_norm(x)
        output_embed = self.output_embeddings(x_norm)
        return output_embed