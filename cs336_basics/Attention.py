

import torch
import torch.nn as nn
from .RoPE import RoPE 
from einops import einsum, rearrange
from .Linear import Linear
from .Softmax import Softmax
# def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    
#     x_max = torch.max(x, dim=-1, keepdim=True).values
#     x_stable = x - x_max 
#     x_exp = torch.exp(x_stable)
#     output = x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
    
#     return output 

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of scaled dot product attention.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    # 获取 query, key, value 的维度
    d_k = torch.tensor(Q.shape[-1])
    softmax = Softmax(dim=-1)
    # 计算注意力
    attention_score = Q @ K.transpose(-2, -1) / torch.sqrt(d_k)
    
    if mask is not None:
        # 这里mask为False的位置填充-inf（被屏蔽）
        attention_score = attention_score.masked_fill(~mask, float('-inf')) # fill the mask false value with -inf
    
    
    output = softmax(attention_score) @ V
    # print(attention_score[0,0,:,:])
    return output 


class MultiheadSelfAttention(nn.Module):
    
    def __init__(self,d_model: int, num_heads: int, use_rope: bool = False,
                 max_seq_len: int | None = None, theta: float | None = None,
                 token_positions: torch.Tensor | None = None):
        super().__init__() 
        
        # d_model 是token的总的嵌入维度，比如 512 
        
        self.d_model = d_model # token的总的嵌入维度，比如 512 
        self.num_heads = num_heads # 头的数量，比如 8
        self.use_rope = use_rope
        self.rope = RoPE(theta, d_model//num_heads, max_seq_len) if use_rope else None 

        self.token_positions = token_positions  
        
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        
        batch_size, seq_len, d_model = x.shape 
        # 将 Q、K、V 三个投影层的权重拼接在一起
        qkv_proj = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight])
        #  一次性计算 Q、K、V 的投影
        qkv = x @ qkv_proj.T 
        # 将结果分割成 Q、K、V 三部分
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = rearrange(
            q, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )
        k = rearrange(
            k, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )
        v = rearrange(
            v, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )

        if self.use_rope:
            q = self.rope(q, self.token_positions)
            k = self.rope(k, self.token_positions)
        
        casual_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        casual_mask = casual_mask[None, None, :, :] 
        # print(f"casual_mask: {casual_mask}")
        
        print(q.shape, k.shape, v.shape)
        print(casual_mask.shape)
        output = scaled_dot_product_attention(q, k, v, ~casual_mask)
        
        output = rearrange(
            output, "... h seq_len d_head ->  ... seq_len (h d_head)"
        )
        return self.out_proj(output)
         
        
        
        
        
        