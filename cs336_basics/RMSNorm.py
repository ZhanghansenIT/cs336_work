

import torch 
import torch.nn as nn 
import numpy as np  
# Root Mean Square Layer Normalization

# https://arxiv.org/abs/1910.07467

# RMSNorm is a layer normalization technique that normalizes the input by the root mean square of the input.
# It is a variant of Layer Normalization that is more stable and faster to compute.
# It is also known as Root Mean Square Layer Normalization.
# It is used in the Transformer model to stabilize the training of the model.



class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Initialize an RMSNorm layer.
        Args:
            d_model: int, the dimension of the model
            eps: float, the epsilon value for numerical stability
            device: str, the device to use
            dtype: torch.dtype, the dtype to use
        """
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(self.init_weights(d_model, eps, factory_kwargs))


    def init_weights(self, d_model: int, eps: float,  factory_kwargs: dict) -> torch.Tensor:
        W = torch.ones(d_model,**factory_kwargs)
        
        return W    # 初始化权重为1

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        in_dtype = x.dtype
        # 先将其转换成 torch.float32 防止溢出
        x = x.to(torch.float32)


        rms = torch.sqrt(1/self.d_model * torch.sum(x**2, dim=-1, keepdim=True) + self.eps)

        result = (x / rms) * self.weight
        result = result.to(in_dtype)
        return result