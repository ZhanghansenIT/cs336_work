# Implement the position-wise feed-forward networ
from numpy import int32
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from .Linear import Linear

def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
class Positionwise_fd(nn.Module):

    def __init__(self, d_model: int, d_ff: int):

        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
      
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output = self.w2(silu(self.w1(x))*self.w3(x))
        return output 

# 添加别名以兼容不同的导入方式
PositionwiseFeedForward = Positionwise_fd