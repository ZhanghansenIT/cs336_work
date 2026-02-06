import torch

import torch.nn as nn
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    
    x_max = torch.max(x, dim=-1, keepdim=True).values
    x_stable = x - x_max 
    x_exp = torch.exp(x_stable)
    output = x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
    
    return output 

class Softmax(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return softmax(x, self.dim)