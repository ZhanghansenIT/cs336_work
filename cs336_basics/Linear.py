
# 实现无偏置的线性变换
import torch 
import torch.nn as nn 
import numpy as np  


class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        y = Wx 
        Initialize a linear layer with weight matrix W.
        Args:
            in_features: int, the number of input features
            out_features: int, the number of output features
            device: str, the device to use
            dtype: torch.dtype, the dtype to use
        """
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.device = device 
        self.dtype = dtype 
        self.in_features = in_features 
        self.out_features = out_features 
    
        self.weight = nn.Parameter(self.init_weights(in_features, out_features, factory_kwargs))


    def init_weights(self, in_features: int, out_features: int, factory_kwargs: dict) -> torch.Tensor:

        W = torch.empty(out_features, in_features,**factory_kwargs) 
        mean = 0 
        std = np.sqrt(2 / (in_features+out_features)) 
        nn.init.trunc_normal_(W, mean, std,-3*std,3*std)
        print(W)
        return W 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x @ self.weight.T 
        return output 


"""

参数说明
in_features = 64：输入特征维度
out_features = 128：输出特征维度
输入 x.shape = [4, 12, 64]：(batch_size, sequence_length, in_features)
权重 w.shape = [128, 64]：(out_features, in_features)
输出 y.shape = [4, 12, 128]：(batch_size, sequence_length, out_features)
步骤分解：
转置权重矩阵：

self.weight.shape = [128, 64]

self.weight.T.shape = [64, 128]

矩阵乘法：

x @ self.weight.T 相当于：[4, 12, 64] @ [64, 128]

根据PyTorch的广播规则，这相当于对每个batch、每个序列位置分别进行矩阵乘法

实际计算：

text
对于每个 batch i (0-3) 和 每个序列位置 j (0-11):
    x[i, j, :].shape = [64] (向量)
    执行：x[i, j, :] @ self.weight.T
    = [1×64] @ [64×128] = [1×128]

最终得到：y[i, j, :].shape = [128]

"""