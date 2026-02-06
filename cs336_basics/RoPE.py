import torch 

import torch.nn as nn   


class RoPE(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device= None):
        """
        Args:
            theta: The theta parameter for the RoPE
            d_k: The dimension of the key and query
            max_seq_len: The maximum sequence length
            device: The device to use
            dtype: The dtype to use
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        # 预计算旋转矩阵表
        self.rotation_matrix_table = self.get_rotation_matrix(theta, d_k, max_seq_len)
     
    def rotation_block(self, theta: float, block_index: int, seq_pos: int, d_k: int) -> torch.Tensor:
        """
        Args:
            theta: The theta parameter for the RoPE
            block_index: The index of the block
            seq_pos: The position of the sequence i.e. token_positions
            d_k: The dimension of the key and query
        Returns:
            Float[Tensor, "2 2"]: The rotation matrix
        """
        # block_index = (0, 1, 2, 3)
        # 公式为 ： (theta **(2*block_index/d_k))
        
        # angle_i^(k) = i / (theta **(2*k-1/d_k))
        
        angle = torch.tensor(seq_pos/ (theta **(2*block_index/d_k)))
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        r_matrix = torch.tensor([[cos, -sin], [sin, cos]])
        return r_matrix


    def get_rotation_matrix(self, theta: float, d_k: int, max_seq_len: int) -> torch.Tensor:

        rotation_matrix_table = torch.zeros(max_seq_len, d_k, d_k)
        for i in range(max_seq_len):
            # k(j) 取 {1，2,.....,d_k/2}
            # i 取 {1,2,.....,max_seq_len} 这是token_positions
            # block 就是得到的 d_k//2 个 rotation_block，然后要将它们进行拼接，得到一个 d_k x d_k 的矩阵
            block = [self.rotation_block(theta, j, i, d_k) for j in range(d_k//2)]
            rotation_matrix_table[i,:,:] = torch.block_diag(*block)
        return rotation_matrix_table

    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Run RoPE for a given input tensor.
        Args:
            x (Float[Tensor, "... sequence_length d_k"]): Input tensor(Query or Key) to run RoPE on.
            token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
        Returns:
            Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
        """
        *prefix_dims, seq_len, d_k = x.shape
        
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        
        # 处理 token_positions 的形状
        # token_positions 可能是 (seq_len,) 或 (batch, seq_len) 或其他形状
        # 我们需要提取 seq_len 维度的位置索引
        if token_positions.dim() > 1:
            # 如果是多维的，取第一行（通常所有 batch 的位置相同）
            # 例如 (batch, seq_len) -> 取 (seq_len,)
            token_positions = token_positions[0] if token_positions.shape[0] > 0 else token_positions.flatten()[:seq_len]
        
        # 确保 token_positions 是 1D 的，长度为 seq_len
        token_positions = token_positions.flatten()[:seq_len]
        token_positions = token_positions.to(torch.long)
        
        # 从预计算的旋转矩阵表中索引对应的旋转矩阵
        rotation_matrix = self.rotation_matrix_table[token_positions]  # (seq_len, d_k, d_k)
        
        # 扩展 rotation_matrix 以匹配 x 的前缀维度
        # x 的形状是 (*prefix_dims, seq_len, d_k)
        # rotation_matrix 需要是 (*prefix_dims, seq_len, d_k, d_k)
        for _ in range(len(prefix_dims)):
            rotation_matrix = rotation_matrix.unsqueeze(0)  # 在开头添加维度
        rotation_matrix = rotation_matrix.expand(*prefix_dims, seq_len, d_k, d_k)
        
        # 执行矩阵乘法
        x_unsqueezed = x.unsqueeze(-1)  # (*prefix_dims, seq_len, d_k, 1)
        x_rotated = rotation_matrix @ x_unsqueezed  # (*prefix_dims, seq_len, d_k, 1)
        x_rotated = x_rotated.squeeze(-1)  # (*prefix_dims, seq_len, d_k)
      
        return x_rotated

