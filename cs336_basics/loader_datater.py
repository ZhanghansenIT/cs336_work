

import numpy as np
import torch
from typing import Tuple
import numpy.typing as npt
def run_get_batch(
    x: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从token化的numpy数组中采样批次输入-目标对，用于Transformer LM训练。
    
    参数:
        x: np.ndarray - 1D整数数组，存储所有token的ID（ dtype建议为uint16，符合作业2.7(d)建议）
        batch_size: int - 每个批次的样本数量
        context_length: int - 每个样本的上下文长度（序列长度）
        device: str - PyTorch设备标识，如"cpu"、"cuda:0"、"mps"
    
    返回:
        Tuple[torch.Tensor, torch.Tensor] - (inputs, targets)
            inputs: 形状为(batch_size, context_length)的LongTensor，输入token序列
            targets: 形状为(batch_size, context_length)的LongTensor，目标token序列（输入的下一个token）
    
    异常:
        ValueError - 若数据长度不足（无法构造至少一个有效序列）或批次大小非法
    """
    # 1. 检查输入合法性
    if batch_size <= 0:
        raise ValueError(f"批次大小batch_size必须为正整数，当前为{batch_size}")
    if context_length <= 0:
        raise ValueError(f"上下文长度context_length必须为正整数，当前为{context_length}")
    
    total_tokens = len(x)
    # 可采样的最大起始索引：i + context_length + 1 ≤ total_tokens → i ≤ total_tokens - context_length - 1
    max_start_idx = total_tokens - context_length - 1
    if max_start_idx < 0:
        raise ValueError(
            f"数据长度不足：总token数{total_tokens}，需至少满足 total_tokens ≥ context_length + 1（当前context_length={context_length}）"
        )
    if batch_size > max_start_idx + 1:
        raise ValueError(
            f"批次大小{batch_size}超过可采样序列数{max_start_idx + 1}，请减小batch_size或context_length"
        )
    
    # 2. 随机采样batch_size个不重复的起始索引（确保随机性）
    # np.random.choice范围为[0, max_start_idx]，size=batch_size，replace=False避免重复
    start_indices = np.random.choice(max_start_idx + 1, size=batch_size, replace=False)
    
    # 3. 构造输入和目标序列（批量处理，避免循环提升效率）
    # 初始化空数组存储批次数据（ dtype与输入x一致，通常为uint16）
    inputs_np = np.empty((batch_size, context_length), dtype=x.dtype)
    targets_np = np.empty((batch_size, context_length), dtype=x.dtype)
    
    for idx, start_i in enumerate(start_indices):
        # 输入：[start_i, start_i + context_length)
        inputs_np[idx] = x[start_i : start_i + context_length]
        # 目标：[start_i + 1, start_i + context_length + 1)（与输入错位1个token）
        targets_np[idx] = x[start_i + 1 : start_i + context_length + 1]
    
    # 4. 转换为PyTorch LongTensor（token ID需为整数类型）并转移到指定设备
    inputs = torch.tensor(inputs_np, dtype=torch.long, device=device)
    targets = torch.tensor(targets_np, dtype=torch.long, device=device)
    
    return inputs, targets