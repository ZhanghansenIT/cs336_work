import torch
from torch.nn.parameter import Parameter
from typing import List


def gradient_clipping(parameters: List[Parameter], max_norm: float, eps: float = 1e-6) -> None:
    """
    实现基于L2范数的梯度裁剪，原地修改参数梯度。
    
    参数说明：
        parameters: 模型中所有可训练参数的列表（如model.parameters()返回的迭代器）
        max_norm: 梯度的最大L2范数阈值（超参数，通常取1.0）
        eps: 数值稳定项，避免分母为0（作业指定1e-6，与PyTorch默认一致）
    
    核心逻辑：
        1. 计算所有参数梯度的全局L2范数；
        2. 若范数超过max_norm，按比例缩放所有梯度；
        3. 原地修改参数的.grad属性，不返回任何值。
    """
    # 步骤1：校验输入（确保max_norm合法，且参数有梯度）
    if max_norm <= 0:
        raise ValueError(f"最大梯度范数max_norm必须为正数，当前值：{max_norm}")
    
    # 步骤2：计算所有参数梯度的L2范数平方和
    grad_sq_sum = 0.0
    for p in parameters:
        if p.grad is None:
            continue  # 无梯度的参数（如冻结层）跳过
        # 计算单个参数梯度的L2范数平方（sum(g_i^2)），并累加到全局和
        grad_sq = torch.sum(p.grad.data ** 2)
        grad_sq_sum += grad_sq.item()  # 转为Python标量避免张量累积占用内存
    
    # 步骤3：计算全局梯度L2范数（加eps确保数值稳定）
    global_grad_norm = torch.sqrt(torch.tensor(grad_sq_sum, dtype=torch.float32) + eps)
    
    # 步骤4：判断是否需要裁剪（仅当全局范数>max_norm时执行）
    if global_grad_norm > max_norm:
        # 计算缩放因子：M / (全局范数)
        scale_factor = max_norm / global_grad_norm
        # 遍历所有参数，原地缩放梯度
        for p in parameters:
            if p.grad is None:
                continue
            
            # 原地操作
            p.grad.data.mul_(scale_factor)  # in-place操作：g_i' = g_i * scale_factor