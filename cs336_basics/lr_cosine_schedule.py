import torch
import math
from typing import Union


def get_cosine_lr_schedule(
    t: Union[int, torch.Tensor],
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int
) -> Union[float, torch.Tensor]:
    """
    带预热的余弦退火学习率调度函数，遵循LLaMA训练策略。
    基于更高精度（float64），导致微小差异，需要使用float64进行计算
    参数说明：
        t: 当前训练步数（int或PyTorch张量，支持批量计算）
        alpha_max: 最大学习率（预热结束后的峰值）
        alpha_min: 最小学习率（退火结束后维持的值）
        T_w: 预热步数（t < T_w时线性升温）
        T_c: 余弦退火总步数（T_w ≤ t ≤ T_c时余弦降温）
    
    返回值：
        当前步数对应的学习率（float或与t同形状的张量）
    """
    # 确保输入t为张量（方便批量计算和梯度安全）
    # 使用 float64 以提高数值精度，匹配测试期望值
    if isinstance(t, (int, float)):
        t = torch.tensor(float(t), dtype=torch.float64)
    elif isinstance(t, torch.Tensor):
        t = t.to(torch.float64)  # 转为float64，提高精度
    else:
        # 处理numpy类型或其他可转换类型
        t = torch.tensor(float(t), dtype=torch.float64)

    # 阶段1：预热（t < T_w）- 线性升温
    warmup_condition = t < T_w
    # 阶段2：余弦退火（T_w ≤ t ≤ T_c）- 平滑降温
    anneal_condition = (t >= T_w) & (t <= T_c)
    # 阶段3：后退火（t > T_c）- 维持最小学习率
    post_anneal_condition = t > T_c

    # 计算各阶段学习率（用where实现条件分支，避免for循环，支持批量计算）
    # 预热阶段：α = (t / T_w) * α_max
    lr_warmup = (t / T_w) * alpha_max
    # 余弦退火阶段：按公式计算
    # 先归一化步数到[0, π]（确保cos输入范围正确）
    normalized_step = (t - T_w) / (T_c - T_w) * math.pi
    lr_anneal = alpha_min + 0.5 * (1 + torch.cos(normalized_step)) * (alpha_max - alpha_min)
    # 后退火阶段：固定为α_min
    lr_post_anneal = torch.tensor(alpha_min, dtype=torch.float64)

    # 合并三个阶段的结果（优先级：后退火 > 退火 > 预热）
    lr = torch.where(post_anneal_condition, lr_post_anneal,
                     torch.where(anneal_condition, lr_anneal, lr_warmup))

    # 若输入是标量（单步计算），返回float；否则返回张量
    return lr.item() if lr.numel() == 1 else lr



# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import numpy as np

#     # 生成步数（0到100000步，间隔100步）
#     steps = np.arange(0, 100001, 100)
#     # 计算对应学习率
#     lrs = [get_cosine_lr_schedule(t, alpha_max=3e-4, alpha_min=3e-5, T_w=1000, T_c=99000) for t in steps]

#     # 绘图
#     plt.figure(figsize=(12, 6))
#     plt.plot(steps, lrs, label='Cosine LR Schedule (with Warm-up)')
#     plt.axvline(x=1000, color='red', linestyle='--', label=f'Warm-up End (T_w=1000)')
#     plt.axvline(x=99000, color='green', linestyle='--', label=f'Annealing End (T_c=99000)')
#     plt.xlabel('Training Step (t)')
#     plt.ylabel('Learning Rate (α)')
#     plt.title('LLaMA-style Cosine Learning Rate Schedule')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.show()