from typing import Tuple, Dict, List, Optional, Callable
from collections import defaultdict
import numpy as np
import torch
from torch import Tensor


class AdamW:
    """
    AdamW优化器实现（与PyTorch原生AdamW兼容）
    
    AdamW与Adam的主要区别在于权重衰减的处理方式：
    - Adam：权重衰减直接加在梯度上（L2正则化）
    - AdamW：权重衰减直接作用于参数（真正的权重衰减）
    
    Reference: "Decoupled Weight Decay Regularization" (ICLR 2019)
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4
    ) -> None:
        """
        初始化AdamW优化器
        
        Args:
            params: 待优化的参数（可迭代对象）
            lr: 学习率 (default: 1e-3)
            betas: 用于计算梯度一阶矩和二阶矩的系数 (default: (0.9, 0.999))
            eps: 数值稳定性常数 (default: 1e-8)
            weight_decay: 权重衰减系数 (default: 1e-4)
        """
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # 状态字典，用于存储每个参数的优化器状态
        self.state: Dict = defaultdict(dict)
        
        # 过滤出需要梯度的参数
        if isinstance(params, torch.Tensor):
            params = [params]
        self.params: List[torch.Tensor] = [
            p for p in params if hasattr(p, 'requires_grad') and p.requires_grad
        ]
        
        # 初始化一阶矩（动量）和二阶矩（方差）
        self.m: Dict[torch.Tensor, torch.Tensor] = {}
        self.v: Dict[torch.Tensor, torch.Tensor] = {}
        
        # 时间步（用于偏差修正）
        self.t: int = 0
        
        # 初始化矩估计
        for p in self.params:
            if not torch.is_tensor(p):
                raise TypeError(f"参数必须是torch.Tensor类型，实际类型: {type(p)}")
            
            self.m[p] = torch.zeros_like(
                p, 
                dtype=p.dtype, 
                device=p.device,
                memory_format=torch.preserve_format
            )
            self.v[p] = torch.zeros_like(
                p, 
                dtype=p.dtype, 
                device=p.device,
                memory_format=torch.preserve_format
            )
    
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        执行一次优化步骤
        
        Args:
            closure: 用于重新计算损失的闭包函数 (optional)
            
        Returns:
            如果提供了closure，返回损失值；否则返回None
            
        Raises:
            RuntimeError: 如果参数梯度为None且未提供closure
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self.t += 1
        
        # 计算偏差修正系数
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t
        
        # 计算每一步的有效学习率
        # 注意：step_size 只需要除以 bias_correction1，因为 v_hat 已经包含了 bias_correction2 的修正
        step_size = self.lr / bias_correction1
        
        for p in self.params:
            if p.grad is None:
                if closure is None:
                    raise RuntimeError(
                        f"参数在未提供closure的情况下梯度为None。"
                        f"确保在调用step()之前调用了backward()。"
                    )
                continue
            
            grad = p.grad.data
            
            # 检查梯度是否为有限值
            if not torch.isfinite(grad).all():
                raise RuntimeError("梯度包含非有限值（NaN或Inf）")
            
            # 获取当前参数的矩估计
            m = self.m[p]
            v = self.v[p]
            
            # 更新一阶矩（动量）
            # m_t = β1 * m_{t-1} + (1 - β1) * g_t
            m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            
            # 更新二阶矩（方差）
            # v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
            v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
            
            # 偏差修正
            m_hat = m / bias_correction1
            v_hat = v / bias_correction2
            
            # AdamW更新步骤
            # 1. 应用权重衰减（与Adam的主要区别）
            if self.weight_decay != 0:
                p.data.mul_(1 - self.lr * self.weight_decay)
            
            # 2. 使用自适应学习率更新参数
            # θ_t = θ_{t-1} - η * m_hat / (√v_hat + ε)
            denom = v_hat.sqrt().add_(self.eps)
            p.data.addcdiv_(m_hat, denom, value=-step_size)
        
        return loss
    
    def zero_grad(self, set_to_none: bool = False) -> None:
        """
        清空所有参数的梯度
        
        Args:
            set_to_none: 如果为True，将梯度设置为None而不是零张量，
                        这可以减少内存占用
        """
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    p.grad.zero_()
    
    def state_dict(self) -> Dict:
        """
        获取优化器状态字典
        
        Returns:
            包含优化器所有状态的字典
        """
        return {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "t": self.t,
            "m": self.m,
            "v": self.v,
            "params": [id(p) for p in self.params],  # 只保存参数ID用于引用
        }
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """
        加载优化器状态
        
        Args:
            state_dict: 状态字典
            
        Raises:
            ValueError: 如果状态字典格式不正确
            KeyError: 如果缺少必要的键
        """
        required_keys = {"lr", "beta1", "beta2", "eps", "weight_decay", "t", "m", "v"}
        if not required_keys.issubset(state_dict.keys()):
            missing = required_keys - state_dict.keys()
            raise KeyError(f"状态字典缺少必要的键: {missing}")
        
        self.lr = state_dict["lr"]
        self.beta1 = state_dict["beta1"]
        self.beta2 = state_dict["beta2"]
        self.eps = state_dict["eps"]
        self.weight_decay = state_dict["weight_decay"]
        self.t = state_dict["t"]
        self.m = state_dict["m"]
        self.v = state_dict["v"]
    
    def add_param_group(self, param_group: Dict) -> None:
        """
        添加参数组到优化器（兼容PyTorch接口）
        
        Args:
            param_group: 参数组字典，必须包含'params'键
        """
        if 'params' not in param_group:
            raise ValueError("参数组必须包含'params'键")
        
        new_params = param_group['params']
        if isinstance(new_params, torch.Tensor):
            new_params = [new_params]
        
        for p in new_params:
            if not torch.is_tensor(p):
                raise TypeError(f"参数必须是torch.Tensor类型，实际类型: {type(p)}")
            
            if p not in self.params:
                self.params.append(p)
                
                # 为新参数初始化矩估计
                self.m[p] = torch.zeros_like(
                    p, 
                    dtype=p.dtype, 
                    device=p.device,
                    memory_format=torch.preserve_format
                )
                self.v[p] = torch.zeros_like(
                    p, 
                    dtype=p.dtype, 
                    device=p.device,
                    memory_format=torch.preserve_format
                )
    
    def __repr__(self) -> str:
        """返回优化器的字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"lr={self.lr}, "
            f"betas=({self.beta1}, {self.beta2}), "
            f"eps={self.eps}, "
            f"weight_decay={self.weight_decay})"
        )


# 使用示例
if __name__ == "__main__":
    # 创建测试模型
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1)
    )
    
    # 创建优化器
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 模拟训练步骤
    for epoch in range(10):
        # 模拟前向传播
        inputs = torch.randn(32, 10)
        targets = torch.randn(32, 1)
        outputs = model(inputs)
        
        # 计算损失
        loss = torch.nn.functional.mse_loss(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")