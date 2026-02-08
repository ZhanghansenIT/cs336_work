from typing import Optional
import torch 
class TrainConfig():
    
    # 数据配置 
    train_data_path: str = ".data/train.txt"
    val_data_path: str = ".data/val.txt"
    
    vocab_size: int = 10000
    context_length: int = 256
    
    # 模型配置
    
    
    d_model: int = 512 # 模型维度
    num_layers: int = 4 # transformer 层数
    num_heads: int = 16 # 多头注意力头数
    d_ff: int = 1344 # 前馈神经网络维度
    rope_theta: float = 10000.0 # rope 缩放因子
    
    # 训练配置
    batch_size: int = 32 
    max_steps: int = 5000 
    val_interval: int = 100 # 每多少步验证一次
    ckpt_dir: str = "./checkpoints" # 检查点保存路径
    ckpt_interval: int = 100 # 每多少步保存一次

    load_ckpt_path: Optional[str] = None # 加载检查点路径
    # 优化器配置
    
    # 优化器配置
    lr_max: float = 3e-4                                       # 最大学习率
    lr_min: float = 3e-5                                       # 最小学习率
    lr_warmup_steps: int = 100                                 # 预热步数
    lr_anneal_steps: int = 4900                                # 余弦退火步数（max_steps - warmup_steps）
    weight_decay: float = 1e-4                                 # AdamW权重衰减
    beta1: float = 0.9,
    beta2: float = 0.95,                                       # LLaMA同款β参数
    eps: float = 1e-8                                          # 数值稳定项
    grad_clip_max_norm: float = 1.0                            # 梯度裁剪阈值
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
2
    
    