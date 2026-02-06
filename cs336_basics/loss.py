
import torch 


# 交叉熵损失 

def cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算交叉熵损失，支持批量维度（batch-like dimensions），确保数值稳定性。
    
    参数说明：
        logits: 模型输出的未归一化预测值，形状为 (..., seq_len, vocab_size)
                其中 "..." 代表任意数量的批量维度（如 batch_size）
        targets: 真实标签（token ID），形状为 (..., seq_len)，与 logits 的前 N-1 维完全匹配
    
    返回值：
        平均交叉熵损失（标量 Tensor），对所有批量维度和序列长度取平均
    """
    if inputs.dim() != targets.dim() + 1:
        raise ValueError("输入的 logits 和 targets 维度不匹配。")
    
    if inputs.shape[:-1] != targets.shape:
        raise ValueError("输入的 logits 和 targets 形状不匹配。")
    
    # 计算 log-softmax
    logits_max = inputs.max(dim=-1, keepdim=True)[0]
    logits_stable = inputs - logits_max


    exp_logits = torch.exp(logits_stable)
    exp_sum = exp_logits.sum(dim=-1, keepdim=True)
    log_prons 


    return loss
