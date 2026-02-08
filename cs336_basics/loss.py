
import torch

def softmax(x: torch.Tensor) :
    return torch.exp(x) / torch.sum(torch.exp(x), dim=-1, keepdim=True)

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) :
    """
    计算交叉熵损失，支持批量维度（batch-like dimensions），确保数值稳定性。
    
    参数说明：
        logits: 模型输出的未归一化预测值，形状为 (..., seq_len, vocab_size)
                其中 "..." 代表任意数量的批量维度（如 batch_size）
        targets: 真实标签（token ID），形状为 (..., seq_len)，与 logits 的前 N-1 维完全匹配
    
    返回值：
        平均交叉熵损失（标量 Tensor），对所有批量维度和序列长度取平均
    """
    # 假设输入是 logits = [8,8,10000] 表示8个样本，每个样本8个token，每个token有10000个类别

    # 确保 logits 比 targets 多一个维度 
    if logits.dim() != targets.dim() + 1: 
        raise ValueError("logits 和 targets 的维度不匹配")
    
    # 确保 logits 的前 N-1 维与 targets 的形状一致
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"logits 前 {logits.dim()-1} 维（{logits.shape[:-1]}）必须与 targets 形状（{targets.shape}）一致"
        )
        
    
    # 计算交叉熵损失
    # 1. 计算logits的最大值 目的是为了防止指数爆炸 [8,8,10000] -> [8,8,1    ]
    logits_max = torch.max(logits, dim=-1, keepdim=True).values
    # 2. 计算logits - logits_max [8,8,10000] -> [8,8,10000]
    log_exp = logits - logits_max
    # 3. 计算exp(log_exp) [8,8,10000] -> [8,8,10000]
    exp_logits = torch.exp(log_exp)
    # 4. 计算sum(exp(log_exp)) [8,8,10000] -> [8,8,1]
    exp_sum = torch.sum(exp_logits, dim=-1, keepdim=True)
    # 5. 计算log_probs 公式为 log(softmax(logits)) = logits - log(sum(exp(logits))) [8,8,10000] -> [8,8,10000]
    
    # 计算log_probs 公式为 log(softmax(logits)) = logits - log(sum(exp(logits)))
    # [8,8,10000] -> [8,8,10000]
    log_probs = log_exp - torch.log(exp_sum)
    
    # -------------------------- 4. 提取目标位置的 log(softmax) 值 --------------------------
    # 方法：使用 torch.gather 从 log_softmax 的最后一维（vocab_size）中提取 targets 对应索引的值
    # 1. 将 targets 扩展为 (..., seq_len, 1)，与 log_softmax 的维度匹配
    #[8,8] -> [8,8,1]
    targets_unsqueezed = targets.unsqueeze(dim=-1)
    # print(targets_unsqueezed.shape)
    # 2. 按最后一维 gather，得到每个目标 token 对应的 log(softmax) 值
    #    gather 维度为 -1（vocab_size 维），索引为 targets_unsqueezed
    # [8,8,10000] -> [8,8,1]
    
    # 从每个 batch、每个序列位置的 log 概率分布中，精准提取 “真实目标 token（下一个要预测的 token）对应的 log 概率值”。
    log_probs = torch.gather(log_probs, dim=-1, index=targets_unsqueezed)
    # 3. 移除最后一维（从 (..., seq_len, 1) 变为 (..., seq_len)）
    # [8,8,1] -> [8,8]
    log_probs = log_probs.squeeze(dim=-1)

    # -------------------------- 5. 计算平均交叉熵损失 --------------------------
    # 交叉熵损失 = -log(softmax(logits)[targets])，对所有元素取平均
    cross_entropy = -log_probs
    # 对所有批量维度和序列长度取平均（总元素数 = 所有维度大小的乘积）
    avg_loss = cross_entropy.mean()

    return avg_loss
    
    
    
if __name__ == "__main__":
    
    # 比如这个tensor表示两个样本，每个样本有4个类别，每个类别的概率为0.5,0.5,0.7,0.8
    x = torch.tensor([[0.5,0.5,0.7,0.8],[4,1,3,9]])
    sf = softmax(x)
    
    
    print(sf)
    
    print(sf[0].sum())
    print(sf[1].sum())
    
    """
    logits = [batch_size, sequence_length, vocab_size]
    对应你的例子：[512, 128, 10000] = [批次大小, 序列长度, 词汇表大小]
    逐维度拆解（结合实际场景）
    
    1. 第一个维度：512 → batch_size（批次大小）
    含义：这一次模型前向传播，一次性处理了 512 个独立的文本序列（可以理解为 “512 句话”）；
    类比：就像老师一次批改 512 份作业，而不是逐份改，目的是利用 GPU 并行计算加速训练；
    举例：这 512 个序列可能是：
    第 1 个序列："I love cat"（ID: [5,8,12,...]）
    第 2 个序列："He plays football"（ID: [9,15,20,...]）
    ...
    第 512 个序列："She eats apple"（ID: [11,18,25,...]）
    2. 第二个维度：128 → sequence_length（序列长度）
    含义：每一个文本序列的长度都是 128 个 token（即 “每句话被截断 / 补齐成 128 个词”），且模型会为序列中每个位置输出一组预测分数；
    关键关联：这个维度对应你之前问的 
    i
    （预测位置）—— 每个序列的 128 个位置，对应模型要预测 128 次 “下一个 token”（即 
    i=1
    到 
    i=128
    ）；
    举例：对第 1 个序列（"I love cat..."）来说：
    第 1 个位置（i=1）：输入语境是第 1 个 token（I），预测第 2 个 token（love）；
    第 2 个位置（i=2）：输入语境是前 2 个 token（I love），预测第 3 个 token（cat）；
    ...
    第 128 个位置（i=128）：输入语境是前 128 个 token，预测第 129 个 token；
    3. 第三个维度：10000 → vocab_size（词汇表大小）
    含义：模型为 “每个位置” 输出 10000 个分数（对应词汇表中 10000 个 token 的 “偏好分数”），这就是 logits 的核心 ——未归一化的预测分数；
    关键关联：这个维度对应你之前问的 
    x 
    i+1
    ​
    
    （真实目标 token）—— 用 
    x 
    i+1
    ​
    
    的 ID 作为索引，就能从这 10000 个分数中找到 “模型给标准答案的分数”；
    举例：对第 1 个序列的第 1 个位置（i=1），模型输出的 10000 个分数中：
    第 5 位（I）：分数 0.6；
    第 8 位（love）：分数 8.5（最高，说明模型认为 love 是最可能的下一个 token）；
    第 12 位（cat）：分数 1.3；
    ...（其余 9997 个 token 的分数都很低）。
        
    
    """