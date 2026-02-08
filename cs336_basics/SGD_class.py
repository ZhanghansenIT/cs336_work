
import torch
from typing import Optional, Callable
import math 
class SGD(torch.optim.Optimizer):
    
    def __init__(self, params, lr=0.03):
        
        if lr < 0.0 : 
            raise ValueError("lr must be positive")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        
    def step(self, closure :Optional[Callable] = None ):
        loss = None if closure is None else closure()
        
        # 遍历所有参数组
        for group in self.param_groups:
            # 遍历每个参数组中的所有参数
            lr = group['lr'] 
            for p in group['params']:
                if p.grad is not None:
                    continue
                
                state = self.state[p]
                t = state.get('t', 0)
                grad = p.grad.data
                p.data -= lr /math.sqrt(t+1) * grad
                state['t'] = t + 1
        return loss
    
if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=0.03)
    
    for t in range(100):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimization step