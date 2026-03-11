import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, batch_size, temperature_f, device, walk_steps=3, alpha=0.5):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.temperature_f = temperature_f
        self.device = device
        self.walk_steps = walk_steps  
        self.alpha = alpha  
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")


    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N), device=self.device).fill_diagonal_(0)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        return mask.bool()

    @torch.no_grad()
    def random_walk_affinity(self, S, step=None):
        """ 使用高阶随机游走计算相似性矩阵 """
        if step is None:
            step = self.walk_steps
        S = S / S.sum(dim=1, keepdim=True)  
        S_high_order = torch.matrix_power(S, step)  
        S_final = self.alpha * torch.eye(S.shape[0], device=S.device) + (1 - self.alpha) * S_high_order
        return S_final

    def Structure_guided_Contrastive_Loss(self, h_i, h_j, S):
        """ 结合高阶随机游走计算的结构引导对比损失 """
        S_high_order = self.random_walk_affinity(S)  
        S_1 = S_high_order.repeat(2, 2)  
        all_one = torch.ones(self.batch_size * 2, self.batch_size * 2, device=self.device)
        S_2 = all_one - S_1  

        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)  
        sim = torch.matmul(h, h.T) / self.temperature_f  

        sim1 = sim * S_2  
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        mask = self.mask_correlated_samples(N)
        negative_samples = sim1[mask].reshape(N, -1)  

        labels = torch.zeros(N, device=positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels) / N  

        return loss
