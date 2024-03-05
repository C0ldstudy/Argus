import torch
import torch.nn as nn
from torch import distributed as dist
from torch.nn import functional as F


class GlobalContrastiveLoss(nn.Module):
    """For MoCov3
    """
    def __init__(self, N=1.2e6, T=1.0):
        super(GlobalContrastiveLoss, self).__init__()
        self.u = torch.zeros(N).reshape(-1, 1) 
        self.T = T
        
    def forward(self, q, k, index, gamma):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        N_lagre = k.shape[0] # batch size of total GPUs
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k])
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
            
        # compute negative weights
        labels_one_hot = F.one_hot(labels, N_lagre) 
        neg_mask = 1-labels_one_hot
        neg_logits = torch.exp(logits/self.T)*neg_mask 
        u = (1 - gamma) * self.u[index].cuda() + gamma * torch.sum(neg_logits, dim=1, keepdim=True)/(N_lagre-1)
        p_neg_weights = (neg_logits/u).detach_()
            
        # gather all u & index from all machines
        u_all = concat_all_gather(u)
        index_all = concat_all_gather(index)
        self.u[index_all] = u_all.cpu()

        # compute loss 
        expsum_neg_logits = torch.sum(p_neg_weights*logits, dim=1, keepdim=True)/(N_lagre-1)
        normalized_logits = logits - expsum_neg_logits
        loss = -torch.sum(labels_one_hot * normalized_logits, dim=1)
            
        return loss.mean()* (2 * self.T)
    
    
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


# alias 
GCLoss = GlobalContrastiveLoss
