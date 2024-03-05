import torch 

def squared_loss(margin, t):
    return (margin - t)** 2

def squared_hinge_loss(margin, t):
    return torch.max(margin - t, torch.zeros_like(t)) ** 2

def logistic_loss(margin, t):
    return torch.log(1+torch.log(-margin*t))
