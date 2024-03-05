# credit to LibAUC
import torch 
from torch import nn
import torch.nn.functional as F

# Multilayer Perceptron
class MLP(torch.nn.Module):
    def __init__(self, input_dim=29, hidden_sizes=(16,), activation='relu', num_classes=1):
        super().__init__()
        self.inputs = torch.nn.Linear(input_dim, hidden_sizes[0]) 
        layers = []
        for i in range(len(hidden_sizes)-1):
            layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1])) 
            if activation=='relu':
               print ('relu')
               layers.append(nn.ReLU())
            elif activation=='elu':
               layers.append(nn.ELU())
            else:
               pass 
        self.layers = nn.Sequential(*layers)
        self.classifer = torch.nn.Linear(hidden_sizes[-1], num_classes)

    def forward(self, x):
        """forward pass"""
        x = self.inputs(x)
        x = self.layers(x)
        return self.classifer(x) 
    
