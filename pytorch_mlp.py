import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiplicativeBias(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(hidden))
    
        
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.bias, 0, 1) 

    def forward(self, x):
        return torch.clamp(x * self.bias.unsqueeze(0), min=0)

    # def backward(self, grad_output):
    #     grad_input = grad_output * self.bias.unsqueeze(0)
    #     grad_bias = (grad_output * self.input).sum(dim=0)
    #     return grad_input, grad_bias

class CustomModel(nn.Module):
    def __init__(self, ninputs, noutputs, hidden):
        super(CustomModel, self).__init__()
        self.linear1 = nn.Linear(ninputs, hidden, bias=False)
        self.multiplicative_bias1 = MultiplicativeBias(hidden)
        self.linear2 = nn.Linear(hidden, noutputs, bias=False)
        self.multiplicative_bias2 = MultiplicativeBias(hidden)
        self.linear3 = nn.Linear(hidden, noutputs, bias=False)
        self.multiplicative_bias3 = MultiplicativeBias(noutputs)

        # self.linear4 = nn.Linear(hidden, noutputs)


    def forward(self, x):
        x = self.linear1(x.float())
        x = self.multiplicative_bias1(x.float())
        x = self.linear2(x.float())
        x = self.multiplicative_bias2(x.float())
        x = self.linear3(x.float())
        x = self.multiplicative_bias3(x.float())
        #x = self.linear4(x.float())
        return x

    # def backward(self, grad_output):
    #     grad_output = grad_output.unsqueeze(-1)
    #     grad_linear2 = grad_output * self.multiplicative_bias.bias.unsqueeze(-1)
    #     grad_bias = grad_output * self.linear2.weight.unsqueeze(0) * self.linear1.weight.unsqueeze(-1) * x.unsqueeze(-1)
    #     grad_bias = grad_bias.sum(dim=0)
    #     grad_multiplicative_bias, grad_linear1 = self.multiplicative_bias.backward(grad_linear2)
    #     grad_input = self.linear1







