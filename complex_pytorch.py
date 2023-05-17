import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 

from matplotlib import pyplot as plt


minimum_norm = .1
max_bias_norm = 1
max_output_norm = 2 #?


def lin_interpolator(points, upscale_factor=10, der=0):
    X = points[0]
    Y = points[1]
    new_xpoints = []
    new_ypoints = []
    for i in range(len(X)-2):
        set = [] 
        x0 = X[i]
        x1 = X[i+1] 
        y0 = Y[i]
        y1 = Y[i+1]
        new_xpoints += list(np.linspace(x0, x1, upscale_factor))
        new_ypoints += list(np.linspace(y0, y1, upscale_factor))
    
    return new_xpoints, new_ypoints


def get_max_vector_size(vectors):
    max_size = 0
    for vec in vectors:
        vec_size = vec.size(0)
        if vec_size > max_size:
            max_size = vec_size
    return max_size

def complex_power(signals, ref):
    ref = signals[0]
    ref_norm = np.linalg.norm(ref)
    real = [ref_norm]
    imag = [0]
    for i in range(1, len(signals)):
        sig_norm = np.linalg.norm(signals[i])
        product = np.dot(signals[i], ref) / ref_norm
        real.append(product) 
        b = np.sqrt(sig_norm**2 - product**2) 
        imag.append(b) 
    complex_tensor = torch.complex(real, imag)
    return complex_tensor 

def prepare_data(data): #make sure data is all of same length
    max_size = get_max_vector_size(data)
    temp2 = []
    for a in data:
        new = F.pad(a, (0, 0, 0, max_size-a.shape[0]), 'constant', 0).float()
        temp2.append(new)
    result = torch.stack(temp2)
    return result 

class MultiplicativeBias(nn.Module):    #modify to include complex numbers 
    def __init__(self, hidden):
        super().__init__()
        self.bias = nn.Parameter(max_bias_norm * torch.rand(hidden, dtype=torch.cfloat))
    
        #self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.bias, 0, 1) 

    def forward(self, x):
        return max_output_norm * torch.arctan(torch.relu(torch.relu(torch.abs(x)-minimum_norm)*x * self.bias.unsqueeze(0)))
        #return x * self.bias.unsqueeze(0)







