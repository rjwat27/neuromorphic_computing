import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools

inputs2 = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
targets2 = torch.tensor([[0.0], [1.0], [1.0], [0.0]]) 



# Leaky neuron model, overriding the backward pass with a custom function
class LeakySurrogate(nn.Module):
  def __init__(self, beta, threshold=[1.0, 1.0, 1.0]):
      super(LeakySurrogate, self).__init__()

      # initialize decay rate beta and threshold
      self.beta = beta
      self.threshold = nn.Parameter(torch.Tensor(threshold)) 
      self.spike_gradient = self.ATan.apply
  
  # the forward function is called each time we call Leaky
  def forward(self, input_, mem):
    spk = self.spike_gradient((mem-self.threshold))  # call the Heaviside function
    reset = (self.beta * spk * self.threshold).detach() # remove reset from computational graph
    mem = self.beta * mem + input_ - reset # Eq (1)
    return spk, mem

  # Forward pass: Heaviside function
  # Backward pass: Override Dirac Delta with the ArcTan function
  @staticmethod
  class ATan(torch.autograd.Function):
      @staticmethod
      def forward(ctx, mem):
          spk = (mem > 0).float() # Heaviside on the forward pass: Eq(2)
          ctx.save_for_backward(mem)  # store the membrane for use in the backward pass
          return spk

      @staticmethod
      def backward(ctx, grad_output):
          (mem,) = ctx.saved_tensors  # retrieve the membrane potential 
          grad = 1 / (1 + (np.pi * mem).pow_(2)) * grad_output # Eqn 5
          return grad
      
#print(test2._parameters['weight'].detach().numpy()) 


# Define Network
class Net_Chip(nn.Module):
    def __init__(self, num_inputs=3, num_outputs=1, loss=0):
        super().__init__()

        # Input layer
        self.input_layer = LeakySurrogate(loss, threshold = torch.rand(3))  

        # Hidden layers layers
        self.fc1 = nn.Linear(num_inputs, 3, False)
        self.lif1 = LeakySurrogate(loss, threshold = torch.rand(3)) 

        self.fc2 = nn.Linear(3, 3, False)
        self.lif2 = LeakySurrogate(loss, threshold = torch.rand(3))  

        # Output layers
        self.fc3 = nn.Linear(3, num_outputs, False)
        self.lif3 = LeakySurrogate(loss, threshold = torch.rand(3))

    def forward(self, x, num_steps=1):

        # Initialize hidden states at t=0
        # mem1 = self.lif1.init_leaky()
        # mem2 = self.lif2.init_leaky()
        # mem3 = self.lif3.init_leaky() #???
        mem1 = 0
        mem2 = 0 
        mem3 = 0

        # Record the final layer
        spk3_rec = []
        mem3_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc2(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
            mem3_rec.append(mem2)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)
    
test = Net_Chip() 

test_input = torch.Tensor([1.0, 1.0, 1.0]) 

print(test.forward(test_input, num_steps=10)) 

print('done') 

