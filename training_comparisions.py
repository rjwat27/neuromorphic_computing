import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_weight_finder as pwf 

import PhaseDomainNeuron as pdn
import pdn_net as pn 
import weight_bias_transform as wbt 

from matplotlib import pyplot as plt

'''Survey of various training regiments to arrive at good set of parameters for the phase-domain
   spiking neural network

   For all variations, training will be done with parameter clipping being performed after training, 
   during training, in addition to training with 8-bit integers

   First, pytorch ANN approximation of pdn network at large sizes, then at small sizes

   Second, my numpy model in the same fashion

   Third, direct pytorch model of pdn network will be trained in the same fashion

   Fourth, numpy model of pdn network in same fashion

   Each experiment is performed 10 times and averaged in the graphical display, however performance for each experiment
   will be stored individually

   Models are trained on a three-input xor function
   '''

###Helper functions###

#function to cast 32bit value between -1 and 1 to 0-8 fixed point number, courtesy of our friend, chatGPT
def float_to_fixed_point(value):
    # check if value is between -1 and 1
    if value < -1.0 or value > 1.0:
        raise ValueError("Value must be between -1 and 1.")

    # calculate the range of values represented by each fixed point value
    range_per_fixed_point = 2 / 2**8
    
    # calculate the nearest fixed point value
    fixed_point_value = round((value + 1) / range_per_fixed_point)
    
    # clamp the fixed point value between 0 and 255
    fixed_point_value = max(0, min(255, fixed_point_value))

    #this part was ryan's addition, gpt gets no credit
    result = np.int8(fixed_point_value)
    
    return result 

xor_inputs = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]) 
xor_outputs = np.array([0, 1, 1, 0, 1, 0, 0, 1]) 

inputs = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
targets = torch.tensor([[0.0], [1.0], [1.0], [0.0], [1.0], [0.0], [0.0], [1.0]])

####################
#numpy experiments##
####################


#LARGE NETS 

###clipping during###



###############################
#pytorch experiments###########
###############################


#LARGE NETS

###clipping during###
loss, model = pwf.get_weights(inputs, targets, int(60e3), clipping='during') 
n_params = pwf.pytorch_params_to_numpy(model) 


###clipping after###
loss, model = pwf.get_weights(inputs, targets, int(60e3), clipping='after') 
n_params = pwf.pytorch_params_to_numpy(model) 


###training with 8-bit numbers###
loss, model = pwf.get_weights(inputs, targets, int(60e3), clipping=None, eight_bit=True) 
n_params = pwf.pytorch_params_to_numpy(model) 




