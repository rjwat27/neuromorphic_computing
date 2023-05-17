import numpy as np
import PhaseDomainNeuron as pdn

import pdn_net as pn 

import weight_bias_transform as wbt 

from matplotlib import pyplot as plt

import torch
import torch.nn as nn


pdn.energy_per_spike = 1#max_bias * 1.1 

import pytorch_weight_finder as pwf 
import json 


import sys

if len(sys.argv) != 4:
    print("Usage: python toolchain.py samples targets bitstream_destination\n\nInput and Target files must be torch .pt files storing a single tensor")
    sys.exit(1)

input_file = sys.argv[1]
target_file = sys.argv[2]
destination_file = sys.argv[3]


# Load the model from the .pt file
inputs = torch.load(input_file)

# Load the model from the .pt file
targets = torch.load(target_file)


'''generating model from network similar to pdn'''

loss, model = pwf.learn(inputs, targets, num_epochs=int(10e3), clipping='during', graph=False, quantized=False) 

n_params = pwf.pytorch_params_to_numpy(model) 

'''Saving learned parameters'''

def numpy_to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy array to Python list
    raise TypeError("Object of type '%s' is not JSON serializable" % type(obj).__name__)

# Convert the list of NumPy arrays to a JSON-compatible format
json_data = json.dumps(n_params, default=numpy_to_json)

with open('model_parameters.json', 'w') as file:
    file.write(json_data)

#recover parameters
def json_to_numpy(obj):
    if '__ndarray__' in obj:
        data = obj['__ndarray__']
        return np.array(data, dtype=data['dtype'])
    return obj


with open('model_parameters.json', 'r') as file:
    json_data = file.read()


n_params = json.loads(json_data, object_hook=json_to_numpy)

weights = [np.array(n_params[0]).T, np.array(n_params[1]).T, np.array(n_params[2]).T]

biases = [np.array(n_params[3]), np.array(n_params[4]), np.array(n_params[5]), np.array(n_params[6])]
 

#using pytorch model to calibrate a pdn model

net = pn.pdn_network(3, 3, [3, 3, 3, 3]) 

net.weights = weights 
 
net.calibrate_network_biases([1, 1, 1], biases)

net.Save() 

#generate weight bitstream 

model.generate_bitstream(destination_file)

















