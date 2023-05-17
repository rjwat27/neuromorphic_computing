import numpy as np
import PhaseDomainNeuron as pdn

import pdn_net as pn 

import weight_bias_transform as wbt 

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
#from torch.utils.data import Dataset, DataLoader

xor_inputs = np.array([[0, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1]]) 
xor_outputs = np.array([0, 0, 0, 0, 1, 1, 1, 1]) 

xor_inputs2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_outputs2 = np.array([0, 1, 1, 0])

inputs = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
targets = torch.tensor([[0.0], [1.0], [1.0], [0.0], [1.0], [0.0], [0.0], [1.0]])

inputs2 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
targets2 = torch.tensor([[0.0], [1.0], [1.0], [0.0]]) 


pdn.energy_per_spike = 1#max_bias * 1.1 

import pytorch_weight_finder as pwf 
import json 


'''importing parameters from pytorch model'''
# TEST = pwf.Chip() 
# TEST.generate_bitstream()


'''generating model from network similar to pdn'''

loss, model = pwf.learn(inputs2, targets2, num_epochs=int(10e3), clipping='during', graph=False, quantized=False) 
#input()
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


weights = [np.array(n_params[0]).T, np.array(n_params[1]).T, np.array(n_params[2]).T]#, n_params[3]]

biases = [np.array(n_params[3]), np.array(n_params[4]), np.array(n_params[5]), np.array(n_params[6])]#, n_params[7]] 
 
net = pn.pdn_network(3, 3, [3, 3, 3, 3]) 

net.weights = weights 
 
print('beginning calibration')
net.calibrate_network_biases([1, 1, 1], biases)

'''pytorch loading done'''


'''loading parameters from numpy model'''

def load():
    weights = np.load('xor_weights.npy', allow_pickle=True)

    fanout_codes = np.load('xor_fanout_codes.npy', allow_pickle=True) 

    biases = np.load('xor_biases.npy', allow_pickle=True)

    return weights, fanout_codes, biases 

net.Save() 
#input('calibration done and saved') 
#net.Load()
#print('loaded successfully')
# input(net.weights)
# B = [n.vref for n in net.input_layer] 
# input(B)

# input(np.average(net.activate([1, 1, 1]))) 
# input(np.average(net.activate_burst([1, 1, 1], 1000))) 
# input(np.average(net.activate_burst([1, 1, 1], 1000))) 
# input(np.average(net.activate_burst([1, 1, 1], 1000))) 
# input(np.average(net.activate_burst([1, 1, 1], 1000))) 
# input(np.average(net.activate_burst([1, 1, 1], 1000))) 
#net.weights = net.weights.reverse()
n = net.output_layer[0]

# net.input_layer[0].update_vref(1)
# net.input_layer[1].update_vref(1)
# net.input_layer[2].update_vref(1) 

#input(biases) 
#n.update_vref(100) 


#input('Done')

#net.output_layer[0].update_vref(40) 

n1 = net.input_layer[0]
n2 = net.input_layer[1] 
n3 = net.input_layer[2] 

n1.update_vref(.05)
n2.update_vref(.05) 
n3.update_vref(.05)

# vals = net.input_layer[0].forward_burst(1, 1)
# input(net.input_layer[0].vref)

O = []
graphs = []
for i in inputs2:
    graph = net.activate_burst(i, 1000)
    #graph = net.hidden_layer_burst(i, 10000, 1) 
    graphs.append(graph) 
    O.append(np.average(graph, axis=0))
    #O.append(np.average(graph, axis=1))
    #O.append(graph)
#o1 = net.activate_burst([1, 0, 0], 10000)
#f = np.fft.fft(o, axis=0) 
x = range(1000)
#print(np.fft.fftfreq(1000))
#f1 = np.fft.fft(o1, axis=0) 

#
#input(np.shape(o1))
# plt.plot(x, f, color='red')
# plt.plot(x, f1, 'green')
# plt.show()

#o = net.hidden_layer_burst([1, 1, 0], 1000, 0) 
# u = net.output_layer[0].INPUT_STREAM 
# t = net.hidden_layers[0][0].output_stream
#avg = np.average(o)#, axis=0) 
#print(avg)
for i in range(len(inputs2)):
    print(inputs2[i], ': ', O[i], '; ', targets2[i]) 


fig, axes = plt.subplots(len(inputs2), 1) 

for i in range(len(inputs2)):
    axes[i].plot(x, graphs[i]) 
    #axes[i].title(str(xor_inputs2[i])) 

plt.show() 

print('Done.') 



