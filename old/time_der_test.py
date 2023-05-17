import numpy as np

import fanout_layer as fl 

from matplotlib import pyplot as plt

xor_inputs = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]] 
xor_outputs = [0, 1, 1, 0, 1, 0, 0, 1]  

min_bias = .18
max_bias = 1

def spiking_activation(x, bias):
    return np.maximum(0, np.multiply(x, bias)) 
def spiking_der(x, bias):
    return bias if x > 0 else .1

net = fl.fanout_network(3, 1, 2, [3, 12, 1], fanout=0, learning_rate=.1, ordered=False, growth=True) 
net.set_activation(spiking_activation) 
net.set_der(spiking_der) 
net.set_bias_bounds(min_bias, max_bias) 
net.set_weight_bounds(-1, 1)
net.randomize_biases()
#net.set_layers_to_grow([0]) 


def learn(net):
    error = 10 

    epoch = 0

    while error > .5 and epoch < 10:
        epoch += 1
        # print(net.layer_sizes)
        # input() 
        error, iter = fl.run_learn_cycle(net, xor_inputs, xor_outputs, .5, len(xor_inputs)) 
        print('Epoch: ', epoch)
        print('Iter: ', iter) 
        print('error: ', error) 
        print('size of network: ', len(net.hidden_layers[0].biases1))
        print('biases: ', net.hidden_layers[0].biases1)

    #print results 
    for i in range(len(xor_inputs)):
        result = net.activate(xor_inputs[i]) 
        answer = xor_outputs[i]
        e = answer - result 
        print(xor_inputs[i], ':, ', result, answer, e)
    print('size of network: ', len(net.hidden_layers[0].biases1))
    print('biases: ', net.hidden_layers[0].biases1) 
    print(net.hidden_layers[1].biases1) 

def save(net):
    weights = np.array([net.hidden_layers[n].w1 for n in range(net.layers)])
    fanout_codes = np.array([net.hidden_layers[n].fanout_encoding1 for n in range(net.layers)])
    biases =  np.array([net.hidden_layers[n].biases1 for n in range(net.layers)])
    np.save('xor_weights', weights, allow_pickle=True) 
    np.save('xor_biases', biases, allow_pickle=True) 
    np.save('xor_fanout_codes', fanout_codes, allow_pickle=True) 
    print('Saved Successfully') 

def load():
    weights = np.load('xor_weights.npy', allow_pickle=True)

    fanout_codes = np.load('xor_fanout_codes.npy', allow_pickle=True) 

    biases = np.load('xor_biases.npy', allow_pickle=True)

    return weights, fanout_codes, biases 

# learn(net) 
# save(net)

#
# input('add layer...') 
net.growth_flag = True
# w, f, b = load()
# net.import_weights(w, f)
# net.import_biases(b) 
learn(net)
#input('try normalized weights') 
net.normalize_weights(bias_scale=True) 
net.set_weight_bounds(-1, 1) 

learn(net) 
for h in net.hidden_layers:
    for w in h.w1:
        for y in w:
            if abs(y) > 1:
                input('What the heck') 
save(net) 
input('all good') 

'''block that adds a layer to the network'''
# #now add layer to network
# net.layers += 1
# net.layer_sizes.insert(-1, 3)
# l = fl.actual_fanout_layer(net.layer_sizes[1], 3)
# l.w1 = np.array([w[-1], w[-1], w[-1]]).T[0]
# # # print(np.shape(l.w1[0]))
# # # input()
# net.hidden_layers.insert(-1, l)
# w1 = np.array([[np.random.rand()*2-1 for i in range(1)] for j in range(3)])
# net.hidden_layers[-1].w1 = w1
# net.hidden_layers[-1].ninputs = 3

# net.set_activation(spiking_activation) 
# net.set_der(spiking_der) 
# net.set_bias_bounds(min_bias, max_bias) 
# net.set_weight_bounds(-1, 1) 
# net.hidden_layers[-2].randomize_biases()

# # # #set only new layer to be the one learning 
# net.set_layers_to_grow([])
# net.set_layers_to_adjust([2])

# # #input(net.hidden_layers)



# net.update_layer_sizes()

# #test
# # print('layer sizes: ', net.layer_sizes)
# # print('actual: ', [l.noutputs for l in net.hidden_layers]) 
# # input()

# #learn again 
# learn(net) 
# input()
save(net) 
input()
'''end of block adding a layer'''

print('distilling net')



weights, fanout_codes, biases = load() 


# print([np.shape(w) for w in weights]) 
# input()

net1 = fl.fanout_network(3, 1, 3, [3, net.layer_sizes[1], net.layer_sizes[2], 1], fanout=0, learning_rate=.1, ordered=False, growth=True)
net1.set_activation(spiking_activation)
net1.set_der(spiking_der) 
#net1.set_layers_to_grow([]) 
net1.set_bias_bounds(min_bias, max_bias) 
net1.set_weight_bounds(-1, 1) 

wt1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
wt2 = [[.5, .5, 0], [-.5, -.5, 0], [0, 0, 1]]
wt3 = [[1, 0, 1], [0, 0, 0], [0, 1, -1]]
wt4 = [[0, 0, 0], [1, 0, 1], [0, 0, 0]]
bt1 = [.1, .1, .1]
bt2 = [.1, .1, .1]
bt3 = [.1, .1, .1]
bt4 = [.1, 1, .1] 

weights_test = np.array([wt1, wt2, wt3, wt4])
biases_test = np.array([bt1, bt2, bt3, bt4]) 

net1.import_weights(weights, fanout_codes)
#net1.import_weights(weights_test, weights_test) 


#print(net1.hidden_layers[0].noutputs, net1.hidden_layers[1].ninputs) 
# input()

net1.import_biases(biases)
#net1.import_biases(biases_test) 


learn(net1)
for i in range(len(xor_inputs)):
    result = net1.activate(xor_inputs[i]) 
    answer = xor_outputs[i]
    e = answer - result 
    print(xor_inputs[i], ':, ', result, answer, e)

print('starting distillation:')
input() 

net1.growth_flag = False 

for l in [0, 1]:
    while net1.hidden_layers[l].noutputs > 12:
        net1.distill(l, net1.hidden_layers[l].noutputs-1) 
        #print(net1.layer_sizes)
        #input()
        learn(net1) 
        print('layer size: ', net1.hidden_layers[l].noutputs) 
    break 
    input('next layer....') 

weights = np.array([net1.hidden_layers[n].w1 for n in range(net1.layers)])
fanout_codes = np.array([net1.hidden_layers[n].fanout_encoding1 for n in range(net1.layers)])
biases =  np.array([net1.hidden_layers[n].biases1 for n in range(net1.layers)])


np.save('xor_weights', weights, allow_pickle=True) 
np.save('xor_biases', biases, allow_pickle=True) 
np.save('xor_fanout_codes', fanout_codes, allow_pickle=True) 

for l in [0, 1, 2]:
    print('layer size: ', net1.hidden_layers[l].noutputs) 







