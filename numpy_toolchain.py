import numpy as np 
import pickle

debug_print = True 

project_name = 'xor_test'

'''train by any means - growth and evolution'''
samples = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]] 
answers = [0, 1, 1, 0, 1, 0, 0, 1]  

import mlp 

#methods for saving and loading networks
def save(net, name):
    weights = np.array([net.hidden_layers[n].w1 for n in range(net.layers)])
    fanout_codes = np.array([net.hidden_layers[n].fanout_encoding1 for n in range(net.layers)])
    biases =  np.array([net.hidden_layers[n].biases1 for n in range(net.layers)])
    np.save('xor_weights_'+name, weights, allow_pickle=True) 
    np.save('xor_biases_'+name, biases, allow_pickle=True) 
    np.save('xor_fanout_codes_'+name, fanout_codes, allow_pickle=True) 

    print('Saved Successfully') 

def load(name):
    weights = np.load('xor_weights_'+name+'.npy', allow_pickle=True)

    fanout_codes = np.load('xor_fanout_codes_'+name+'.npy', allow_pickle=True) 

    biases = np.load('xor_biases_'+name+'.npy', allow_pickle=True)

    return weights, fanout_codes, biases 

def save_net(net, name):
    with open(name+".pickle", "wb") as f:
        pickle.dump(net, f)
    print('Saved Successfully')

def load_net(name):
    net = None 
    with open(name+".pickle", "rb") as f:
        net = pickle.load(f)
    return net 

#learning subroutine 
def learn(net, samples, answers, error_margin):
    error = 10 

    epoch = 0

    while error > error_margin and epoch < 10:
        epoch += 1
        # print(net.layer_sizes)
        # input() 
        error, iter = mlp.run_learn_cycle(net, samples, answers, error_margin, len(samples)) 
        print('Epoch: ', epoch)
        print('Iter: ', iter) 
        print('error: ', error) 
        print('size of network: ', len(net.hidden_layers[0].biases1))
        print('biases: ', net.hidden_layers[0].biases1)

    #print results 
    if debug_print:
        for i in range(len(samples)):
            result = net.activate(samples[i]) 
            answer = answers[i]
            e = answer - result 
            print(samples[i], ':, ', result, answer, e)
        print('size of network: ', len(net.hidden_layers[0].biases1))
        print('biases: ', net.hidden_layers[0].biases1) 
        print(net.hidden_layers[1].biases1) 
        input('press enter to continue...') 

acceptable_error_margin = .5
ninputs = len(samples[0]) 
noutputs = 1#len(answers[0]) 
hidden = 12

#since voltage biases in the pdn network are range-limited, stands to reason their analogs here would be as well 
min_bias = .18
max_bias = 1
min_weight = -1
max_weight = 1 

def create_new_network():
    #initialize network
    ninputs = len(samples[0]) 
    noutputs = 1#len(answers[0]) 
    hidden = 12

    #since voltage biases in the pdn network are range-limited, stands to reason their analogs here would be as well 
    min_bias = .18
    max_bias = 1
    min_weight = -1
    max_weight = 1 

    weights, fanout_codes, biases = load(project_name) 

    net = mlp.mlp(ninputs, noutputs, 3,[ninputs, 20, 12, noutputs], growth=True) 
    net.growth_flag = True  #just to make sure


    net.set_activation(mlp.spiking_activation) 
    net.set_der(mlp.spiking_der) 
    net.set_bias_bounds(min_bias, max_bias)
    #
    # net.set_weight_bounds(min_weight, max_weight) 

    net.randomize_biases()
    net.set_layers_to_grow([2]) 
    net.set_layers_to_adjust([2])


    #train the network
    learn(net, samples, answers, acceptable_error_margin) 

    save(net, project_name) 
    save_net(net, project_name) 

#net.update_layer_sizes()

#save(net, project_name) 


# net.import_biases(biases)
# net.import_weights(weights, fanout_codes) 


# # for i in range(len(samples)):
# #     result = net.activate(samples[i]) 
# #     answer = answers[i]
# #     e = answer - result 
# #     print(samples[i], ':, ', result, answer, e)
# # input()

# #try normalizing weights before distillation
#NORMALIZING


# save(net, project_name)
#load(project_name)

'''prune as far as possible through simple distillation'''
def shrink():
    target_layer_sizes = [3, 3]

    weights, fanout_codes, biases = load(project_name) 
    net = mlp.mlp(ninputs, noutputs, 3,[ninputs, len(biases[0]), len(biases[1]), noutputs], growth=False) 

    net.import_biases(biases) 
    net.import_weights(weights, fanout_codes)

    #normalize weights
    scaling_factors = net.normalize_weights(False) 
    net.set_output_amplification(scaling_factors) 

    net.growth_flag=False 
    net.set_weight_bounds(min_weight, max_weight) 


    for i in range(len(samples)):
        result = net.activate(samples[i]) 
        answer = answers[i]
        e = answer - result 
        print(samples[i], ':, ', result, answer, e)
    input()

    net, size_results = mlp.distill(net, target_layer_sizes, samples, answers, acceptable_error_margin, len(samples))

    for i in range(len(samples)):
        result = net.activate(samples[i]) 
        answer = answers[i]
        e = answer - result 
        print(samples[i], ':, ', result, answer, e)
    input()

    if debug_print:
        print('new layer sizes: ', size_results) 
        response = input('wanna save?')
        if response=='y':
            save(net, project_name)
            save(net, project_name) 
        input('continue...') 
    

    for i in range(len(samples)):
        result = net.activate(samples[i]) 
        answer = answers[i]
        e = answer - result 
        print(samples[i], ':, ', result, answer, e)
    input()

# print(net.hidden_layers[0].w1)
# print(net.hidden_layers[-1].w1) 

# create_new_network()
# shrink()

#input()

'''*experimental* matrix conversions'''

def condense():
    weights, fanout_codes, biases = load(project_name) 
    # net = mlp.mlp(ninputs, noutputs, 3,[ninputs, len(biases[0]), len(biases[1]), noutputs], growth=False) 

    net = load_net(project_name)  

    # net.import_biases(biases) 
    # net.import_weights(weights, fanout_codes)
    # net.set_activation(mlp.spiking_activation) 
    # net.set_der(mlp.spiking_der) 

    for i in range(len(samples)):
        result = net.activate(samples[i]) 
        # result = net1.hidden_layers[1].activate(net1.activation(net1.hidden_layers[0].activate(samples[i]), net1.hidden_layers[0].biases1))
        # result = np.dot(result, net1.hidden_layers[-1].w1) 
        answer = answers[i]
        e = answer - result 
        print(samples[i], ':, ', result, answer, e)
    input()

    #experimental, start by just combining inner layer matrices

    U = net.hidden_layers[0].w1 
    sh = np.shape(U)
    neg_vec = -1*np.ones(sh[0]) / np.sqrt(sh[0]) 
    filter_vec = np.dot(neg_vec, U) 



    new_weights = net.hidden_layers[0].w1@net.hidden_layers[1].w1 
    test = -1 * new_weights.T

    s = np.shape(new_weights) 
    
    net1 = mlp.mlp(ninputs, noutputs, 3,[ninputs, s[1], 3, noutputs], growth=False) 
    net1.set_activation(mlp.spiking_activation) 
    net1.set_der(mlp.spiking_der) 

    net1.output_amplification = net.output_amplification


  
    #new weights again
    W1 = []
    #pos correlated features
    w = np.maximum(weights[-1], np.zeros(s[0]))[:,0]
    W1.append(w) 
    #neg correlated features
    w = -1 * np.maximum(-1*weights[-1], np.zeros(s[0]))[:,0] 

    W1.append(w) 

    #suppressed feature
    bad_vector = -1*np.ones(sh[1]) / np.sqrt(sh[1]) 
    #bad vector after transform
    bad_vector = np.dot(bad_vector, net.hidden_layers[1].w1 ) 
    W1.append(bad_vector) 

    W1 = np.array(W1)

 
    # print(W1@(np.maximum(input2@new_weights, np.zeros(6)))) 

    # print(weights[-1]) 
    # print(W1) 
    #input()

    end_weights = np.array([[1, 1, 1]]).T


    W = [new_weights, W1.T, end_weights]
    B = [biases[1], np.array([1, 1, 1])]  
    

    net1.import_biases(B)
    net1.import_weights(W, fanout_codes)

    net1.hidden_layers[-1].biases1 = [1] 

    #input(net1.hidden_layers[-1].biases1) 

    for i in range(len(samples)):
        result = net1.activate(samples[i]) 
        result1 = net1.hidden_layers[0].activate(samples[i])
        result2 = net1.hidden_layers[1].activate(result1)
        result3 = np.dot(result2, net1.hidden_layers[2].w1) 
        result4 = net1.activation(result3, net1.hidden_layers[-1].biases1) 
        answer = answers[i]
        e = answer - result 
        print(samples[i], ':, ', result, result1, result2, result3,result4, answer, e)
    input()
    net1.set_layers_to_adjust([1, 2])
    learn(net1, samples, answers, .5) 
    #net1 = mlp.run_learn_cycle(net1, samples, answers, .5, 8)[0]

    for i in range(len(samples)):
        result = net1.activate(samples[i]) 
        # result = net1.hidden_layers[1].activate(net1.activation(net1.hidden_layers[0].activate(samples[i]), net1.hidden_layers[0].biases1))
        # result = np.dot(result, net1.hidden_layers[-1].w1) 
        answer = answers[i]
        e = answer - result 
        print(samples[i], ':, ', result, answer, e)
    input()

    net2 = mlp.mlp(ninputs, noutputs, 3,[ninputs, s[1], s[1], noutputs], growth=False) 

    filter_matrix = 1 
# condense() 
# input()


net = 1
'''transform weights and biases to pdn form'''
weights, fanout_codes, biases = load(project_name)
#create pdn network object
import pdn_net as pn 
pdnNet = pn.pdn_network(ninputs, noutputs, [net.hidden_layers[0].noutputs])

pdnNet.Load() 
pdnNet.weights = weights 

'''calibrate pdn network'''
#pdnNet.calibrate_network_biases([1, 1, 1], biases) 

pdnNet.input_layer[0].update_vref(1)
pdnNet.input_layer[1].update_vref(1)
pdnNet.input_layer[2].update_vref(1) #temporary

pdnNet.output_layer[0].update_vref(.1) 

'''test pdn network'''
test_results = []
iter = 0
for s in samples:
    print(iter+1)
    iter += 1
    o = pdnNet.activate_burst(s, 1000)
    test_results.append(np.average(o)) 

for i in range(len(test_results)):
    print(samples[i], ': ', test_results[i], '; ', answers[i]) 

'''save all results'''
pdnNet.Save() 

