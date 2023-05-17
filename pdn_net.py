'''

Author: Ryan Watson

Class 'PDN_Network' for real-time simulation 
of multiple phase domain neurons
in a generic network

'''


import numpy as np

import PhaseDomainNeuron as pdn

class PDN_Network():
    def __init__(self):
        self.ninputs = 0
        self.noutputs = 0

        self.input_layer = []

        self.hidden = []

        self.output_layer = []

        self.fanout = 0 
        self.learning_rate = .01 

        self.MAX_VREF = 340
        self.MIN_VREF = 0 

        '''experimental PID controller parameters for variable learning rate'''
        self.E = 1
        self.I = .1
        self.D = .1

        self.connections = {} 
        self.neurons = {} 


        #plotting tools
        self.STREAM_SIZE = int(8e2) 
        self.input_stream = [0 for i in range(self.STREAM_SIZE)] 
        self.output_stream = [0 for i in range(self.STREAM_SIZE)] 

    '''sets up all layers with neuron objects
       and initializes random biases and weights
       between layers'''
    def configure(self, params):
        '''pass dict of params for network'''
        self.ninputs = params['ninputs']
        self.noutputs = params['noutputs']
        self.fanout = params['fanout'] 

        key = 1

        #create layers

        self.input_layer = [pdn.PDN(vref = 10, key=key+i) for i in range(self.ninputs)] 

        key += self.ninputs 
        
        tmp = params['hidden'] 
        
        for t in tmp:
            self.hidden.append([pdn.PDN(vref = np.random.rand()*340, key=key+i) for i in range(t)])
            key += t  

        self.output_layer = [pdn.PDN(vref = np.random.rand()*340, key=key+i) for i in range(self.noutputs)] 

        #aggregate all neurons in all layers
        for n in self.input_layer + self.output_layer:
            self.neurons[n.key] = n 
        for h in self.hidden:
            for n in h:
                self.neurons[n.key] = n 

        #create network connections between layers
        #does not consider fanout limitations yet
        if len(self.hidden) > 0:
            for i in self.input_layer:
                for h in self.hidden[0]:
                    self.connections[(i.key, h.key)] = 1*np.random.rand()

            for i in range(len(self.hidden)-1):
                h1 = self.hidden[i]
                h2 = self.hidden[i+1]
                for i in h1:
                    for j in h2:
                        self.connections[(i.key, j.key)] = 1*np.random.rand()

            for h in self.hidden[-1]:
                for j in self.output_layer:
                    self.connections[(h.key, j.key)] = 1*np.random.rand()
        else:
            for i in self.input_layer:
                for j in self.output_layer:
                    self.connections[(i.key, j.key)] = 1*np.random.rand()

    def update_stream_size(self, size):
        self.input_stream += [0 for i in range(size-self.STREAM_SIZE)]
        self.output_stream += [0 for i in range(size-self.STREAM_SIZE)]
        self.STREAM_SIZE = int(size)

    def import_weights(self, weights, fanout_codes):
        '''i really, really hope this works'''

        #first, zero out all connections
        for c in self.connections:
            self.connections[c] = 0 

        temp = {}
        layer_sizes = [self.ninputs] + self.hidden + [self.noutputs] 
        for i in range(self.ninputs):
            for j in fanout_codes[0][i]:
                n1 = self.input_layer[i]
                n2 = self.hidden[0][j]
                temp[n1.key, n2.key] = weights[0][i][j]
        for i in range(len(self.hidden)-1):
            for j in range(len(self.hidden[i])):
                for k in fanout_codes[i+1][j]:
                    n1 = self.hidden[i][j]
                    n2 = self.hidden[i+1][k]
                    temp[n1.key, n2.key] = weights[i+1][j][k] 

        for i in range(len(self.hidden[-1])):
            for j in fanout_codes[-1][i]:
                n1 = self.hidden[-1][i]
                n2 = self.output_layer[j]
                temp[n1.key, n2.key] = weights[-1][i][j] 
          
        self.connections = temp 

    def import_biases(self, biases):
        '''biases ought to be array of vectors for biases in each layer'''

        '''need to intelligently scale the biases to fit the vref range of the pdn's'''
        #first layer
        for i in range(len(self.hidden[0])):
            n = self.hidden[0][i]
            b = biases[0][i]
            n.update_vref(self.MAX_VREF*(b+self.ninputs)/(2*self.ninputs))
        #the hidden layers
        for i in range(1, len(self.hidden)):
            for j in range(len(self.hidden[i])):
                n = self.hidden[i][j]
                b = biases[i][j]
                n.update_vref(self.MAX_VREF*(b+len(self.hidden[i-1]))/(2*len(self.hidden[i-1])))

        #output layer
        for i in range(self.noutputs):
            n = self.output_layer[i]
            b = biases[-1][i]
            n.update_vref(self.MAX_VREF*(b+len(self.hidden[-1]))/(2*len(self.hidden[-1])))
        pass 
    '''pushes inputs onto input neurons and
       pulls output from output neurons'''

    def forward(self, input_vector):
        self.input_stream.append(input_vector)
        self.input_stream.pop(0) 
        for i in range(self.ninputs):
            self.input_layer[i].forward(input_vector[i])

        for c in self.connections:
            n1 = self.neurons[c[0]]
            n2 = self.neurons[c[1]] 
            o = (n1.output()) * self.connections[c] 
            n2.forward(o) 

        output = [n.output() for n in self.output_layer] 

        '''this part only makes sense for a scalar output'''
        self.output_stream.append(output[0])
        self.output_stream.pop(0) 

        return output


    '''pushes feedback onto the output neurons'''

    def backward(self, feedback_vector):
        for i in range(self.noutputs):
            self.output_layer[i].backward(feedback_vector[i]) 

        for c in self.connections:
            n1 = self.neurons[c[0]]
            n2 = self.neurons[c[1]] 
            o = abs(n2.backpropagate())*self.connections[c] 
            #print(feedback_vector[0], o)
            n1.backward(o) 

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x)) 

    def tweak_learning_rate(self):
        error = np.array([n.feedback_stream[-100:-1] for n in self.output_layer])

        error = np.sum(error, axis=0)
        total = abs(np.sum(error)) 

        diff = error[-98:-1] - error[-99:-2] 
  
        dev = np.std(diff) 

        self.learning_rate += -1*self.sigmoid(dev)*self.learning_rate + total*self.I 


    '''Tick all neurons in the network'''

    def tick_network(self):
        for n in self.neurons:
            self.neurons[n].tick() 

        '''weights not updated since imported externally'''
        # for c in self.connections:
        #     n1 = self.neurons[c[0]]
        #     n2 = self.neurons[c[1]]

        #     self.connections[c] += n1.output() * n2.backpropagate() * self.learning_rate  





class pdn_network():
    def __init__(self, ninputs, noutputs, hidden):
        '''hidden is list of layer sizes, excluding input layer and output layer'''
        self.ninputs = ninputs
        self.noutputs = noutputs 

        self.hidden = hidden 

        self.input_layer = [pdn.PDN() for i in range(ninputs)] 

        self.hidden_layers = [[pdn.PDN() for i in range(j)] for j in hidden]

        # self.hidden_layers.pop(0)
        # self.hidden.pop(0)

        self.output_layer = [pdn.PDN() for i in range(noutputs)]

        self.weights = []
        self.biases = []

        #plotting stuf
        self.stream_size = 10000
        self.output_stream = [0 for i in range(self.stream_size)]

        #stuff for saving data
        self.weights_file_addr = 'weights.npy'
        self.biases_file_addr = 'biases.npy' 

    def Save(self):
        np.save('weights', self.weights, allow_pickle=True)
        self.enumerate_biases() 
        np.save('biases', self.biases, allow_pickle=True) 
        print('Saved successfully') 

    def Load(self):
        weights = np.load(self.weights_file_addr, allow_pickle=True)
        #input(weights)
        biases = np.load(self.biases_file_addr, allow_pickle=True)
  
        self.weights = weights 

        '''input neuron vref not set here!'''

        layers = self.hidden_layers + [self.output_layer] 
        for i in range(len(layers)):
            for j in range(len(layers[i])):
                if hasattr(layers[i], 'iter'):
                    layers[i].update_vref(biases[i][j])
                else:
                    layers[i][j].update_vref(biases[i][j])

        print('Loaded successfully')
        
    def enumerate_biases(self):
        temp = []
        for l in self.hidden_layers:
            te = []
            for n in l:
                te.append(n.vref)
            temp.append(te)
        tem = []
        for n in self.output_layer:
            tem.append(n.vref) 
        temp.append(tem) 
        self.biases = temp 

    def activate(self, input1):
        input1 = [self.input_layer[i].forward(input1[i]) for i in range(self.ninputs)]
        #print('first result:', input1)
        # print(self.weights[0]) 
        input1 = np.dot(input1, self.weights[0])  
      

        for i in range(len(self.hidden)-1):
            input1 = [self.hidden_layers[i][j].forward(input1[j]) for j in range(self.hidden[i])]
            #print(input1)
            input1 = np.dot(input1, self.weights[i]) 
            
        # print(self.weights[-1])
        # input()
        self.output = [self.output_layer[i].forward(input1[i]) for i in range(self.noutputs)]
        #print([n.vref for n in self.output_layer])
        # print(self.weights[-1]) 
        # input()
        #self.tick_network()
        return self.output 

    def activate_hidden_layer(self, input1, layer_num):
        '''layer_num=0 for output of first hidden layer, etc.'''
        input1 = [self.input_layer[i].forward(input1[i]) for i in range(self.ninputs)]

        input1 = np.dot(input1, self.weights[0])  
      
        for i in range(layer_num):
            input1 = [self.hidden_layers[i][j].forward(input1[j]) for j in range(self.hidden[i])]
            #print(input1)
            input1 = np.dot(input1, self.weights[i+1])

        return [self.hidden_layers[layer_num][j].forward(input1[j]) for j in range(self.hidden[layer_num])]

    '''often more useful to get a whole burst of outputs for a single driven
        input, for an average energy output metric. Well gents, here 'tis'''
    def activate_burst(self, input1, length):
        output = []

        for i in range(length):
            #print(i+1)
            output.append(self.activate(input1)) 
            #self.tick_network() 
        return output

    def hidden_layer_burst(self, input1, length, hidden_layer_num):
        output = []
        for i in range(length):
            #print(i+1)
            output.append(self.activate_hidden_layer(input1, hidden_layer_num)) 
            self.tick_network() 
        return output

    def import_biases(self, biases):
        '''ensure biases are already transformed'''
        for i in range(self.ninputs):
            self.input_layer[i].update_vref(biases[0][i])

        for i in range(len(self.hidden)):
            for j in range(self.hidden[i]):
                self.hidden_layers[i][j].update_vref(biases[i][j]) 

        for i in range(self.noutputs):
            self.output_layer[i].update_vref(biases[-1][i]) 

    def tick_network(self):
        for n in self.input_layer:
            n.tick()
        for l in self.hidden_layers:
            for n in l:
                n.tick()
        for n in self.output_layer:
            n.tick() 

    def calibrate_network_biases(self, max_input, biases):
        '''this assumes a multiplicative bias relu-type unit'''
        b = biases[0] 
        print('calibrating input layer') 
        for i in range(self.ninputs):
            print(i)
            #max_output = max_input[i]*b[i]
            max_output = b[i]
            print(max_input)
            result = self.input_layer[i].calibrate_vref(max_input[i], max_output) 
            # if not result:
            #     print('neuron calibration failed') 

        max_input = [np.average(n.forward_burst(max_input[i], iter=100)) for n in self.input_layer] 

        w = self.weights[0]
        temp = max_input.copy() 
        max_input = np.zeros(np.shape(self.weights[0])[1]) 
        for i in range(len(max_input)):
            max_input[i] = np.sum([w[j][i]*temp[j] for j in range(len(temp)) if w[j][i] > 0])

        print('\n\ncalibrating hidden layers') 
        
        for j in range(len(self.hidden_layers)-1):
            b = biases[j] 
            print('layer: ', j) 
            for i in range(len(self.hidden_layers[j])):
                print(i)
                if hasattr(b, '__iter__'):
                    #max_output = max_input[i]*b[i]
                    max_output = b[i] 
                else:
                   #max_output = max_input[i]*b 
                   max_output = b
                #input(max_input[i]) 
                result = self.hidden_layers[j][i].calibrate_vref(max_input[i], max_output) 
                # if not result:
                #     print('neuron calibration failed') 
            
           
            max_input = [np.average(n.forward_burst(max_input[i], iter=100)) for n in self.hidden_layers[j]]

            w = self.weights[j]#+1]
            temp = max_input.copy()
            max_input = np.zeros(np.shape(self.weights[j])[1]) 
            for i in range(len(max_input)):
                max_input[i] = np.sum([w[k][i]*temp[k] for k in range(len(temp)) if w[k][i] > 0])

        print('\n\ncalibrating output layer') 
        #output layer          
        b = biases[-1] 
        for i in range(self.noutputs):
            print(i) 
            #max_output = max_input[i]*b[i] 
            max_output = b[i]
            result = self.output_layer[i].calibrate_vref(max_input[i], max_output) 
            # if not result:
            #     print('neuron calibration failed') 

        #store calibrated vrefs in self.biases
        b1 = [n.vref for n in self.input_layer]
        b = [[n.vref for n in l] for l in self.hidden_layers]
        b_last = [n.vref for n in self.output_layer] 
        self.biases.append(b1)
        for v in b:
            self.biases.append(v)
        self.biases.append(b_last) 

  
            

























