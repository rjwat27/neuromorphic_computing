import numpy as np
#import PhaseDomainNeuron as pdn 
xor_inputs = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]] 
xor_outputs = [0, 1, 1, 0, 1, 0, 0, 1]  

def relu(x, bias):
    return np.maximum(0, np.arctan(x+bias)) 
def sig(x, bias):
    return relu(x, bias) 
    return 1/(1+np.exp(-(x+bias))) 
def sig_der(x, bias):
    return 1 if x > 0 else .1#(1/(1+np.exp(-x)))*(1-1/(1+np.exp(-x))) 
    return sig(x, bias)*(1-sig(x, bias)) 


  


class actual_fanout_layer():
    '''simple one hidden layer network'''
    def __init__(self, inputs, outputs, learning_rate=.1, fanout=0, activation_function=sig, ordered=False):
        self.ninputs = inputs
        self.noutputs = outputs
        
        self.fanout = fanout 

        self.activation_function = activation_function
        self.act_der = sig_der 

        self.input = 0
        self.input1 = 0
        

        self.biases1 = np.array([-inputs/2 + i*inputs/outputs for i in range(self.noutputs)])



        self.delta1 = np.zeros(self.noutputs)

        self.max_change = 0 
        #TODO for many layers may be wise and even very effective to test convergence on individual 
        #layers and update them individually on a rollover signal from the following layer-
        #for now just a single max_change for global convergence, though this may drastically increase learning times as global convergence
        #on large mult-layer networks will take a while

        self.learning_rate = learning_rate

        self.w1 = np.array([[np.random.rand()*2-1 for i in range(self.noutputs)] for j in range(self.ninputs)])
        self.weight_low = -100
        self.weight_high = 100


        '''set up fanout weight connections'''
        '''for now just a random wiring, TODO option to link manually or with patterned automation instead'''

        layer1 = range(self.noutputs) 
        if self.fanout < self.noutputs and self.fanout!=0:
            if ordered:
                #for i in range(self.ninputs):
                self.fanout_encoding1 = [[(i+j)%self.noutputs for j in range(self.fanout)] for i in range(self.ninputs)]
            else:
                self.fanout_encoding1 = [np.random.choice(layer1, size=self.fanout, replace=False) for i in range(self.ninputs)]
        else:
            self.fanout_encoding1 = [layer1 for j in range(self.ninputs)] 

        self.feature_similarity_threshold = .9 #arbitario

        self.back_signal = [0 for i in range(self.ninputs)] 

    def set_activation(self, function):
        self.activation_function = function 

    def set_der(self, function):
        self.act_der = function 

    def activate(self, input):
     
        self.input = input
        if self.fanout != 0:
            self.input1 = np.zeros(self.noutputs) 
            for i in range(self.ninputs):
                for j in self.fanout_encoding1[i]:
                    self.input1[j] += self.w1[i][j]*input[i]
        else:
            self.input1 = np.dot(input, self.w1) 
        
        output= self.activation_function(self.input1, self.biases1)
        self.output = output 
        return output 
   
    def delta(self, feedback):
        '''update learning rate''' 
        self.learning_rate = .01
        bias_changes = []
        ders = np.zeros(self.noutputs) 

        for i in range(self.noutputs):
            self.delta1[i] = feedback[i]
            bias_changes.append(self.learning_rate*feedback[i]*self.output)
            ders[i] = self.act_der(self.output[i]*feedback[i], 0) 
        '''how to reconcile custom activation function with sig_der?'''
        #ders = np.array([sig_der(self.output[i])*feedback[i] for i in range(self.noutputs)])
        
        if self.fanout != 0: 
            self.back_signal = [np.sum([ders[j]*self.w1[i][j] for j in self.fanout_encoding1[i]]) for i in range(self.ninputs)]  
        else:
            self.back_signal = [np.dot(ders, self.w1[i]) for i in range(self.ninputs)]  


        return max(bias_changes[0])  
        
    def backpropogate(self):
        return self.back_signal 

    def adjust(self):
        weight_changes = []
    
        for i in range(self.ninputs):
            for j in range(self.noutputs):
                weight_changes.append(self.learning_rate*self.delta1[j]*self.input[i])             
                self.w1[i][j] += self.learning_rate*self.delta1[j]*self.input[i] 
                self.w1[i][j] = min(max(self.weight_low, self.w1[i][j]), self.weight_high)

        self.max_change = max(weight_changes)
    
        return self.max_change

    def randomize_biases(self):
        self.biases1 = [np.random.rand()*(self.bias_b2 - self.bias_b1) + self.bias_b1 for i in range(self.noutputs)] 

    def set_bias_bounds(self, bound1, bound2):
        self.bias_b1 = bound1
        self.bias_b2 = bound2 

    def set_weight_bounds(self, b1, b2):
        self.weight_low = b1
        self.weight_high = b2

    def normalize_weights(self, bias_scale=False):
        '''scale weight values between -1 and 1'''
        W = self.w1.T
        new_W = np.zeros((self.noutputs, self.ninputs)) 
        for i in range(self.noutputs):
            w = W[i]
            big = max(np.abs(w))
            #input(big)
            new_w = w / big 
            new_W[i] = new_w 
            if bias_scale:
                self.biases1[i] *= big
        self.w1 = new_W.T  
        pass 

    def adjust_biases(self):
        for i in range(self.noutputs):
            #TODO remove dependency on the output 
            self.biases1[i] += (self.learning_rate*self.delta1[i])#*self.output[i])#*sig_der(self.input1[i])

    def add_hidden_node(self, num=1):
        # if self.noutputs > 100: #remove?
        #     return 
        for g in range(num):
            #average_bias_space = np.random.rand()*2*self.ninputs - self.ninputs
            average_bias_space = np.random.rand()*(self.bias_b2 - self.bias_b1) + self.bias_b1
            new_weights = np.array([np.random.rand()*2 - 1  for i in range(self.ninputs)]) 

            self.w1 = np.append(self.w1.T, np.array([new_weights]), axis=0).T 
            self.biases1 = np.append(self.biases1, np.array([average_bias_space]), axis=0)

            self.delta1 = np.append(self.delta1, np.array([0]), axis=0) 

            self.noutputs += 1

    def prune_worst(self, external_weights, num=1, force=False):
        m = None 
        worst = []
        if num < 1:
            print("a;lsdkjf;asldjldjf")
            input()
        for j in range(num):
            relevance_scores = []
            for i in range(self.noutputs):
                temp = self.w1.T 
                #print(external_weights[i]) 
                if np.max(np.absolute(external_weights[i])) < .01 / (.01 + relu(np.sum(np.absolute(temp[i])), self.biases1[i])) or force:
                    relevance_scores.append(np.max(np.absolute(external_weights[i])))

            if not relevance_scores:
                return None 
    
            m = np.argmin(relevance_scores)
            worst.append(m)  

            self.w1 = np.delete(self.w1.T, m, axis=0).T 
            self.biases1 = np.delete(self.biases1, m, axis=0)
            self.noutputs -= 1

        '''this really only makes sense with one replacement at a time'''
        return worst #other external layers need this information   

    def prune_weights(self, m):
        self.w1 = np.delete(self.w1, m, axis=0) 
        if m is iter:
            self.ninputs -= len(m)
        else:
            self.ninputs -= 1    

    def adjust_fanout(self, new_fanout):
        if self.fanout == new_fanout:
            return 
        # elif new_fanout >= self.ninputs:
        #     print('new fanout size too big dummy')
        #     return 
        elif new_fanout >= self.noutputs:
            return 
        else:
            for i in range(len(self.w1)):
                '''be certain that weights outside the fanout scope are zero'''
                w_sorted = np.argsort(abs(self.w1[i]))
                self.fanout_encoding1[i] = [w_sorted[-j] for j in range(1, new_fanout+1)] 

            self.fanout = new_fanout 

        

    def add_weights(self):
        new_weights = np.array([np.random.rand()*2 for i in range(self.noutputs)])

        self.w1 = np.append(self.w1, np.array([new_weights]), axis=0)

        self.ninputs += 1

    def vectorize_node_weights(self, node):
        vector = np.array([self.w1[i, node] for i in range(self.ninputs)]) 
        return (vector / np.linalg.norm(vector)) #normalize 

    def is_converged(self):
        return (self.max_change < .01) 


class fanout_network():
    '''layer sizes includes the input size and output size
    layers denotes the number of input-set weights into a neuron vector, which equals (num_hidden + output_layer)
    therefore, the layer_sizes will be one greater than layers'''
    def __init__(self, inputs, outputs, layers, layer_sizes=[], fanout=0, learning_rate=.1, ordered=False, growth=False):
        self.ninputs = inputs
        self.noutputs = outputs 
        self.layers = layers 
        self.layer_sizes = layer_sizes
        self.fanout = fanout 
        self.learning_rate = learning_rate

        self.max_change = 0
        self.max_layer_size = 100

        self.growth_flag = growth 
        self.layers_to_grow = [i for i in range(layers)] 
        self.layers_to_adjust = [i for i in range(1, layers)] 

        self.ordered = ordered 

        self.activation = relu 
        self.act_der = sig_der 

        if not layer_sizes:
            print("ya yer gonna need some data on the layer sizes") 
            return 
        else:
            self.hidden_layers = [actual_fanout_layer(self.layer_sizes[i], self.layer_sizes[i+1], self.learning_rate, self.fanout, ordered=ordered) for i in range(self.layers)]

    def set_activation(self, function):
        for l in range(self.layers):
            self.hidden_layers[l].set_activation(function) 
        self.activation = function 

    def set_der(self, function):
        for l in range(self.layers):
            self.hidden_layers[l].set_der(function) 
        self.act_der = function 

    def import_weights(self, weights, fanouts):
       # temp = [self.ninputs]
        for i in range(len(weights)):
            self.hidden_layers[i].w1 = np.array(weights[i])
            #temp.append(np.shape(weights[i])[1]) 
            self.hidden_layers[i].fanout_encoding1 = np.array(fanouts[i]) 
            #np.reshape(self.hidden_layers[i].w1, (self.hidden_layers[i].ninputs, self.hidden_layers[i].noutputs)) 

        #self.layer_sizes = temp 
        

    def import_biases(self, biases):
        for i in range(len(biases)):
            self.hidden_layers[i].biases1 = biases[i] 
   
    def set_bias_bounds(self, bound_low, bound_high):
        for l in range(self.layers):
            self.hidden_layers[l].set_bias_bounds(bound_low, bound_high) 

    def set_weight_bounds(self, w_low, w_high):
        for l in range(self.layers):
            self.hidden_layers[l].set_weight_bounds(w_low, w_high) 

    def activate(self, input1):
        for l in self.hidden_layers:
            #print(l, l.noutputs) 
            #print('TEST: ', np.shape(l.w1))
            input1 = l.activate(input1)

        return input1 
        
    def delta(self, feedback):
        for l in range(1, self.layers+1):
            self.hidden_layers[-l].delta(feedback)
            feedback = self.hidden_layers[-l].backpropogate() 

    def normalize_weights(self, bias_scale):
        for l in self.hidden_layers:
            l.normalize_weights(bias_scale=bias_scale) 

    def adjust(self):
        for l in self.layers_to_adjust:
            j = self.hidden_layers[l]
            #print(l, j.noutputs, self.layers_to_adjust) 
            self.hidden_layers[l].adjust()

    def randomize_biases(self):
        for i in range(self.layers):
            self.hidden_layers[i].randomize_biases() 

    def adjust_biases(self):
        for l in range(self.layers):
            self.hidden_layers[l].adjust_biases() 
        # self.hidden_layers[-1].adjust_biases()
        # self.hidden_layers[-2].adjust_biases() 

    def adjust_final_biases(self, depth=1):
        for l in range(1, depth+1):
            self.hidden_layers[-l].adjust_biases() 

    def adjust_fanout(self, new_fanout):
        for i in range(self.layers):
            self.hidden_layers[i].adjust_fanout(new_fanout=new_fanout) 

    def adjust_specific_biases(self, layer=0):
        self.hidden_layers[layer].adjust_biases() 

    def adjust_final_weights(self):
        self.hidden_layers[-1].adjust() 

    def set_biases_high(self):
        for l in range(self.layers):
            w = self.hidden_layers[l].w1.T
            for i in range(len(self.hidden_layers[l].biases1)):
                # print(len(self.hidden_layers[l].biases1), len(w[i]))
                # input()
                self.hidden_layers[l].biases1[i] = .1#np.sum(w[i])

    def set_layers_to_grow(self, layers):
        self.layers_to_grow = layers

    def set_layers_to_adjust(self, layers):
        self.layers_to_adjust = layers

    def evolve(self, force=False):
        for l in range(1, self.layers):
            if self.hidden_layers[l].is_converged() or force:
                external_weights = self.hidden_layers[l].w1
                #print(self.layer_sizes) 
                n = self.hidden_layers[l-1].prune_worst(external_weights, num=1)  #arbitrario
                print('pre test: ', self.hidden_layers[l-1].noutputs, len(self.hidden_layers[l-1].biases1), n)
                if n!=None:
                    self.hidden_layers[l].prune_weights(n) 
                
                if self.growth_flag and (l in self.layers_to_grow) and len(self.hidden_layers[l].biases1) < self.max_layer_size:
                    #print('got here ', l)
                    self.hidden_layers[l-1].add_hidden_node(3)    #3 arbitrario  
                    for i in range(3):
                        self.hidden_layers[l].add_weights()           
                elif n!=None:
                    print('TEST: ', self.hidden_layers[l-1].noutputs, len(self.hidden_layers[l-1].biases1), n)
                    self.hidden_layers[l-1].add_hidden_node(len(n))
                    for i in n:
                        self.hidden_layers[l].add_weights()
        self.update_layer_sizes()

    def update_layer_sizes(self):
        temp = [self.ninputs]
        for l in self.hidden_layers:
            temp.append(l.noutputs) 
        self.layer_sizes = temp 

    '''layer argument cannot be used to distill output layer'''
    def distill(self, layer, target_size):
  
        external_weights = self.hidden_layers[layer+1].w1

        to_remove = self.hidden_layers[layer].noutputs - target_size

        n = self.hidden_layers[layer].prune_worst(external_weights, num=to_remove, force=True)  #arbitrario

        self.hidden_layers[layer+1].prune_weights(n) 

        self.update_layer_sizes()




class actual_fanout_layer_with_neuron_structures():
    '''simple one hidden layer network'''
    def __init__(self, inputs, outputs, learning_rate=.1, fanout=0, neural_object=None):
        if neural_object==None:
            print("what are you thinking, if no provide object, why not use the simpler version?")
            return
        self.ninputs = inputs
        self.noutputs = outputs
        
        self.fanout = fanout 

        self.neural_object = neural_object

        self.input = 0
        self.input1 = 0
        

        #self.biases1 = np.array([-inputs/2 + i*inputs/outputs for i in range(self.noutputs)])
        '''neural object should accept a 'bias' parameter'''
        self.nodes = np.array([self.neural_object(-inputs/2 + i*inputs/outputs) for i in range(self.noutputs)])


        self.delta1 = np.zeros(self.noutputs)

        self.max_change = 0 
        #TODO for many layers may be wise and even very effective to test convergence on individual 
        #layers and update them individually on a rollover signal from the following layer-
        #for now just a single max_change for global convergence, though this may drastically increase learning times as global convergence
        #on large mult-layer networks will take a while

        self.learning_rate = learning_rate

        self.w1 = np.array([[np.random.rand()*2-1 for i in range(self.noutputs)] for j in range(self.ninputs)])
       


        '''set up fanout weight connections'''
        '''for now just a random wiring, TODO option to link manually or with patterned automation instead'''

        layer1 = range(self.noutputs) 
        if self.fanout < self.noutputs and self.fanout!=0:
            self.fanout_encoding1 = [np.random.choice(layer1, size=self.fanout, replace=False) for i in range(self.ninputs)]
        else:
            self.fanout_encoding1 = [layer1 for j in range(self.ninputs)] 

        self.feature_similarity_threshold = .9 #arbitario

        self.back_signal = [0 for i in range(self.ninputs)] 


    def activate(self, input):
        self.input = input
        self.input1 = np.zeros(self.noutputs) 
        for i in range(self.ninputs):
            for j in self.fanout_encoding1[i]:
                self.input1[j] += self.w1[i][j]*input[i]
        
        '''neuron structure needs a 'forward' method'''
        output= [self.nodes[n].forward(self.input1[n]) for n in range(self.noutputs)] 
        self.output = output 
        return output 

        
    def delta(self, feedback):

        bias_changes = []

        for i in range(self.noutputs):
            self.delta1[i] = feedback[i]
            bias_changes.append(self.learning_rate*feedback[i]*self.output[i])

        for n in range(self.noutputs):
            '''nodes structure need 'backward' method'''
            self.nodes[n].backward(feedback[n]) 

        '''how to reconcile custom activation function with derivative of activation when hard to define or non-existant?'''
        ders = np.array([sig_der(self.output[i])*feedback[i] for i in range(self.noutputs)])
       

        self.back_signal = [np.sum([ders[j]*self.w1[i][j] for j in self.fanout_encoding1[i]]) for i in range(self.ninputs)]  


        return max(bias_changes)  
        
    def backpropogate(self):
        return self.back_signal 

    def adjust(self):
        weight_changes = []
    
        for i in range(self.ninputs):
            for j in range(self.noutputs):
                weight_changes.append(self.learning_rate*self.delta1[j]*self.input[i])             
                self.w1[i][j] += self.learning_rate*self.delta1[j]*self.input[i] 

        self.max_change = max(weight_changes)
    
        return self.max_change



    def add_hidden_node(self, num=1):
        if self.noutputs > 100: #remove?
            return 
        for g in range(num):
            average_bias_space = np.random.rand()*2*self.ninputs - self.ninputs
            new_weights = np.array([np.random.rand()*2 - 1  for i in range(self.ninputs)]) 

            self.w1 = np.append(self.w1.T, np.array([new_weights]), axis=0).T 
            #self.biases1 = np.append(self.biases1, np.array([average_bias_space]), axis=0)
            self.nodes = np.append(self.nodes, self.neural_object(average_bias_space), axis=0)

            self.delta1 = np.append(self.delta1, np.array([0]), axis=0) 


    def prune_worst(self, external_weights, num=1):
        m = None 
        if num < 1:
            print("a;lsdkjf;asldjldjf")
            input()
        if True:#for j in range(num):
            relevance_scores = []
            temp = self.w1.T 
            for i in range(self.noutputs):
                
                if np.max(np.absolute(external_weights[i])) < .01 / (.01 + self.noutputs):
                    relevance_scores.append(np.max(np.absolute(external_weights[i])))

            if not relevance_scores:
                return None 
    
            m = np.argmin(relevance_scores) 

            self.w1 = np.delete(self.w1.T, m, axis=0).T 
            self.nodes = np.delete(self.nodes, m, axis=0)

        '''this really only makes sense with one replacement at a time'''
        return m #other external layers need this information   

    def prune_weights(self, m):
        self.w1 = np.delete(self.w1, m, axis=0)    

    def add_weights(self):
        new_weights = np.array([np.random.rand()*2 for i in range(self.noutputs)])

        self.w1 = np.append(self.w1, np.array([new_weights]), axis=0)

    def vectorize_node_weights(self, node):
        vector = np.array([self.w1[i, node] for i in range(self.ninputs)]) 
        return (vector / np.linalg.norm(vector)) #normalize 

    def tick_neurons(self):
        for n in self.nodes:
            n.tick() 

    def is_converged(self):
        return (self.max_change < .01) 


class fanout_network_with_neuron_structures():
    '''layer sizes includes the input size and output size'''
    '''layers denotes the number of input-set weights into a neuron vector, which equals (num_hidden + output_layer)'''
    '''therefore, the layer_sizes will be one greater than layers'''
    def __init__(self, inputs, outputs, layers, layer_sizes=[], fanout=0, learning_rate=.1):
        self.ninputs = inputs
        self.noutputs = outputs 
        self.layers = layers 
        self.layer_sizes = layer_sizes
        self.fanout = fanout 
        self.learning_rate = learning_rate

        self.max_change = 0


        if not layer_sizes:
            print("ya yer gonna need some data on the layer sizes") 
            return 
        else:
            self.hidden_layers = [actual_fanout_layer_with_neuron_structures(self.layer_sizes[i], self.layer_sizes[i+1], self.learning_rate, self.fanout, neural_object=pdn.PDN) for i in range(self.layers)]

    def activate(self, input1):
        for l in self.hidden_layers:
            input1 = l.activate(input1)

        return input1 
        
    def delta(self, feedback):
        for l in range(1, self.layers+1):
            self.hidden_layers[-l].delta(feedback)
            feedback = self.hidden_layers[-l].backpropogate() 

    def adjust(self):
        for l in range(1, self.layers):
            self.hidden_layers[l].adjust()

    def evolve(self, force=False):
        for l in range(1, self.layers):
            if self.hidden_layers[l].is_converged() or force:
                external_weights = self.hidden_layers[l].w1
                # if max(1, int(np.log10(self.hidden_layers[l-1].noutputs))) != 1:
                #     print("invalid number of prune neurons entered to prune_worst")
                #     print(max(1, int(np.log10(self.hidden_layers[l-1].noutputs))), l)
                #     input()
                n = self.hidden_layers[l-1].prune_worst(external_weights, num=1)  #arbitrario
                # if (self.hidden_layers[l-1].noutputs - len(self.hidden_layers[l-1].biases1)) <= 0:
                #     print('ERROR REPLACING NODE, NON-POSITIVE INPUT GIVEN')
                #     input() 
                
                if n!=None:
                    self.hidden_layers[l].prune_weights(n) 
                
                    self.hidden_layers[l-1].add_hidden_node(self.hidden_layers[l-1].noutputs - len(self.hidden_layers[l-1].biases1))
                    self.hidden_layers[l].add_weights() 

    def tick_neurons(self):
        for l in self.hidden_layers:
            l.tick_neurons() 




'''standard epoch run'''
def run_learn_cycle(net, samples, answers, error_margin, cohort, random=False, num_iter=10000):
    errors = 5
    iter = 0
    max_change = 10
    
    net.evolve(force=True) 
    print([l.noutputs for l in net.hidden_layers])
    #input()
    while np.sum(np.abs(errors)) >= error_margin and iter<num_iter:
        e = 0
        errors = []
        changes = []
        for j in range(cohort):
            if random:
                i = np.random.choice(len(samples)) 
            else:
                i=j
            result = net.activate(samples[i])

            answer = answers[i]

            error = answer - result
        
            errors.append(error) 

            net.delta(error)
            net.adjust() 

        if (iter%1000)==0:
            print("Iteration: ", iter) 

        iter += 1

    return np.sum(np.abs(errors)), iter  


