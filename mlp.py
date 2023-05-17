import numpy as np

def relu(x, bias):
    return np.maximum(0, np.arctan(x+bias)) 
def sig(x, bias):
    return relu(x, bias) 
    return 1/(1+np.exp(-(x+bias))) 
def sig_der(x, bias):
    return 1 if x > 0 else .1#(1/(1+np.exp(-x)))*(1-1/(1+np.exp(-x))) 
    return sig(x, bias)*(1-sig(x, bias)) 

min_bias = .18
max_bias = 1

def spiking_activation(x, bias):
    return np.maximum(0, np.multiply(x, bias)) 
def spiking_der(x, bias):
    return bias if x > 0 else .1



class single_layer():
    '''simple one hidden layer network'''
    def __init__(self, inputs, outputs, learning_rate=.1, fanout=0, activation_function=spiking_activation, ordered=False):
        self.ninputs = inputs
        self.noutputs = outputs
        
        self.fanout = fanout 

        self.activation_function = activation_function
        self.act_der = spiking_der 

        self.input = 0
        self.input1 = 0
        
        self.bias_b1 = -inputs/2
        self.bias_b2 = inputs/2
        self.biases1 = np.array([-inputs/2 + i*inputs/outputs for i in range(self.noutputs)])
        



        self.delta1 = np.zeros(self.noutputs)

        self.max_change = 0 
        #TODO for many layers may be wise and even very effective to test convergence on individual 
        #layers and update them individually on a rollover signal from the following layer-
        #for now just a single max_change for global convergence, though this may drastically increase learning times as global convergence
        #on large mult-layer networks will take a while

        self.learning_rate = learning_rate


        self.weight_low = -inputs   #maybe destructive, who knows?
        self.weight_high = inputs 
        self.w1 = np.array([[np.random.rand()*2-1 for i in range(self.noutputs)] for j in range(self.ninputs)])
       


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
            #print(self.input1) 
        
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

    def duplicate(self, num=1):
        b_dupe = self.biases1.copy()
        new_outputs = len(b_dupe) 
        d_dupe = self.delta1.copy()
        w_dupe = self.w1.copy() 
        for g in range(num):
            #duplicate biases
            self.biases1 = np.append(self.biases1, b_dupe, axis=0) 
            #duplicate delta 
            self.delta1 = np.append(self.delta1, d_dupe, axis=0) 
            self.noutputs += new_outputs  

            #duplicate weights
            self.w1 = np.append(self.w1, w_dupe, axis=1) 

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

    def normalize_weights(self, bias_scale=False):
        '''scale weight values between -1 and 1'''
        W = self.w1.T
        new_W = np.zeros((self.noutputs, self.ninputs)) 
        big=1
        for i in range(self.noutputs):
            w = W[i]
            big = max(np.abs(w))
            #norm = np.linalg.norm(w) 
            #big=norm 
            #big = 2
            #input(big)
            new_w = w / big 
            new_W[i] = new_w 
            if bias_scale:
                self.biases1[i] *= big
        self.w1 = new_W.T  
   
        return big


    def add_weights(self):
        new_weights = np.array([np.random.rand()*2 for i in range(self.noutputs)])

        self.w1 = np.append(self.w1, np.array([new_weights]), axis=0)

        self.ninputs += 1

    def dup_weights(self, num=1):
        new_inputs = self.ninputs 
        w_dupe = self.w1.copy() 
        for g in range(num):
            self.ninputs += new_inputs  
            #duplicate weights
            self.w1 = np.append(self.w1, w_dupe, axis=1) 
 

    def vectorize_node_weights(self, node):
        vector = np.array([self.w1[i, node] for i in range(self.ninputs)]) 
        return (vector / np.linalg.norm(vector)) #normalize 

    def is_converged(self):
        return (self.max_change < .01) 


class mlp():
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

        self.output_amplification=1

        if not layer_sizes:
            print("ya yer gonna need some data on the layer sizes") 
            return 
        else:
            self.hidden_layers = [single_layer(self.layer_sizes[i], self.layer_sizes[i+1], self.learning_rate, self.fanout, ordered=ordered) for i in range(self.layers)]

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
           # input()
            input1 = l.activate(input1)
            #print(input1)
       # input(input1) 
        return input1*self.output_amplification  
        
    def set_output_amplification(self, normalizing_factors): 
        self.output_amplification = np.product(normalizing_factors)

    def delta(self, feedback):
        for l in range(1, self.layers+1):
            self.hidden_layers[-l].delta(feedback)
            feedback = self.hidden_layers[-l].backpropogate() 
            # print(feedback) 
            # input()

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
               #print('pre test: ', self.hidden_layers[l-1].noutputs, len(self.hidden_layers[l-1].biases1), n)
                if n!=None:
                    self.hidden_layers[l].prune_weights(n) 
                
                if self.growth_flag and (l in self.layers_to_grow) and len(self.hidden_layers[l].biases1) < self.max_layer_size:
                    #print('got here ', l)
                    self.hidden_layers[l-1].add_hidden_node(3)    #3 arbitrario  
                    for i in range(3):
                        self.hidden_layers[l].add_weights()           
                elif n!=None:
                   #print('TEST: ', self.hidden_layers[l-1].noutputs, len(self.hidden_layers[l-1].biases1), n)
                    self.hidden_layers[l-1].add_hidden_node(len(n))
                    for i in n:
                        self.hidden_layers[l].add_weights()
        self.update_layer_sizes()

    def duplicate(self, num=1):
        for i in range(len(self.hidden_layers)-1):
            self.hidden_layers[i].duplicate(num=num)
            self.hidden_layers[i+1].dup_weights(num=num) 

    def normalize_weights(self, bias_scale=False, duplicate=False):
        big_factors = []
        for i in range(1, len(self.hidden_layers)):
            big = self.hidden_layers[i].normalize_weights(bias_scale=bias_scale)
            big_factors.append(big) 
            if duplicate:
                self.hidden_layers[i-1].duplicate(num=int(big)) 
                self.hidden_layers[i].dup_weights(num=int(big)) 
       # self.hidden_layers[-1].normalize_weights(bias_scale=bias_scale) 
        return big_factors 

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


'''standard epoch run using my mpl models'''
def run_learn_cycle(net, samples, answers, error_margin, cohort, random=False, num_iter=10000):
    errors = 5
    iter = 0
    max_change = 10
    
    net.evolve(force=True) 

    samples_to_learn = []
    answers_to_learn = []
    for j in range(cohort):
        if random:
            i = np.random.choice(len(samples)) 
            samples_to_learn.append(samples[i])
            answers_to_learn.append(answers[i]) 
        else:   #this aint good
            i=j
            samples_to_learn.append(samples[i]) 
            answers_to_learn.append(answers[i]) 

    while np.sum(np.abs(errors)) >= error_margin and iter<num_iter:
        e = 0
        errors = []
        changes = []

        for i in range(cohort):
            result = net.activate(samples_to_learn[i])

            answer = answers_to_learn[i]

            error = answer - result
       
            errors.append(error) 

            net.delta(error)
            net.adjust() 

        if (iter%1000)==0:
            print("Iteration: ", iter) 


        iter += 1

    return np.sum(np.abs(errors)), iter  

'''distillation method'''

def distill(net, target_layer_sizes, samples, answers, error_margin, cohort, debug_print=True):

    net.growth_flag = False 

    distill_limit_met = False 

    size_results = []

    for l in [1]:#range(len(net.hidden_layers)-1):
        net.set_layers_to_adjust([2])  #only one ? TODO 
        while (net.hidden_layers[l].noutputs > target_layer_sizes[l]) and not distill_limit_met:
            net.distill(l, net.hidden_layers[l].noutputs-1) 
 
            error, iter = run_learn_cycle(net, samples, answers, error_margin, cohort)

            #if error too large, assume complexity too great for further distillation in layer
            if error > error_margin:
                distill_limit_met = True 

            if debug_print:
                print('layer size: ', net.hidden_layers[l].noutputs) 

        distill_limit_met = False 

        size_results.append(net.hidden_layers[l].noutputs) 

        if debug_print:
            input("next_layer...") 

    # for l in [0, 1, 2]:
    #     net.set_layers_to_adjust([l]) 
    #     error, iter = run_learn_cycle(net, samples, answers, error_margin, cohort)


    return net, size_results
    

'''matrix transform stuf'''








