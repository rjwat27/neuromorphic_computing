
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt 




spike_width = 50e-9 #seconds
#vref = 150 #mV
v_change = 100 #mV
base_cap = 50 #picoFarad 

X = torch.Tensor([0.0525,0.07,0.0875,0.105,0.1225,0.14,0.1575,0.175,0.1925,0.21,0.2275,0.245,0.2625,0.28,0.2975,0.315,0.3325,0.35])
Y = torch.Tensor([644404.0862,761653.8961,939789.2253,1178813.607,1498410.95,1937406.885,2546507.201,3374555.172,4487038.625,5974375.188,7940600.158,10476316.7,13607669.83,17208745.11,20945786.46,24370070.17,27157403.26,29234740.57])
vco_curve = torch.Tensor([[X[i], Y[i]] for i in range(len(X))])

acceptable_weights = torch.Tensor([-0.9375, -0.875, -0.8125, -0.75, -0.6875,
                                    -0.625, -0.5625, -0.5, -0.4375, -0.375,
                                    -0.3125, -0.25, -0.1875, -0.125, -0.0625, 
                                   0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 
                                   0.375, 0.4375, 0.5, 0.5625, 0.625, 
                                   0.6875, 0.75, 0.8125, 0.875, 0.9375])

base_capacitance = 10e-9
npn_capacitance = 50e-15
energy_per_spike = .00000005
spike_amplitude = energy_per_spike / spike_width 
vcoChange_per_voltageInput_perSecond = 1 

decay_rate = .1

time_step = 1e-9 #seconds


cap_bank = np.array([base_capacitance*i for i in [.25, .5, 1, 2]]) 



def linear_interpolator(x, x_values=vco_curve[:,0], y_values=vco_curve[:,1]):

    # Find the index of the x value that is closest to the given x
    temp = x.reshape(-1, 1)
    size = x_values.size()
    a = (torch.abs(x_values - temp))
    index = torch.argmin(a, dim=1)
    #input(index)
    # Check if the given x value is within the range of x values
    # if x < x_values[0] or x > x_values[-1]:
    #     raise ValueError('Given x value is outside the range of x values.')
    
    # Check if the given x value matches an existing x value
    # if x.any() == x_values[index]:
    #     return y_values[index]
    
    # Calculate the slope between the two nearest y values
    slope = (y_values[index] - y_values[index-1]) / (x_values[index] - x_values[index-1])
    
    # Calculate the y value at the given x using the slope and the nearest y value
    #print(x.size(), x_values[index-1].size(), y_values[index-1].size())
    y = slope * (x - x_values[index-1]) + y_values[index-1]
    
    return y, slope 

def weight_to_bits(weight):
    sign = 1 if weight<0 else 0
    w = abs(weight) 
    temp = [0, 0, 0, 0, 0] 
    temp[4] = 1 if w>.5 else 0 
    temp[3] = 1 if w>(.5*temp[4] + .25) else 0 
    temp[2] = 1 if w>(.5*temp[4] + temp[3]*.25 + .125) else 0 
    temp[1] = 1 if w>(.5*temp[4] + temp[3]*.25 + temp[2]*.125 + .0625) else 0
    temp[0] = sign 
    return temp 
    

class Lin_Int(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Implement the forward pass of the function
        output, der = linear_interpolator(input) 
        # Save the input tensor for the backward pass
        ctx.save_for_backward(der)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        #print(';alskdjflkds') 
        # Retrieve the input tensor
        der, = ctx.saved_tensors
        # Compute the gradient of the input tensor using the chain rule
        grad_input = grad_output
        return grad_input * der 
    
volt_to_freq = Lin_Int.apply
    
class Spike_Generator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.where(input>=torch.pi, spike_amplitude, 0)
        ctx.save_for_backward(output)
        return output 
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        return torch.where(output>0, grad_output*spike_amplitude, grad_output*.1) 

class PDNNeuron_Layer(nn.Module):
    def __init__(self, output_size=3, decay=0.1):
        super().__init__()
        self.output_size = output_size
        self.freq_finder = Lin_Int.apply
        self.vref = nn.Parameter(.15*torch.ones(self.output_size)) 
        self.vco_bias = nn.Parameter(.15*torch.ones(self.output_size))
        self.local_freq = self.freq_finder(self.vref)
        self.phase = torch.zeros(self.output_size)
        self.freq = self.freq_finder(self.vco_bias)
        self.decay = decay
        self.output_val = torch.zeros(self.output_size) 

        self.spike_flag = torch.ones(self.output_size)#change back to zeros
        self.spike_time = torch.zeros(self.output_size)

        self.spike_generator = Spike_Generator.apply

        self.time = 0 #seconds

        #debug streams
        self.output_stream = []
        self.vco_stream = []
        self.freq_stream = []
        self.phase_stream = []
    
    def forward(self, x):
        
        # Integrate input and membrane potential
        #TODO re-incorporate time step into vco change
        #self.vco_bias = torch.where(self.spike_flag==1, self.vco_bias * self.decay + x*vcoChange_per_voltageInput_perSecond, torch.zeros(self.output_size))
        self.vco_bias = nn.Parameter(torch.relu(self.vco_bias + x*vcoChange_per_voltageInput_perSecond - self.vco_bias * self.decay))
        self.vco_stream.append(self.vco_bias.detach().clone().numpy()) 
        # get frequencies and phases
        self.freq = self.freq_finder(self.vco_bias)#[0]
        self.freq_stream.append(self.freq.detach().clone().numpy())
        self.local_freq = self.freq_finder(self.vref) 
        #self.phase = torch.abs(self.local_freq-self.freq)*2*torch.pi*(self.time%(1/self.local_freq + 1/self.freq))
        #self.spike_flag = torch.where(self.phase>=torch.pi, 1, 0) 
        #self.output_val = self.spike_flag*self.vref*spike_amplitude
        #print(spike_amplitude) 
      
        #self.time += time_step 
        #self.spike_time = self.spike_time + torch.mul(self.spike_flag, time_step)
        #print(self.spike_time, torch.mul(self.spike_flag, time_step))
        #self.spike_flag = torch.where(self.spike_time>=spike_width, 0, 1) 
        #self.spike_time = torch.where(self.spike_time>=spike_width, 0, self.spike_time) 
        # Return output
        phase = (self.freq-self.local_freq)*2*torch.pi*(1/self.local_freq+1/self.freq)
        self.phase_stream.append(phase.detach().numpy())
        # self.spike_flag = 1 if (phase>=torch.pi and not self.spike_flag) else (1 if (self.spike_flag and self.spike_time<spike_width) else 0) 
        # self.spike_time
        #return self.spike_generator(self.vref, self.local_freq)
        self.vco_bias = nn.Parameter(torch.where((self.freq-self.local_freq)*2*torch.pi*(1/self.local_freq+1/self.freq)>=torch.pi, 0, self.vco_bias))
        # print((self.freq-self.local_freq)*2*torch.pi*(1/self.local_freq+1/self.freq))
        # input()
        output = self.spike_generator(phase)
        self.output_stream.append(output.detach().clone().numpy())#output.detach().numpy())
        return output 


    def generate_weight_bistream(self):

        pass 
    # @staticmethod 
    # def backward(self, grad_output):
      
    #     return 1*grad_output if self.output_val > 0 else .1*grad_output




class Chip(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_size = 3
        self.hidden_size = 3
        self.output_size = 1

        # Define spiking neuron layers
        self.input_layer = PDNNeuron_Layer()
        self.hidden_layer1 = PDNNeuron_Layer()
        self.hidden_layer2 = PDNNeuron_Layer()
        self.output_layer = PDNNeuron_Layer(output_size=1)

        # Define linear weights for each layer
        self.input_weights = nn.Linear(self.input_size, self.hidden_size, bias=False) 
        self.hidden_weights1 = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.hidden_weights2 = nn.Linear(self.input_size, self.output_size, bias=False)

    def forward(self, x):
        # Propagate input through spiking neuron layers
        x = self.input_layer(x)
        x = self.input_weights(x)
        x = self.hidden_layer1(x)
        x = self.hidden_weights1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_weights2(x)
        x = self.output_layer(x)
        #x = torch.relu(x) 
        return x
    
    def generate_bitstream(self):

        pass 


def compute_chip_weights(model, inputs, targets, num_epochs=100, sample_exposure=10):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # Train the model
    for epoch in range(num_epochs):
        # Forward pass
        #y_pred = []
        for i in range(len(inputs)):
            for j in range(sample_exposure):
                #print(j)
                l = inputs[i] 
                t = targets[i]
                #y_pred.append(model(l)) 
                y_pred = model(l) 

                #y_pred = model(inputs) 

                #y_pred = torch.stack(y_pred, dim=0) 
                #input(y_pred)

                # Compute the loss
                loss = nn.MSELoss()(y_pred, t)

                # Backward pass and optimization step
                optimizer.zero_grad()
                #torch.autograd.set_detect_anomaly(True)
                loss.backward(retain_graph=False)
                #print('completed backward')
                optimizer.step()

        # for name, param in model.named_parameters():
        #     if "vref" in name:
        #         param.data = torch.clamp(param.data, min=0, max=.35)

        # Print the loss every 1000 epochs, adjust weights to legal values
        if epoch % 10 == 0:
            # for name, param in model.named_parameters():
            #     if "weight" in name:
            #         # Use torch.clamp to limit the tensor values to the discrete set of values
            #         clamped_weight = torch.clamp(param.data, min=acceptable_weights.min().item(), max=acceptable_weights.max().item())
            #         clamped_weight = acceptable_weights[(clamped_weight - acceptable_weights[0]).abs().argmin()]
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Test the model
    with torch.no_grad():
        test_inputs = inputs
        test_targets = targets
        test_outputs= []
        for l in inputs:
            test_outputs.append(model(l)) 
        test_outputs = torch.stack(test_outputs, dim=0) 
        test_loss = nn.MSELoss()(test_outputs, test_targets)
        print(f"Test Loss: {test_loss.item()}, Test Outputs: {test_outputs}, Test Targets: {test_targets}")

    return test_loss, model 




# Example usage
# x = torch.ones(3)
# test =Chip() 
# boo = PDNNeuron_Layer(output_size=1)

# inputs = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
# targets = torch.tensor([[0.0], [1.0], [1.0], [0.0], [1.0], [0.0], [0.0], [1.0]])

# #input(boo(x))
# # for p in boo.named_parameters():
# #     print(p) 
# # input()
# compute_chip_weights(test, inputs, targets)

# fig, axes = plt.subplots(1, 3)

# s = test.output_layer.vco_stream
# n = test.output_layer.output_stream
# f = test.output_layer.freq_stream 
# p = test.output_layer.phase_stream
# # input(n)

# x = range(len(n))
# axes[0].plot(x, s, 'blue')
# axes[1].plot(x, f, 'green') 
# axes[2].plot(x, p, 'magenta') 


# plt.show()












