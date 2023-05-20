import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 

from matplotlib import pyplot as plt

# Define the 3-input XOR function
def xor(x):
    return torch.logical_xor(torch.logical_xor(x[0], x[1]), x[2]).float()

def map_to_nearest_4(val):
    sign = -1 if val<0 else 1 
    # Convert the value to a fixed-point unsigned integer with 4 fractional bits
    fixed_val = round(val * 15)
    
    # Map the fixed-point value to the nearest multiple of 5
    nearest_5 = 5 * round(fixed_val / 4)
    
    # Convert the nearest multiple of 5 back to a floating-point value
    nearest_5_val = nearest_5 / 31.0
    
    return nearest_5_val

def weight_to_bits(weight, reverse=False):
    sign = 1 if weight<0 else 0
    w = abs(weight) 
    temp = [0, 0, 0, 0, 0] 
    temp[4] = 1 if w>.5 else 0 
    temp[3] = 1 if w>(.5*temp[4] + .25) else 0 
    temp[2] = 1 if w>(.5*temp[4] + temp[3]*.25 + .125) else 0 
    temp[1] = 1 if w>(.5*temp[4] + temp[3]*.25 + temp[2]*.125 + .0625) else 0
    temp[0] = sign 
    if reverse:
        temp.reverse() 
    
    return temp 
    

class MultiplicativeBias(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(hidden))
    
        
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.bias, 0, 1) 

    def forward(self, x):
        return torch.arctan(x * self.bias.unsqueeze(0)) 
        #return x * self.bias.unsqueeze(0)

class Chip(nn.Module):
    def __init__(self):
        super(Chip, self).__init__()
        self.fc0 = nn.Linear(3, 3, bias=False) 
        self.fc1 = nn.Linear(3, 3, bias=False)
        self.fc2 = nn.Linear(3, 3, bias=False)
    
        self.relu0 = torch.nn.LeakyReLU(negative_slope=-.1) 
        self.relu1 = torch.nn.LeakyReLU(negative_slope=-.1) 
        self.relu2 = torch.nn.LeakyReLU(negative_slope=-.1) 
        self.relu3 = torch.nn.LeakyReLU(negative_slope=-.1) 

        self.bias0 = MultiplicativeBias(3) 
        self.bias1 = MultiplicativeBias(3) 
        self.bias2 = MultiplicativeBias(3)
        self.bias3 = MultiplicativeBias(3)

    def forward(self, x):
        x = self.bias0(x)
        x = self.relu0(x) 
        x = self.fc0(x)
        x = self.bias1(x)
        x = self.relu1(x)
        x = self.fc1(x)
        x = self.bias2(x)
        x = self.relu2(x)
        x = self.fc2(x)
        x = self.bias3(x)
        x = self.relu3(x) 
        #x = torch.sigmoid(x) 

        return x
    

    #generate bitstream of weights->csv
    def generate_bitstream(self, destination_file):
        weights = pytorch_params_to_numpy(self)[0:3]
        w0 = weights[0]
        w1 = weights[1]
        w2 = weights[2]

        W0 = []
        for vector in w0:
            W0 += weight_to_bits(vector[0])
            W0 += weight_to_bits(vector[1])
            W0 += weight_to_bits(vector[2])
        

        W1 = []
        for vector in w1:
            W1 += weight_to_bits(vector[0])
            W1 += weight_to_bits(vector[1])
            W1 += weight_to_bits(vector[2])
        W1.reverse()

        W2 = []
        for vector in w2:
            W2 += weight_to_bits(vector[0])
            W2 += weight_to_bits(vector[1])
            W2 += weight_to_bits(vector[2])
       

        W = []
        W += W0 + W1 + W2 
        W = np.array(W, dtype=int) 
        np.savetxt(destination_file, W)
        #print(len(W))
        #return W 
    


def learn(inputs, targets, num_epochs=int(10e3), clipping='during', quantized=False, graph=False):

    #resize inputs if necessary
    num_inputs = len(inputs)
    temp = len(inputs[0])
    tail = torch.zeros(num_inputs, 3-temp) 
    if temp != 0:
       inputs = torch.concat((inputs, tail), dim=1)


    #resize targets if necessary
    num_targets = len(targets)
    temp = len(targets[0])
    tail = torch.zeros(num_targets, 3-temp) 
    if temp != 0:
       targets = torch.concat((targets, tail), dim=1)

    # Initialize the model and the optimizer
    model = Chip()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    graph_stuff = []    #for post-learning analysis
    for epoch in range(num_epochs):
        # Forward pass
        y_pred = model(inputs)

        # Compute the loss
        loss = nn.MSELoss()(y_pred, targets)
        graph_stuff.append(loss.clone().detach().numpy()) 

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if clipping=='during':
            for name, param in model.named_parameters():
                if "weight" in name:
                    param.data = torch.clamp(param.data, -1, 1)
                    if quantized:
                        param.data = (torch.round(param.data*15) / 16)
                if "bias" in name:
                    param.data = torch.clamp(param.data, min=.13, max=.9)

        # Print the loss every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            # if clipping=='during':
            #     for name, param in model.named_parameters():
            #         if "weight" in name:
            #             param.data = torch.clamp(param.data, -1, 1)
            #             if quantized:
            #                 param.data = (torch.round(param.data*15) / 16)
            #         if "bias" in name:
            #             param.data = torch.clamp(param.data, min=.13, max=.9)


    if clipping=='after':
        for name, param in model.named_parameters():
            if "weight" in name:
                param.data = torch.clamp(param.data, -1, 1)
                if quantized:
                    param.data = (torch.round(param.data*15) / 16)
            if "bias" in name:
                param.data = torch.clamp(param.data, min=0, max=1)

    # Test the model
    with torch.no_grad():
        test_inputs = inputs
        test_targets = targets
        test_outputs = model(test_inputs)
        test_loss = nn.MSELoss()(test_outputs, test_targets)
        print(f"Test Loss: {test_loss.item()}, Test Outputs: {test_outputs}, Test Targets: {test_targets}")

    if graph:
        x = range(len(graph_stuff))
        plt.plot(x, graph_stuff)
        plt.title('Rounding to nearest quantized value every update')
        plt.show() 

    return test_loss, model 


def pytorch_params_to_numpy(model):
    '''bias vectors are listed first, follewed by weights matrices'''
    params = list(model.parameters())
    return [p.detach().numpy().astype(object) for p in params]
  

def lin_interpolator(points, upscale_factor=10, der=0):
    X = points[0]
    Y = points[1]
    new_xpoints = []
    new_ypoints = []
    for i in range(len(X)-2):
        set = [] 
        x0 = X[i]
        x1 = X[i+1] 
        y0 = Y[i]
        y1 = Y[i+1]
        new_xpoints += list(np.linspace(x0, x1, upscale_factor))
        new_ypoints += list(np.linspace(y0, y1, upscale_factor))
    
    return new_xpoints, new_ypoints


def get_max_vector_size(vectors):
    max_size = 0
    for vec in vectors:
        vec_size = vec.size(0)
        if vec_size > max_size:
            max_size = vec_size
    return max_size

def complex_power(signals, ref):
    ref = signals[0]
    ref_norm = np.linalg.norm(ref)
    real = [ref_norm]
    imag = [0]
    for i in range(1, len(signals)):
        sig_norm = np.linalg.norm(signals[i])
        product = np.dot(signals[i], ref) / ref_norm
        real.append(product) 
        b = np.sqrt(sig_norm**2 - product**2) 
        imag.append(b) 
    complex_tensor = torch.complex(real, imag)
    return complex_tensor 

def prepare_data(data): #make sure data is all of same length
    max_size = get_max_vector_size(data)
    temp2 = []
    for a in data:
        new = F.pad(a, (0, 0, 0, max_size-a.shape[0]), 'constant', 0).float()
        temp2.append(new)
    result = torch.stack(temp2)
    return result 






