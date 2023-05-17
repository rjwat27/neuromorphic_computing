import os
import numpy as np

# Path to folder containing CSV files
folder_path = './dataset'

# Loop through all files in folder  -   Thanks chatGPT!
arrays_from_csv = []
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        # Load CSV file into numpy array
        csv_path = os.path.join(folder_path, file_name)
        csv_array = np.loadtxt(csv_path, delimiter=',')


        arrays_from_csv.append(csv_array) 

#extract the final array and make the target array for training

targets = arrays_from_csv[-1]
arrays_from_csv.pop(-1) 

samples = arrays_from_csv 

#pytorch spiking neural network
print('begin') 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class SpikingReLU(nn.Module):
    def __init__(self, threshold=0.5, reset=0.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.reset = nn.Parameter(torch.tensor(reset))

    def forward(self, x):
        spikes = torch.zeros_like(x)
        spikes[x >= self.threshold] = 1.0
        output = spikes * (x - self.threshold)
        output[x < self.threshold] = self.reset
        return output

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input[self.threshold <= 0] = 0
        return grad_input, None


# Define the SNN model
class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
    
        self.lin1 = nn.Linear(3, 3, bias=False)
        self.spike1 = SpikingReLU()

        self.lin2 = nn.Linear(3, 3, bias=False)
        self.spike2 = SpikingReLU()

        self.lin3 = nn.Linear(3, 3, bias=False)
        self.spike3 = SpikingReLU()

        self.lin4 = nn.Linear(3, 3, bias=False)
        self.spike4 = SpikingReLU()

    def forward(self, x):
        x = self.lin1(x)
        x = self.spike1(x)
        x = self.lin2(x)
        x = self.spike2(x)
        x = self.lin3(x)
        x = self.spike3(x)
        x = self.lin4(x)
        x = self.spike4(x)
        return x

class SNN_Layer(nn.Module):
    def __init__(self, num_modules):
        super(SNN_Layer, self).__init__()
        self.size = num_modules 
        self.output_size = self.size * 3 
        self.neurons = nn.ModuleList([SNN() for i in range(self.size)])


    def forward(self, x):
        x = x.unfold(0, 3, 1)
        out = []
        for i in range(x.shape[0]):
            chunk = x[i]
            y = self.neurons[i].forward(chunk) 
            out.append(y) 

        # Concatenate the outputs
        result = torch.cat(out, dim=0)

        return result

class SNN_Super(nn.Module): #this one is mine
    def __init__(self, ninputs, noutputs, hidden_layer_sizes):
        super(SNN_Super, self).__init__() 
        self.ninputs = ninputs 
        self.noutputs = noutputs 
        self.hidden_layer_sizes = hidden_layer_sizes

        self.bias = nn.Parameter(torch.randn(3))

        self.input_layer = SNN_Layer(max(1, self.ninputs-2)) 
        size = self.input_layer.output_size

        self.hidden_layers = []
        for i in range(len(self.hidden_layer_sizes)):
            next = SNN_Layer(max(1, size-2))
            self.hidden_layers.append(next)
            size = next.output_size 

        self.output_layer = SNN_Layer(1) #for now end with a single chip

        #self.layers = [self.input_layer] + self.hidden_layers 

        self.layers = nn.ModuleList([self.input_layer] + self.hidden_layers)


    def forward(self, x):
       
        for l in self.layers:
            x = l(x) 
       
        x = x.unfold(0, 3, 1)
        x = torch.sum(x, dim=1) #focus all inputs into single chip...might destroy all information *shrug* 
        x = self.output_layer(x)

        return x 



def get_max_vector_size(vectors):
    max_size = 0
    for vec in vectors:
        vec_size = vec.size(0)
        if vec_size > max_size:
            max_size = vec_size
    return max_size

# Define the training function
def train(model, samples, targets, num_epochs=100, learning_rate=0.0001):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i in range(len(samples)):
            target = targets[i]
            for time_step in samples[i]:
                optimizer.zero_grad()
                outputs = model(time_step)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

        if epoch % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))





#convert numpy array data into pytorch data
temp = []
for a in samples:
    new = torch.from_numpy(a).float()
    shape = new.shape 
 
    temp.append(new)

max_size = get_max_vector_size(temp)
temp2 = []
for a in temp:
    new = F.pad(a, (0, 0, 0, max_size-a.shape[0]), 'constant', 0).float()
    temp2.append(new)

inputs = torch.stack(temp2, dim=0) 

#refit targets to appropiate output size
new_targets = []
targets = targets[:,1]

for t in targets:
    if t==1:
        new_targets.append(torch.Tensor([1, 0, 0]).float())
    elif t==-1:
        new_targets.append(torch.Tensor([0, 1, 0]).float())
    else:
        print('nonsense')
        new_targets.append(torch.Tensor([0, 0, 0]).float()) 

targets = torch.stack(new_targets, dim=0) 

model = SNN_Super(4, 2, [30, 10]) #layers sizes not used yet

print('begin training')
train(model, samples=inputs, targets=targets)




