import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np 

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
    
    def set_data(self, data, targets):
        self.data = list(zip(data, targets))
    





def train_model_on_data(model, data, targets):
    #make sure data and targets are pytorch tensors
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data) 
    if not isinstance(targets, torch.Tensor):
        targets = torch.from_numpy(targets) 

    #dataset = MyDataset(list(zip(data, targets)))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    #set limits on weights and biases, to be compatible for the pdn network
    # weight_clip_value = 1
    # bias_top_clip_value = 1
    # bias_bottom_clip_value = 0
    # for name, param in model.named_parameters():
    #     if "bias" in name:
    #         param.register_hook(lambda grad: torch.clamp(grad, bias_bottom_clip_value, bias_top_clip_value))
    #     elif "weight" in name:
    #         param.register_hook(lambda grad: torch.clamp(grad, -weight_clip_value, weight_clip_value))

  
    num_epochs = 10000
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i in range(len(data)):
            # Get a batch of data
            inputs = data[i]
            target = targets[i]
            
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            # print(outputs)
            # print(target)
            # input()
            # Compute the loss
            loss = criterion(outputs, target.unsqueeze(dim=0).unsqueeze(dim=1).float())

            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

        # Compute the average loss for the epoch
        avg_loss = running_loss / len(data)

        if avg_loss < .01:
            break 

        if (epoch%1000==0):
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


    return model 
