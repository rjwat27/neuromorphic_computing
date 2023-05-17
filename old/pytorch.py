from difflib import get_close_matches
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from time import sleep 

#import cryptoVision_Top as ct 
import market_forecast.DataModule as dm 


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DataBase = dm.DataBase('BTCUSD', "tick_data_BTCUSD.txt")
Watcher = dm.Watcher()  
desired_keys =  ['lastPrice', "highPrice", "lowPrice", "priceChangePercent", "volume"]
depth = 10

import VirtualMarketModule as vm 

market = vm.Market() 

assets = 100 

wait = 15


# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def prep(depth=20):
    print('Prepping...') 
    for i in range(depth):
        data = Watcher.full_data('BTCUSD') 
        #Market.price_log() 
        DataBase.log(data) 
        sleep(wait) 

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)




class DQN(nn.Module):

    def __init__(self, inputs, outputs, hidden):
        super(DQN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
        nn.Linear(inputs, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, outputs) ) 

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        # linear_input_size = convw * convh * 32
        # self.head = nn.Linear(linear_input_size, outputs)




    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = x.double() 
        #x.flatten()
        #x = self.flatten(x)
        x.resize_(depth*size)
        x.unsqueeze(dim=0) 
        print('\n\nSTATE: ', x, '\n\n')
        #x = torch.tensor(x, dtype=torch.float32) 
        #print(np.shape(x))
        logits = self.linear_relu_stack(x)
        return logits
        return self.head(x.view(x.size(0), -1))


# resize = T.Compose([T.ToPILImage(),
#                     T.Resize(40, interpolation=Image.CUBIC),
#                     T.ToTensor()])

import VisualizeModule as vis
def get_market_data(depth):
    d = Watcher.full_data()

    DataBase.log(d) 

    #market.price_log(float(d['lastPrice'])) 

    data = [DataBase.live_data[int(i+1)] for i in range(DataBase.size - depth, DataBase.size)]
    
    # data = DataBase.raw_strip(data, desired_keys) 
    data = DataBase.strip(data, desired_keys)

    data =  torch.from_numpy(data) 

    #data = torch.tensor(data, dtype=torch.double) 

    data.double() 

    return data




BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


prep(depth=depth) 
'''buy sell or do nothing'''

n_actions = 3#env.action_space.n  

init_market_state = get_market_data(depth) 

size = len(desired_keys) 

#init_market_state.resize_(depth, size)

policy_net = DQN(depth*size, n_actions, 32).to(device)
policy_net.double()
target_net = DQN(depth*size, n_actions, 32).to(device)
target_net.double() 
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    # print('\n\nSTATE: ', state, '\n\n') 
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state.flatten()).max(0)
    else:
        print('ya i went random') 
        return torch.tensor(random.randrange(n_actions), device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def get_state():
    market_state = get_market_data(depth)
    #market_state.resize_(depth, size)
    return market_state

import time

num_episodes = 300
for i_episode in range(num_episodes):

    market_state = get_market_data(depth)

    #market_state.resize_(depth, size)
    state = market_state
    #state = torch.tensor(state, dtype=torch.double) 
    time0 = time.time() 
    for t in count():
        
        d = Watcher.full_data()

        market.price_log(float(d['lastPrice'])) 

        # Select and perform an action
        action = select_action(state)
        print('TEST: ', action)

        print('current price: ', market.current_price)
        market.decide(action) 
        market.stat() 
        print('Assets: ', market.assets)
        reward = market.assets - assets 
        assets = market.assets 

        reward = torch.tensor(reward, device=device)
        reward.double()

        # Observe new state
        #last_screen = current_screen
        #next_state = get_state()
        #state = torch.tensor(state, dtype=torch.double) 
        # if not done:
        #     next_state = current_screen - last_screen
        # else:
        #     next_state = None

        # Store the transition in memory
        #memory.push(state, action, next_state, reward)

        # Move to the next state
        # state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        vis.decision_graph(market.price_history, market.decisions)
        
        time2 = time.time()
        if time2-time0 < wait:
            sleep(wait - (time2-time0)) 

        time0 = time.time() 

        next_state = get_state()
        # Store the transition in memory
        memory.push(state, torch.tensor(action), next_state, reward)

        state = next_state

        if reward < 0:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())





