import numpy as np

from matplotlib import pyplot as plt


base_capacitance = 10e-9
npn_capacitance = 50e-9#-15
energy_per_spike = 1
vup_down_multiplier = 1

cap_bank = np.array([base_capacitance*i for i in [2, 1, .5, .25]]) 
#print(np.sum(cap_bank))

def sig(x):
    return 1/(1+np.exp(-x)) 

def freq_from_v(self, v):
    v2 = np.ones(len(self.vco_curve))*v 
    idx = np.argmin(abs(self.vco_curve[:,0]*1000-v2))
    # print(abs(self.vco_curve[:,0]*1000-v2))
    # input()

    '''test with linear curve'''
    b = self.vco_curve[0][1]
    e = self.vco_curve[-1][1]
    #return (b + (e-b)*(v/np.shape(self.vco_curve)[0]))
    return self.vco_curve[idx][1] 

def simple_weight_transform(weight):
    '''weight value ought to be between -1 and 1'''
    sign = 1 if weight >= 0 else 0
    new_w = int(abs(weight) * 15 )
    binary = bin(new_w)[2:]
    #zero pad 'binary'
    if len(binary) < 4:
        pad = ['0' for i in range(4-len(binary))]
        binary = ''.join(pad) + binary 
    w_bar = np.zeros(5)
    w_bar[0] = sign 
    w_bar[1] = int(binary[0])
    w_bar[2] = int(binary[1])
    w_bar[3] = int(binary[2])
    w_bar[4] = int(binary[3])

    return w_bar 

def bits_to_weights(weight_bits):
    temp = weight_bits[1::] 
    input_cap = np.sum(np.multiply(cap_bank, temp)) 
    return np.sum(input_cap/(input_cap+npn_capacitance)) if weight_bits[0]==1 else -np.sum(input_cap/(input_cap+npn_capacitance))
    
    
def weight_transform(weight):
    sign = 1 if weight >= 0 else -1
    w = abs(weight) 
    if w < .9:
        val = int(15 * npn_capacitance / (1 - w/vup_down_multiplier))
    else:
        val = 15 
    binary = bin(val)[2:]
    if len(binary) < 4:
        pad = ['0' for i in range(4-len(binary))]
        binary = ''.join(pad) + binary 
    w_bar = np.zeros(5)
    w_bar[0] = sign 
    w_bar[1] = int(binary[0])
    w_bar[2] = int(binary[1])
    w_bar[3] = int(binary[2])
    w_bar[4] = int(binary[3])

    return w_bar 



    pass

def bias_to_vco_transform(bias):
    return energy_per_spike * 100000 / bias 
    



