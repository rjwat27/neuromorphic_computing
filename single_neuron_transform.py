import numpy as np

import PhaseDomainNeuron as pdn 
import weight_bias_transform as wbt
from matplotlib import pyplot as plt

# x = range(len(pdn.vco_curve))
# x2 = np.linspace(0, 330, 1000)
# y = [pdn.interpolate(i, pdn.vco_curve) for i in x2]
# plt.plot(x, pdn.vco_curve)
# plt.plot(x2, y, color='magenta')
# plt.show()
# print(pdn.interpolate(200, pdn.vco_curve))
# input()


in_val = .1

out_val = 0.3092086823759779

beta = out_val / in_val 

vref = 150#wbt.bias_to_vco_transform(beta)
#input(vref)

n = pdn.PDN(vref=vref) 


#vals = n.forward_burst(in_val, 10000, adjust=True, output_val=out_val) 

n.calibrate_vref(in_val, out_val) 

vals = n.forward_burst(in_val, 10000, adjust=True, output_val=out_val) 

print(np.average(vals), n.vref) 


# y1 = n.INPUT_STREAM 
# y2 = n.vco_bias_stream
# y3 = n.output_stream

# plt.plot(x, y1)
# plt.show()

# plt.plot(x, y2)
# plt.show()

# plt.plot(x, y3)
# plt.show()














