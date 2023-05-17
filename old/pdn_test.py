import numpy as np
import PhaseDomainNeuron as pdn 


from matplotlib import pyplot as plt


test = pdn.PDN() 

# print(test.period) 
# input()

#test vco curve linear version
# x = np.linspace(0, 300, 1000) 
# y = []
# for i in x:
#     y.append(test.freq_from_v(i)) 

# plt.plot(x, y)
# plt.show() 

# input()



while test.time_step < 50e3:
    test.tick()
    if (test.time_step % 3300) >= 0 and (test.time_step % 3300) < 50:
        test.input_value = 1
    else:
        test.input_value = 0 

    # if test.time_step==5000:
    #     test.spike_flag = 1 

    if (test.time_step % 1000) == 0:
        print(test.time_step) 

    #test.plot_tick()

while True:
    test.plot_tick()

#test.tick_loop() 
#test.start_threads() 
# test.plot_input() 

# while True:
#     pass 

# test_bias = np.linspace(0, 1000, 300) 

# f = []

# for t in test_bias:
#     f.append(test.freq_from_v(t))

# plt.plot(test_bias, f)
# plt.show() 



