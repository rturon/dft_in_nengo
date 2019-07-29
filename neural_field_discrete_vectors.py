import nengo
import numpy as np

# build discrete neural field with 10x10 discretization of two-dimensional space 
tau = 0.1
global_inh = -0.2
c_exc, c_inh, c_glob = 0.2, 0.05, 0.1
std_exc, std_inh = 1, 1

synapse = 0.1
model = nengo.Network()

with model:
    # u(x,t)
    u = nengo.Ensemble(500, dimensions=100)
    
    # s(x,t)
    input = nengo.Node([0]*100)
    s = nengo.Ensemble(500, dimensions=100)
    
    def input_function(x):
        return x / tau
        
    def kernel_function(d, c_exc, c_inh,c_glob, s_exc, s_inh):
        exc = c_exc * np.exp(-d**2 / 2*s_exc**2)
        inh = c_inh * np.exp(-d**2 / 2*s_inh**2)
        return  exc - inh - c_glob
        
    def sigmoid(x, beta=1):
        return 1 / (1 + np.exp(-beta * x))

    def recurrent_function(x):
        tmp = -x + global_inh
        
        integrals = np.zeros(len(x))
        for j in range(len(x)):
            x_pos = float(int(j / 100))
            y_pos = float(j % 100)
            for i in range(100):
                x_2_pos = float(int(i / 100))
                y_2_pos = float(i % 100)
                d = np.sqrt((y_pos - y_2_pos)**2 + (x_pos - x_2_pos)**2)
                k = kernel_function(d, c_exc, c_inh, c_glob, std_exc, std_inh)
                g = sigmoid(x)
                integrals[j] += k*g
                
        return tmp + integrals
        
    input_connections = []
    external_input_connections = []
    recurrent_connections = []

    nengo.Connection(input, s)
    nengo.Connection(s, u, synapse=synapse, function=input_function)
    for i in range(100):                                             
        recurrent_connections.append(nengo.Connection(u[i], u[i],
                                                      synapse=synapse,
                                                      function=lambda x: recurrent_function(x)))