# dy/dt = - y/tau + h + x/tau 
# neural field function without integral part and only for ONE LOCATION!!!
# --> ensembles x, y perform the right computations, but only for one location,
# to do this for several locations i need to copy the whole computation nxn 
# times to get a field with nxn locations --> and then i still only have discrete 
# locations 

import nengo

model = nengo.Network()
with model:
    stim_x = nengo.Node([0,0])
    x = nengo.Ensemble(n_neurons=50, dimensions=2)
    nengo.Connection(stim_x, x)
    
    y = nengo.Ensemble(n_neurons=50, dimensions=2)
    tau = 0.5
    synapse = 0.1
    
    
    x_1 = nengo.Ensemble(n_neurons=50, dimensions=2)
    nengo.Connection(stim_x, x_1)
    
    y_1 = nengo.Ensemble(n_neurons=50, dimensions=2)
    
    
    x_2 = nengo.Ensemble(n_neurons=50, dimensions=2)
    nengo.Connection(stim_x, x_2)
    
    y_2 = nengo.Ensemble(n_neurons=50, dimensions=2)
    
    
    x_3 = nengo.Ensemble(n_neurons=50, dimensions=2)
    nengo.Connection(stim_x, x_3)
    
    y_3 = nengo.Ensemble(n_neurons=50, dimensions=2)
    
    
    x_4 = nengo.Ensemble(n_neurons=50, dimensions=2)
    nengo.Connection(stim_x, x_4)
    
    y_4 = nengo.Ensemble(n_neurons=50, dimensions=2)
    
    
    x_5 = nengo.Ensemble(n_neurons=50, dimensions=2)
    nengo.Connection(stim_x, x_5)
    
    y_5 = nengo.Ensemble(n_neurons=50, dimensions=2)
    
    
    x_6 = nengo.Ensemble(n_neurons=50, dimensions=2)
    nengo.Connection(stim_x, x_6)
    
    y_6 = nengo.Ensemble(n_neurons=50, dimensions=2)
    
    
    x_7 = nengo.Ensemble(n_neurons=50, dimensions=2)
    nengo.Connection(stim_x, x_7)
    
    y_7 = nengo.Ensemble(n_neurons=50, dimensions=2)
    
    
    x_8 = nengo.Ensemble(n_neurons=50, dimensions=2)
    nengo.Connection(stim_x, x_8)
    
    y_8 = nengo.Ensemble(n_neurons=50, dimensions=2)
    
    
    def input_function(x):
        return x / tau * synapse
    def recurrent_function(y):
        return (-y / tau) * synapse + y
    
    
    nengo.Connection(x, y, synapse=synapse, function=input_function)
    nengo.Connection(y, y, synapse=synapse, function=recurrent_function)
    
    nengo.Connection(x_1, y_1, synapse=synapse, function=input_function)
    nengo.Connection(y_1, y_1, synapse=synapse, function=recurrent_function)
    
    nengo.Connection(x_2, y_2, synapse=synapse, function=input_function)
    nengo.Connection(y_2, y_2, synapse=synapse, function=recurrent_function)
    
    nengo.Connection(x_3, y_3, synapse=synapse, function=input_function)
    nengo.Connection(y_3, y_3, synapse=synapse, function=recurrent_function)
    
    nengo.Connection(x_4, y_4, synapse=synapse, function=input_function)
    nengo.Connection(y_4, y_4, synapse=synapse, function=recurrent_function)
    
    nengo.Connection(x_5, y_5, synapse=synapse, function=input_function)
    nengo.Connection(y_5, y_5, synapse=synapse, function=recurrent_function)
    
    nengo.Connection(x_6, y_6, synapse=synapse, function=input_function)
    nengo.Connection(y_6, y_6, synapse=synapse, function=recurrent_function)
    
    nengo.Connection(x_7, y_7, synapse=synapse, function=input_function)
    nengo.Connection(y_7, y_7, synapse=synapse, function=recurrent_function)
    
    nengo.Connection(x_8, y_8, synapse=synapse, function=input_function)
    nengo.Connection(y_8, y_8, synapse=synapse, function=recurrent_function)
    
    
    field = nengo.Ensemble(n_neurons=50, dimensions=18)
    nengo.Connection(y, field[:2])
    nengo.Connection(y_1, field[2:4])
    nengo.Connection(y_2, field[4:6])
    nengo.Connection(y_3, field[6:8])
    nengo.Connection(y_4, field[8:10])
    nengo.Connection(y_5, field[10:12])
    nengo.Connection(y_6, field[12:14])
    nengo.Connection(y_7, field[14:16])
    nengo.Connection(y_8, field[16:18])
    
    
    
    
    
    
    
    
    