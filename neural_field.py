import numpy as np 
import nengo

def absSigmoid(x, beta=100):
    return 0.5 * (1 + beta * x) / (1 + beta * np.abs(x))

def neural_field_equation_recurrent(x, 
                                    kernel_fn, 
                                    h=-5., 
                                    c_glob=0., 
                                    beta=100, 
                                    tau=100, 
                                    sizes=[10,10],
                                    normalize=False):
    
    dims = len(sizes)
    integrals = np.zeros(len(x))
    gs = absSigmoid(x)
    g_sum = np.sum(gs)

    for i in range(len(x)):
        kernel = np.zeros(len(x))
        g_sum = 0
        if dims == 0:
            # don't know what should be done in this case
            continue
        if dims == 1:
            x_pos = i
        elif dims == 2:
            x_pos = i / sizes[1]
            y_pos = i % sizes[1]
        elif dims == 3:
            x_pos = i / (sizes[1]*sizes[2])
            y_pos = (i % (sizes[1]*sizes[2])) / sizes[2]
            z_pos = (i % (sizes[1]*sizes[2])) % sizes[2]
        for j in range(len(x)):
            if dims == 1:
                x_2_pos = j
                d = x_pos - x_2_pos
            elif dims == 2:
                x_2_pos = j / sizes[1]
                y_2_pos = j % sizes[1]
                d = np.sqrt((y_pos - y_2_pos)**2 + (x_pos - x_2_pos)**2)
            elif dims == 3:
                x_2_pos = j / (sizes[1]*sizes[2])
                y_2_pos = (j % (sizes[1]*sizes[2])) / sizes[2]
                z_2_pos = (j % (sizes[1]*sizes[2])) % sizes[2]
                d = np.sqrt((z_pos - z_2_pos)**2 + (y_pos - y_2_pos)**2 + (x_pos - x_2_pos)**2)
            
            k = kernel_fn(d)
            kernel[j] = k

        # if normalize is set first need to normalize the k values before multiplying with g's
        if normalize:
            kernel /= np.sum(kernel)
        integrals[i] = np.sum(np.multiply(kernel, gs))

    # return (h + integrals - c_glob * g_sums - x) / tau + x 
    return h + integrals - c_glob * g_sum

def gauss_kernel(d, c, s):
    # TODO: no normalization yet
    return c * np.exp(- d**2 / (2 * s**2)) 


def neural_field(num_neurons,  # or compute appropriate size from sizes?
                 radius,
                 kernel_fn,
                 sizes,
                 h=-5,
                 tau=100,
                 c_glob=0,
                 beta=100,
                 normalize=False,
                 synapse=0.1):  # add input_noise_gain
        
        dimensions = np.prod(sizes)
        print(dimensions)
        u = nengo.Ensemble(num_neurons, 
                                dimensions=dimensions, 
                                radius=radius, 
                                label="NeuralField activation",
                                neuron_type=nengo.Direct())

        recurrent_fn = lambda x: neural_field_equation_recurrent(x, kernel_fn, 
                                                                 h, c_glob, beta, 
                                                                 tau, sizes, normalize)
        nengo.Connection(u, u, synapse=synapse, function=recurrent_fn)

        return u









