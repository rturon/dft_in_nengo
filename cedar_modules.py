import numpy as np
import scipy.signal
import nengo

##############################################################################
############################ helper functions ################################
class AbsSigmoid(object):
    def __init__(self, beta=100, threshold=0):
        self.beta = beta
        self.threshold = threshold
    def __call__(self, x):
        return 0.5 * (1 + self.beta * (x-self.threshold) / (1 + self.beta * np.abs(x-self.threshold)))

def one_dimensional_peak(p, std, a, size=10):
    x = np.arange(size)

    activations = a * np.exp(-0.5*(x-p)**2/std**2)

    return activations

def two_dimensional_peak(p, std, a, size=[10,10]):
    
    x = np.arange(size[1])
    y = np.arange(size[0])
    grid_x, grid_y = np.meshgrid(x, y)
    

    activations = a * np.exp(-0.5*((grid_x - p[1])**2/std[1]**2 + \
                                   (grid_y - p[0])**2/std[0]**2))

    return activations

def three_dimensional_peak(p, std, a, size=[10,10,10]):
    x = np.arange(size[0])
    y = np.arange(size[1])
    z = np.arange(size[2])
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z)

    activations = a * np.exp(-0.5 * ((grid_x - p[0])**2 /std[0]**2 + \
                                     (grid_y - p[1])**2 /std[1]**2 + \
                                     (grid_z - p[2])**2 /std[2]**2))

    return activations

def make_gaussian(sizes, centers, sigmas, a):
    if len(sizes) == 1:
        activations = one_dimensional_peak(centers[0], sigmas[0], a, sizes[0])
        
    elif len(sizes) == 2:
        activations = two_dimensional_peak(centers, sigmas, a, sizes)
        
    elif len(sizes) == 3:
        activations = three_dimensional_peak(centers, sigmas, a, sizes)
        
    # TODO: what to do if len(sizes) not between 1 and 3?
    return activations

def reduce(inp, dimension_mapping, compression_type):
    # get axis/axes to reduce
    axis_reduce = tuple([int(key) for key in dimension_mapping if type(dimension_mapping[key]) == bool])
    # reduce inp by using sum or max
    if compression_type == 'sum':
        out = np.sum(inp, axis=axis_reduce)
    elif compression_type == 'max':
        out = inp.max(axis=axis_reduce)
    
    # transpose array st order of dimensions is as given in dimension mapping
    dim_order = [int(value) for value in dimension_mapping.values() if value]

    out = out.transpose(*dim_order)
    
    return out

def add_dimensions(inp, dimension_mapping, out_sizes):
    
    added_dims = list(np.arange(len(out_sizes)))
    mapped_dims = [int(value) for value in dimension_mapping.values()]
    [added_dims.remove(dim) for dim in mapped_dims]

    # add dimensions
    for dim in added_dims:
        # add the dimension
        inp = np.expand_dims(inp, axis=dim)
        # repeat dimension to get right size
        inp = np.repeat(inp, repeats=out_sizes[dim], axis=dim)
        
    return inp

class GaussKernel(object):
    def __init__(self, c, sigma, normalize=True, dims=2):
        self.c = c
        self.sigma = sigma
        self.normalize = normalize
        self.kernel_width = int(np.ceil(sigma*5)) # limit is always 5 
        # kernel width should always be an odd number --> like that in cedar
        if self.kernel_width % 2 == 0:
            self.kernel_width += 1
        self.dims = dims
        
        x = np.arange(self.kernel_width)
        cx = self.kernel_width//2
        
        if self.dims == 1 or self.dims == 0:
            dx = np.abs(x - cx)
            
        elif self.dims == 2:
            grid_x, grid_y = np.meshgrid(x, x)
            dx = np.sqrt((grid_x - cx)**2 + (grid_y - cx)**2)
         
        elif self.dims == 3:
            grid_x, grid_y, grid_z = np.meshgrid(x, x, x)
            dx = np.sqrt((grid_x - cx)**2 + (grid_y - cx)**2 + (grid_z - cx)**2)
            
        kernel_matrix = np.exp(-dx**2 / (2*self.sigma**2))
        if self.normalize:
            kernel_matrix /= np.sum(kernel_matrix)
        self.kernel_matrix = self.c * kernel_matrix
        
        # if the kernel is 0-dimensional it only consists of the central scalar value
        # need to compute from 1-dimensional kernel for normalizaton to work correctly
        if self.dims == 0:
            self.kernel_matrix = self.kernel_matrix[cx]
            
    def __call__(self):
        return self.kernel_matrix
    
class BoxKernel(object):
    ''' Implementation of the BoxKernel of cedar.
        Since the BoxKernel is only used with 0-dimensional fields in the 
        architecture this is a simplified implementation of the BoxKernel
        without the width parameter which is only needed for 1- or higher 
        dimensional fields.
    '''
    def __init__(self, amplitude):
        # dimensionality of BoxKernel always 0
        self.dims = 0 
        self.amplitude = amplitude
        self.kernel_matrix = amplitude
        
    def __call__(self):
        return self.kernel_matrix
    
def pad_and_convolve(inp, kernel, border):
    
    # test if input consists of more than one number, i.e. field not 0-dimensional
    if inp.shape != ():
        # compute padding values
        kernel_width = kernel.shape[0]
        pad_width_f = kernel_width//2
        pad_width_b = kernel_width//2 if kernel_width % 2 == 1 else kernel_width//2 - 1 

        # translate border type to boundary mode
        mode = 'wrap' if border == 'cyclic' else 'constant'
        # pad the input
        inp_padded = np.pad(inp, pad_width=(pad_width_f, pad_width_b), mode=mode)
    
    # otherwise the kernel is just a scalar and can directly be convolved with the 
    # scalar input
    else:
        inp_padded = inp
    # perform the convolution
    conv = scipy.signal.convolve(inp_padded, kernel, mode='valid')
    
    return conv

def create_template(sizes, invert_sides, horizontal_pattern, sigma_th, 
                    mu_r, sigma_r, sigma_sigmoid):
    if invert_sides:
        invert_sides = -1
    else:
        invert_sides = 1
        
    size_x = sizes[0]
    size_y = sizes[1]
    
    shift_x = ((size_x - 1) / 2)
    shift_y = ((size_y - 1) / 2)
    
    x_grid, y_grid = np.meshgrid(np.arange(size_x), np.arange(size_y))
    
    x_shifted = x_grid - shift_x
    y_shifted = y_grid - shift_y
    
    x = x_shifted
    y = y_shifted 
    
    if horizontal_pattern:
        x = y_shifted
        y = x_shifted
        
    th = np.arctan2(y, invert_sides * x)
    r = np.log(np.sqrt(x**2 + y**2))
    
    gaussian = np.exp(-0.5 * th**2 / sigma_th**2 \
                      - 0.5 * (r - mu_r)**2 / sigma_r**2)
    sigmoid = invert_sides * AbsSigmoid()(x)
    
    pattern = (1 - sigma_sigmoid) * gaussian + sigma_sigmoid * sigmoid
    
    return pattern.transpose(1,0)
##############################################################################
############################# module classes #################################
class NeuralField(object):
    ''' Neural Field object similar to the Neural Field used by cedar.
        Implements the Neural Field equation of Dynamic Field theory.
        The dimensionality of the Neural Field is read from the number
        of elements in the sizes parameter. 
    '''
    def __init__(self, 
                 sizes, 
                 h, 
                 tau, 
                 kernel, 
                 c_glob=1, 
                 nonlinearity=AbsSigmoid(beta=100),
                 border_type='zero-filled borders',
                 input_noise_gain=0.1,
                 name=None):
        self.u = np.ones(sizes) * h
        self.h = h
        self.tau = tau
        self.sizes = sizes
        self.c_glob = c_glob
        self.nonlinearity = nonlinearity
        self.border_type = border_type
        self.input_noise_gain = input_noise_gain
        self.probes = {"sigmoided activation": [],
                       "lateral interaction": [],
                       "activation": [],
                       "sigmoided sum": []}
        
        assert (kernel.dims == len(sizes)), "Kernel must have same number of " + \
                                             "dimensions as Neural Field!"
        self.kernel = kernel
        
        self.kernel_matrix = kernel()
        self.name = name
        
    def update(self, stim):
        a = self.nonlinearity(self.u)
        self.probes["sigmoided activation"].append(a)
        recurr = pad_and_convolve(a, self.kernel_matrix, self.border_type)
        self.probes["lateral interaction"].append(recurr)
        # in cedar the noise is divided by sqrt(tau)
        self.u += (-self.u + self.h + self.c_glob * np.sum(a) + recurr + stim)/self.tau + \
                  (self.input_noise_gain * np.random.randn(*self.sizes)) / self.tau 
        self.probes["activation"].append(np.array(self.u))
        self.probes["sigmoided sum"].append(np.sum(a))
        # return sigmoided activation and not activation?
        # self.u = self.nonlinearity(self.u)
        return self.u
        
    def make_node(self):
        if self.name is not None:
            self.node = nengo.Node(lambda t, x: self.update(x.reshape(self.sizes)).flatten(),
                          size_in=int(np.product(self.sizes)), 
                          size_out=int(np.product(self.sizes)),
                          label=self.name)
        else:
            self.node = nengo.Node(lambda t, x: self.update(x.reshape(self.sizes)).flatten(),
                          size_in=int(np.product(self.sizes)), 
                          size_out=int(np.product(self.sizes)))
                          

class ComponentMultiply(object):
    def __init__(self, inp_size1, inp_size2, name=None):
        self.inp_size1 = inp_size1
        self.inp_size2 = inp_size2
        if len(self.inp_size1) >= len(self.inp_size2):
            self.out_size = self.inp_size1
        else:
            self.out_size = self.inp_size2
        # internal counter for number of connections to know where to connect to
        self.connections = 0
        self.name = name
        
    def update(self, inp):
        # get the index of where input1 and input2 are seperated
        sep = np.prod(self.inp_size1)
        inp1 = inp[:sep]
        inp2 = inp[sep:]
        
        if self.inp_size1 == self.inp_size2 or self.inp_size1 == [] or self.inp_size2 == []:
            return inp1 * inp2
        
        else:
            inp1 = inp1.reshape(*self.inp_size1)
            inp2 = inp2.reshape(*self.inp_size2)
            return (inp1 * inp2).flatten()
    
    def make_node(self):
        if self.name is not None:
            self.node = nengo.Node(lambda t, x: self.update(x), 
                          size_in=int(np.prod(self.inp_size1)+np.prod(self.inp_size2)), 
                          size_out=np.prod(self.out_size),
                          label=self.name)
        else:
            self.node = nengo.Node(lambda t, x: self.update(x), 
                          size_in=int(np.prod(self.inp_size1)+np.prod(self.inp_size2)), 
                          size_out=np.prod(self.out_size))


class GaussInput(object):
    def __init__(self, sizes, centers, sigmas, a, name=None):
        
        # add asserts to check if sizes same length as centers and sigmas
        self.sizes = sizes
        self.centers = centers
        self.sigmas = sigmas
        self.a = a
        self.name = name
        
    def make_node(self):
        if self.name is not None:
            self.node = nengo.Node(make_gaussian(self.sizes, self.centers, self.sigmas, self.a).flatten(),
                                   label=self.name)
        else:
            self.node = nengo.Node(make_gaussian(self.sizes, self.centers, self.sigmas, self.a).flatten())


class ConstMatrix(object):
    def __init__(self, sizes, value, name=None):
        self.sizes = sizes
        self.value = value
        self.name = name
        
    def make_node(self):
        if self.name is not None:
            self.node = nengo.Node(np.ones(np.prod(self.sizes))*self.value,
                                   label=name)
        else:
            self.node = nengo.Node(np.ones(np.prod(self.sizes))*self.value)


class StaticGain(object):
    def __init__(self, sizes, gain_factor, name=None):
        self.sizes = sizes
        self.gain_factor = gain_factor
        self.name = name
        
    def update(self, inp):
        return inp * self.gain_factor
    
    def make_node(self):
        if self.name is not None:
            self.node = nengo.Node(lambda t, x: self.update(x), size_in=np.prod(self.sizes),
                                   label=self.name)
        else:
            self.node = nengo.Node(lambda t, x: self.update(x), size_in=np.prod(self.sizes))


class Flip(object):
    def __init__(self, sizes, flip_dimensions, name=None):
        self.sizes = sizes
        self.flip_dimensions = flip_dimensions
        self.name = name
        
    def update(self, inp):
        
        out = inp.reshape(*self.sizes)
        if self.flip_dimensions[0]:
            out = np.flip(out, axis=0)
        if self.flip_dimensions[1]:
            out = np.flip(out, axis=1)
            
        return out.flatten()
    
    def make_node(self):
        if self.name is not None:
            self.node = nengo.Node(lambda t, x: self.update(x), size_in=np.prod(self.sizes), 
                          size_out=np.prod(self.sizes), label=self.name)
        else:
            self.node = nengo.Node(lambda t, x: self.update(x), size_in=np.prod(self.sizes), 
                          size_out=np.prod(self.sizes))


class Projection(object):
    def __init__(self, sizes_out, sizes_in, dimension_mapping, compression_type,
                 name=None):
        self.sizes_out = sizes_out
        self.sizes_in = sizes_in
        self.dimension_mapping = dimension_mapping
        self.compression_type = compression_type
        self.name = name
        
    def update(self, inp):
        # reshape inp
        if self.sizes_in != []:
            out = inp.reshape(*self.sizes_in)
        else:
            out = inp
        
        # either downsizing
        if len(self.dimension_mapping) > len(self.sizes_out):
            out = reduce(out, self.dimension_mapping, self.compression_type)
            
        # or upsizing
        elif len(self.sizes_out) > len(self.dimension_mapping):
            out = add_dimensions(out, self.dimension_mapping, self.sizes_out)
            
        return out.flatten()
            
        
    def make_node(self):
        if self.name is not None:
            self.node = nengo.Node(lambda t, x: self.update(x), 
                          size_in=np.prod(self.sizes_in) if self.sizes_in != [] else 1, 
                          size_out=np.prod(self.sizes_out) if self.sizes_out != [] else 1,
                          label=self.name)
        else:
            self.node = nengo.Node(lambda t, x: self.update(x), 
                          size_in=np.prod(self.sizes_in) if self.sizes_in != [] else 1, 
                          size_out=np.prod(self.sizes_out) if self.sizes_out != [] else 1)


class Convolution(object):
    ''' Has two inputs of same size: First input is the kernel for the 
        convolution, second input is the matrix to convolve with. Since they
        always have the same size in the spatial reasoning architecture I only
        implemented this case here for simplicity. 
    '''
    def __init__(self, sizes, border_type, name=None):
        self.sizes = sizes
        self.border_type = border_type
        self.name = name
        
    def update(self, inp):
        kernel = inp[:np.prod(self.sizes)].reshape(*self.sizes)
        matrix = inp[np.prod(self.sizes):].reshape(*self.sizes)
        return pad_and_convolve(matrix, kernel, self.border_type).flatten()
    
    def make_node(self):
        if self.name is not None:
            self.node = nengo.Node(lambda t, x: self.update(x), size_in=np.prod(self.sizes)*2, 
                          size_out=np.prod(self.sizes), label=name)
        else:
            self.node = nengo.Node(lambda t, x: self.update(x), size_in=np.prod(self.sizes)*2, 
                          size_out=np.prod(self.sizes))

    
class SpatialTemplate(object):
    def __init__(self, sizes, invert_sides, horizontal_pattern, sigma_th_hor, 
                 mu_r, sigma_r, sigma_sigmoid_fw, name=None):
        self.sizes = sizes
        self.invert_sides = invert_sides
        self.horizontal_pattern = horizontal_pattern
        self.sigma_th_hor = sigma_th_hor
        self.mu_r = mu_r
        self.sigma_r = sigma_r
        self.sigma_sigmoid_fw = sigma_sigmoid_fw
        self.name = name
        
    def make_node(self):
        if self.name is not None:
            self.node = nengo.Node(create_template(self.sizes, self.invert_sides, self.horizontal_pattern,
                                          self.sigma_th_hor, self.mu_r, self.sigma_r, 
                                          self.sigma_sigmoid_fw).flatten(), label=self.name)
        else:
            self.node = nengo.Node(create_template(self.sizes, self.invert_sides, self.horizontal_pattern,
                                          self.sigma_th_hor, self.mu_r, self.sigma_r, 
                                          self.sigma_sigmoid_fw).flatten())


class Boost(object):
    def __init__(self, strength, name=None):
        self.strength = strength
        self.name = name

    def make_node(self):
        if self.name is not None:
            self.node = nengo.Node([self.strength], label=self.name)
        else:
            self.node = nengo.Node([self.strength])