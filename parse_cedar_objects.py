import cedar_modules
import numpy as np
import nengo

def neural_field_parser(neuralfield_params):
    name = neuralfield_params[1][0][1]
    sizes = [int(s) for s in neuralfield_params[1][6][1]]
    # resting level
    h = float(neuralfield_params[1][8][1])
    # time scale
    tau = float(neuralfield_params[1][7][1])
    # global inhibition
    c_glob = float(neuralfield_params[1][11][1]) 
    borderType = neuralfield_params[1][13][1][0][1]
    border_type = 'zero-filled borders' if borderType == 'Zero' else 'cyclic'
    input_noise_gain = float(neuralfield_params[1][9][1])

    #kernel
    kernel_params = neuralfield_params[1][12][1]
    # need to check if there's only one kernel or several
    if len(kernel_params) == 1:
        if kernel_params[0][0] == 'cedar.aux.kernel.Box':
            amplitude = float(kernel_params[0][1][2][1])
            kernel = cedar_modules.BoxKernel(amplitude)
        elif kernel_params[0][0] == 'cedar.aux.kernel.Gauss':
            # amplitude
            c = float(kernel_params[0][1][2][1])
            sigma = float(kernel_params[0][1][3][1][0])
            normalize = True if kernel_params[0][1][4][1] == 'true' else False
            dims = int(kernel_params[0][1][0][1])
            # in cedar the 0-dimensional neural fields work with 1-dimensional 
            # kernels, but only use the center value
            # this is the same as just using a 0-dimensional gauss kernel
            if len(sizes) == 0 and dims == 1:
                kernel = cedar_modules.GaussKernel(c, sigma, normalize, 0)
            else:
                kernel = cedar_modules.GaussKernel(c, sigma, normalize, dims)
    else:
        if kernel_params[0][0] == 'cedar.aux.kernel.Box':
            amplitude = float(kernel_params[0][1][2][1])
            kernel = cedar_modules.BoxKernel(amplitude)
        elif kernel_params[0][0] == 'cedar.aux.kernel.Gauss':
            # amplitude
            c = float(kernel_params[0][1][2][1])
            sigma = float(kernel_params[0][1][3][1][0])
            normalize = True if kernel_params[0][1][4][1] == 'true' else False
            dims = int(kernel_params[0][1][0][1])
            # in cedar the 0-dimensional neural fields work with 1-dimensional 
            # kernels, but only use the center value
            # this is the same as just using a 0-dimensional gauss kernel
            if len(sizes) == 0 and dims == 1:
                kernel1 = cedar_modules.GaussKernel(c, sigma, normalize, 0)
            else:
                kernel1 = cedar_modules.GaussKernel(c, sigma, normalize, dims)
        if kernel_params[1][0] == 'cedar.aux.kernel.Box':
            amplitude = float(kernel_params[0][1][2][1])
            kernel = cedar_modules.BoxKernel(amplitude)
        elif kernel_params[1][0] == 'cedar.aux.kernel.Gauss':
            # amplitude
            c = float(kernel_params[1][1][2][1])
            sigma = float(kernel_params[1][1][3][1][0])
            normalize = True if kernel_params[1][1][4][1] == 'true' else False
            dims = int(kernel_params[1][1][0][1])
            # in cedar the 0-dimensional neural fields work with 1-dimensional 
            # kernels, but only use the center value
            # this is the same as just using a 0-dimensional gauss kernel
            if len(sizes) == 0 and dims == 1:
                kernel2 = cedar_modules.GaussKernel(c, sigma, normalize, 0)
            else:
                kernel2 = cedar_modules.GaussKernel(c, sigma, normalize, dims)
        kernel = [kernel1, kernel2]
    # else:
    #     print('Kernel not known!')

    # nonlinearity
    sigmoid_params = neuralfield_params[1][10]
    beta = float(sigmoid_params[1][2][1])
    threshold = float(sigmoid_params[1][1][1])
    sigmoid = cedar_modules.AbsSigmoid(beta, threshold)

    return name, cedar_modules.NeuralField(sizes, h, tau, kernel, c_glob,
                                           sigmoid, border_type, 
                                           input_noise_gain, name)
    
def boost_parser(boost_params):
    name = boost_params[1][0][1]
    strength = float(boost_params[1][1][1])

    return name, cedar_modules.Boost(strength, name)

def component_multiply_parser(cm_params):
    name = cm_params[1][0][1]
    inp_size1 = [int(s) for s in cm_params[1][-2][1]]
    inp_size2 = [int(s) for s in cm_params[1][-1][1]]

    return name, cedar_modules.ComponentMultiply(inp_size1, inp_size2, name)

def const_matrix_parser(cm_params):
    name = cm_params[1][0][1]
    sizes = [int(s) for s in cm_params[1][1][1]]
    value = int(cm_params[1][2][1])

    return name, cedar_modules.ConstMatrix(sizes, value, name)

def convolution_parser(conv_params):
    name = conv_params[1][0][1]
    sizes = [int(s) for s in conv_params[1][-1][1]]
    borderType = conv_params[1][2][1][0][1]
    border_type = 'zero-filled borders' if borderType == 'Zero' else 'cyclic'

    return name, cedar_modules.Convolution(sizes, border_type, name)

def flip_parser(flip_params):
    name = flip_params[1][0][1]
    flip_dimensions = [True if flip_params[1][1][1][0] == 'true' else False,
                       True if flip_params[1][1][1][1] == 'true' else False]
    sizes = [int(s) for s in flip_params[1][-1][1]]

    return name, cedar_modules.Flip(sizes, flip_dimensions, name)

def gauss_input_parser(gi_params):
    name = gi_params[1][0][1]
    sizes = [int(s) for s in gi_params[1][2][1]]
    centers = [float(c) for c in gi_params[1][4][1]]
    sigmas = [float(s) for s in gi_params[1][5][1]]
    a = float(gi_params[1][3][1])

    return name, cedar_modules.GaussInput(sizes, centers, sigmas, a, name)

def projection_parser(pro_params):
    name = pro_params[1][0][1]
    sizes_out = [int(s) for s in pro_params[1][3][1]]
    sizes_in = [int(s) for s in pro_params[1][-1][1]]
    compression_type = 'max' if pro_params[1][4][1] == "MAXIMUM" else 'sum'
    dimension_mapping = {}
    for kv_pair in pro_params[1][1][1]:
        dimension_mapping[kv_pair[0]] = int(kv_pair[1]) if int(kv_pair[1]) < 3 else False

    return name, cedar_modules.Projection(sizes_out, sizes_in, dimension_mapping,
                                          compression_type, name)

def spatial_template_parser(template_params):
    name = template_params[1][0][1]
    sizes = [int(template_params[1][1][1]), int(template_params[1][2][1])]
    invert_sides = True if template_params[1][3][1] == 'true' else False
    horizontal_pattern = True if template_params[1][4][1] == 'true' else False
    sigma_th_hor = float(template_params[1][5][1])
    mu_r = float(template_params[1][6][1])
    sigma_r = float(template_params[1][7][1])
    sigma_sigmoid_fw = float(template_params[1][8][1])

    return name, cedar_modules.SpatialTemplate(sizes, invert_sides, 
                                               horizontal_pattern, sigma_th_hor,
                                               mu_r, sigma_r, sigma_sigmoid_fw,
                                               name)


def static_gain_parser(sg_params):
    name = sg_params[1][0][1]
    sizes = [int(s) for s in sg_params[1][-1][1]]
    gain_factor = float(sg_params[1][1][1])

    return name, cedar_modules.StaticGain(sizes, gain_factor, name)

def parse_cedar_params(params):
    if params[0] == 'cedar.dynamics.NeuralField':
        return neural_field_parser(params)
    elif params[0] == 'cedar.processing.sources.Boost':
        return boost_parser(params)
    elif params[0] == 'cedar.processing.ComponentMultiply':
        return component_multiply_parser(params)
    elif params[0] == 'cedar.processing.sources.ConstMatrix':
        return const_matrix_parser(params)
    elif params[0] == 'cedar.processing.steps.Convolution':
        return convolution_parser(params)
    elif params[0] == 'cedar.processing.Flip':
        return flip_parser(params)
    elif params[0] == 'cedar.processing.sources.GaussInput':
        return gauss_input_parser(params)
    elif params[0] == 'cedar.processing.Projection':
        return projection_parser(params)
    elif params[0] == 'cedar.processing.sources.SpatialTemplate':
        return spatial_template_parser(params)
    elif params[0] == 'cedar.processing.StaticGain':
        return static_gain_parser(params)
    else:
        print('Object unknown: %s' %params[0])
        return None, None

def make_connection(source_name, target_name, object_dict):
    source = source_name.rsplit('.',1)[0]
    target = target_name.rsplit('.',1)[0]
    
    target_object = object_dict[target]
    # there are some special cases where the type of object
    # changes the connection that is made 
    # 1. special case: the target is a ComponentMultiply
    if target_object.__class__.__name__ == 'ComponentMultiply':
        if target_object.connections == 0:
            # connect to the first entries
            target_entries = target_object.node[:int(np.product(target_object.inp_size1))]
            target_size = target_object.inp_size1
            target_object.connections += 1
        elif target_object.connections == 1:
            # connect to the last entries
            target_entries = target_object.node[int(np.product(target_object.inp_size1)):]
            target_size = target_object.inp_size2
            # count plus 1 again to realize when trying to add a third connection
            target_object.connections += 1
        else:
            print('Too many connections for ComponentMultiply!')
    # 2. special case: the target is a Convolution
    elif target_object.__class__.__name__ == 'Convolution':
        target_size = target_object.sizes
        # the source can be the kernel or the matrix
        if target_name.rsplit('.',1)[1] == 'kernel':
            target_entries = target_object.node[:np.prod(target_object.sizes)]
        # if it's not the kernel, it's the matrix
        else:
            target_entries = target_object.node[np.prod(target_object.sizes):]
    # otherwise the target entries are all entries of the target node
    else:
        target_entries = target_object.node
        if target_object.__class__.__name__ == 'Projection':
            target_size = target_object.sizes_in 
        else:
            target_size = target_object.sizes

    source_object = object_dict[source]
    # 3. special case: the source is a NeuralField --> need sigmoided activation
    if source_object.__class__.__name__ == 'NeuralField':
        sigmoid = cedar_modules.AbsSigmoid()
        nengo.Connection(source_object.node, target_entries,
                         synapse=0, function=sigmoid)
    # TODO: 4. special case: the source is a static gain and its target has a 
    # different size than the output size of the static gain
    # solution: Put a Projection node in between
    elif  source_object.__class__.__name__ == 'StaticGain' \
        and source_object.sizes != target_size:
        if source_object.sizes == []:
            dimension_mapping = {}
            sizes_in = []
            sizes_out = target_object.sizes
            upscale = cedar_modules.Projection(sizes_out, sizes_in, 
                                               dimension_mapping, 'max',
                                               'Upscale Projection')
            upscale.make_node()
            # connect the Static Gain to the upscale
            nengo.Connection(source_object.node, upscale.node, synapse=0)
            # and connect the upscale to the target_object
            nengo.Connection(upscale.node, target_entries, synapse=0)

        else:
            print('Upscaling from', source_object.sizes, 'to', target_size, 
                  'not implemented yet!' )


    # just normal connection
    else:
        nengo.Connection(object_dict[source].node, target_entries,
                         synapse=0)
