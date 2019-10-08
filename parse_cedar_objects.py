import cedar_modules

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
    if kernel_params[0][0] == 'cedar.aux.kernel.Box':
        amplitude = float(kernel_params[0][1][2][1])
        kernel = cedar_modules.BoxKernel(amplitude)
    elif kernel_params[0][0] == 'cedar.aux.kernel.Gauss':
        # amplitude
        c = float(kernel_params[0][1][2][1])
        sigma = float(kernel_params[0][1][3][1][0])
        normalize = True if kernel_params[0][1][4][1] == 'true' else False
        dims = int(kernel_params[0][1][0][1])
        kernel = cedar_modules.GaussKernel(c, sigma, normalize, dims)
    else:
        print('Kernel not known!')

    # nonlinearity
    sigmoid_params = neuralfield_params[1][10]
    beta = float(sigmoid_params[1][2][1])
    threshold = float(sigmoid_params[1][1][1])
    sigmoid = cedar_modules.AbsSigmoid(beta, threshold)

    # print('Neural Field with params: \n',
    #       'name:', name,
    #       '\n sizes:', sizes,
    #       '\n h:', h,
    #       '\n tau:', tau,
    #       '\n c_glob:', c_glob,
    #       '\n border_type:', border_type,
    #       '\n input_noise_gain:', input_noise_gain,
    #       '\n kernel:', kernel,
    #       '\n nonlinearity:', sigmoid)

    return name, cedar_modules.NeuralField(sizes, h, tau, kernel, c_glob,
                                           sigmoid, border_type, 
                                           input_noise_gain)
    
def boost_parser(boost_params):
    name = boost_params[1][0][1]
    strength = float(boost_params[1][1][1])

    return name, cedar_modules.Boost(strength)

def component_multiply_parser(cm_params):
    name = cm_params[1][0][1]
    inp_size1 = [int(s) for s in cm_params[1][-2][1]]
    inp_size2 = [int(s) for s in cm_params[1][-1][1]]

    # print("ComponentMultiply with params: \n name: %s \n inp_size1:" %name, 
    #       inp_size1, "\n inp_size2:", inp_size2)
    return name, cedar_modules.ComponentMultiply(inp_size1, inp_size2)

def const_matrix_parser(cm_params):
    name = cm_params[1][0][1]
    sizes = [int(s) for s in cm_params[1][1][1]]
    value = int(cm_params[1][2][1])

    # print("ConstantMatrix with params: \n name: %s \n sizes:" %name, sizes,
    #       "\n value: %i" %value)
    return name, cedar_modules.ConstMatrix(sizes, value)

def convolution_parser(conv_params):
    name = conv_params[1][0][1]
    sizes = [int(s) for s in conv_params[1][-1][1]]
    borderType = conv_params[1][2][1][0][1]
    border_type = 'zero-filled borders' if borderType == 'Zero' else 'cyclic'
    
    # print('Convolution with params:', 
    #       '\n name:', name,
    #       '\n sizes:', sizes,
    #       '\n border_type:', border_type)
    return name, cedar_modules.Convolution(sizes, border_type)

def flip_parser(flip_params):
    name = flip_params[1][0][1]
    flip_dimensions = [True if flip_params[1][1][1][0] == 'true' else False,
                       True if flip_params[1][1][1][1] == 'true' else False]
    sizes = [int(s) for s in flip_params[1][-1][1]]

    # print("Flip with params: \n name: %s \n sizes:" %name, sizes,
    #       "\n flip dimensions: ", flip_dimensions)
    return name, cedar_modules.Flip(sizes, flip_dimensions)

def gauss_input_parser(gi_params):
    name = gi_params[1][0][1]
    sizes = [int(s) for s in gi_params[1][2][1]]
    centers = [float(c) for c in gi_params[1][4][1]]
    sigmas = [float(s) for s in gi_params[1][5][1]]
    a = float(gi_params[1][3][1])

    # print("GaussInput with params: \n name: %s \n sizes:" %name, sizes,
    #       "\n centers: ", centers,
    #       "\n sigmas:", sigmas,
    #       "\n amplitude:", a)
    return name, cedar_modules.GaussInput(sizes, centers, sigmas, a)

def projection_parser(pro_params):
    name = pro_params[1][0][1]
    sizes_out = [int(s) for s in pro_params[1][3][1]]
    sizes_in = [int(s) for s in pro_params[1][-1][1]]
    compression_type = 'max' if pro_params[1][4][1] == "MAXIMUM" else 'sum'
    dimension_mapping = {}
    for kv_pair in pro_params[1][1][1]:
        dimension_mapping[kv_pair[0]] = int(kv_pair[1]) if int(kv_pair[1]) < 3 else False

    # print("Projection with params: \n name: %s"%name, "\n sizes_in:", sizes_in,
    #       "\n sizes_out:", sizes_out, "\n compression_type:", compression_type,
    #       "\n dimension_mapping:", dimension_mapping)
    return name, cedar_modules.Projection(sizes_out, sizes_in, dimension_mapping,
                                          compression_type)

def spatial_template_parser(template_params):
    name = template_params[1][0][1]
    sizes = [int(template_params[1][1][1]), int(template_params[1][2][1])]
    invert_sides = True if template_params[1][3][1] == 'true' else False
    horizontal_pattern = True if template_params[1][4][1] == 'true' else False
    sigma_th_hor = float(template_params[1][5][1])
    mu_r = float(template_params[1][6][1])
    sigma_r = float(template_params[1][7][1])
    sigma_sigmoid_fw = float(template_params[1][8][1])

    # print("SpatialTemplate with params: \n name: %s"%name, "\n sizes:", sizes,
    #       "\n invert_sides:", invert_sides, 
    #       "\n horizontal_pattern:", horizontal_pattern,
    #       "\n sigma_th_hor:", sigma_th_hor,
    #       "\n mu_r:", mu_r,
    #       "\n sigma_r:", sigma_r,
    #       "\n sigma_sigmoid_fw:", sigma_sigmoid_fw)
    return name, cedar_modules.SpatialTemplate(sizes, invert_sides, 
                                               horizontal_pattern, sigma_th_hor,
                                               mu_r, sigma_r, sigma_sigmoid_fw)


def static_gain_parser(sg_params):
    name = sg_params[1][0][1]
    sizes = [int(s) for s in sg_params[1][-1][1]]
    gain_factor = float(sg_params[1][1][1])

    # print("StaticGain with params: \n name: %s" %name,
    #       "\n sizes:", sizes, "\n gain_factor:", gain_factor)
    return name, cedar_modules.StaticGain(sizes, gain_factor)

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
