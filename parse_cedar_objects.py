import cedar_modules

def neural_field_parser(neuralfield_params):
    pass

def boost_parser(boost_params):
    pass

def component_multiply_parser(cm_params):
    pass

def const_matrix_parser(cm_params):
    name = cm_params[1][0][1]
    sizes = [int(s) for s in cm_params[1][1][1]]
    value = int(cm_params[1][2][1])

    print("ConstantMatrix with params: \n name: %s \n sizes:" %name, sizes,
          "\n value: %i" %value)

def convolution_parser(conv_params):
    name = conv_params[1][0][1]
    # in the architecture the value for kernels is "", i.e. no kernel
    # TODO: looks like no kernel means just passing on the input --> 
    # check if that's really the case
    # if the convolution has two inputs, one of it is the kernel, the other
    # one is the input to convolve over

def flip_parser(flip_params):
    name = flip_params[1][0][1]
    flip_dimensions = [True if flip_params[1][1][1][0] == 'true' else False,
                       True if flip_params[1][1][1][1] == 'true' else False]
    sizes = [int(s) for s in flip_params[1][-1][1]]

    print("Flip with params: \n name: %s \n sizes:" %name, sizes,
          "\n flip dimensions: ", flip_dimensions)

def gauss_input_parser(gi_params):
    name = gi_params[1][0][1]
    sizes = [int(s) for s in gi_params[1][2][1]]
    centers = [float(c) for c in gi_params[1][4][1]]
    sigmas = [float(s) for s in gi_params[1][5][1]]
    a = float(gi_params[1][3][1])

    print("GaussInput with params: \n name: %s \n sizes:" %name, sizes,
          "\n centers: ", centers,
          "\n sigmas:", sigmas,
          "\n amplitude:", a)

def projection_parser(pro_params):
    name = pro_params[1][0][1]
    sizes_out = [int(s) for s in pro_params[1][3][1]]
    sizes_in = [int(s) for s in pro_params[1][-1][1]]
    compression_type = 'max' if pro_params[1][4][1] == "MAXIMUM" else 'sum'
    dimension_mapping = {}
    for kv_pair in pro_params[1][1][1]:
        dimension_mapping[kv_pair[0]] = int(kv_pair[1]) if int(kv_pair[1]) < 3 else False

    print("Projection with params: \n name: %s"%name, "\n sizes_in:", sizes_in,
          "\n sizes_out:", sizes_out, "\n compression_type:", compression_type,
          "\n dimension_mapping:", dimension_mapping)

def spatial_template_parser(template_params):
    name = template_params[1][0][1]
    sizes = [int(template_params[1][1][1]), int(template_params[1][2][1])]
    invert_sides = True if template_params[1][3][1] == 'true' else False
    horizontal_pattern = True if template_params[1][4][1] == 'true' else False
    sigma_th_hor = float(template_params[1][5][1])
    mu_r = float(template_params[1][6][1])
    sigma_r = float(template_params[1][7][1])
    sigma_sigmoid_fw = float(template_params[1][8][1])

    print("SpatialTemplate with params: \n name: %s"%name, "\n sizes:", sizes,
          "\n invert_sides:", invert_sides, 
          "\n horizontal_pattern:", horizontal_pattern,
          "\n sigma_th_hor:", sigma_th_hor,
          "\n mu_r:", mu_r,
          "\n sigma_r:", sigma_r,
          "\n sigma_sigmoid_fw:", sigma_sigmoid_fw)


def static_gain_parser(sg_params):
    name = sg_params[1][0][1]
    sizes = [int(s) for s in sg_params[1][-1][1]]
    gain_factor = float(sg_params[1][1][1])

    print("StaticGain with params: \n name: %s" %name,
          "\n sizes:", sizes, "\n gain_factor:", gain_factor)