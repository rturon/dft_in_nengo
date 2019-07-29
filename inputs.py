import numpy as np 

def two_dimensional_peak(p_x, p_y, std, a, size=10):
    
    x = np.arange(size)
    grid_x, grid_y = np.meshgrid(x, x)

    activations = a * np.exp(-0.5*((grid_x - p_x)**2/std**2 + \
                                   (grid_y - p_y)**2/std**2))

    return activations

def three_dimensional_peak(p_x, p_y, p_z, std, a, size=10):
    x = np.arange(size)
    grid_x, grid_y, grid_z = np.meshgrid(x, x, x)

    activations = a * np.exp(-0.5 * ((grid_x - p_x)**2 /std**2 + \
                                     (grid_y - p_y)**2 /std**2 + \
                                     (grid_z - p_z)**2 /std**2))

    return activations