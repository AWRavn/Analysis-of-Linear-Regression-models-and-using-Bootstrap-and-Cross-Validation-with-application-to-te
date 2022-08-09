import numpy as np
from imageio import imread

RAND=13



def franke_function(x, y):
    """
    Returns the Franke's function that has two Gaussian peaks of different heights and a smaller dip. 

    Args:
        x (np.Array[float]):        Inputs within [0, 1]
        y (np.Array[float]):        Inputs within [0, 1]

    Returns:
        z (np.Array[float]):        Outputs of the Franke's function
    """

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return term1 + term2 + term3 + term4


def prepare_data(N, sigma=0.3):
    """
    Returns the Franke's function that has two Gaussian peaks of different heights and a smaller dip. 
    The noise may be added stochastic noise with the normal distribution N[0, 1].
    """

    x = np.linspace(0, 1, N) 
    y = np.linspace(0, 1, N) 
    x_2d, y_2d = np.meshgrid(x, y)
    x = np.ravel(x_2d)
    y = np.ravel(y_2d)

    data = franke_function(x, y)

    noise = np.random.normal(0,sigma,(x.shape))
    data = data + noise

    return data, x, y


def prepare_terrain_data(N, filename):
    """
    Returns the aranged terrain data from the provided file.
    """

    # Make dataset
    _data = imread('./data/{}'.format(filename) )
    data = _data[:N, :N]
    data = np.ravel(data)

    x = np.linspace(0,1, N)
    y = np.linspace(0,1, N)
    x_2d, y_2d = np.meshgrid(x, y)
    x = np.ravel(x_2d)
    y = np.ravel(y_2d)

    return data, x, y