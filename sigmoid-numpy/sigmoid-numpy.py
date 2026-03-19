import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x = np.array(x)
    return (1.0/(1.0 + np.exp(-x)))