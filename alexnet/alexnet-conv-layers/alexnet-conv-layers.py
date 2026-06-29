import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """
    AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation).
    """
    h_in, n_sample = image.shape[1], image.shape[0]
    h_out = (h_in + 4 - 11)//4 + 1
    output = np.zeros(shape=(n_sample, h_out, h_out, 96))
    return output