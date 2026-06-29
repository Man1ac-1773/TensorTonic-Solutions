import numpy as np

def dropout(x: np.ndarray, p: float = 0.5, training: bool = True, mask: np.ndarray = None) -> np.ndarray:
    """
    Apply inverted dropout. If mask is provided, use it; otherwise generate one.
    """
    if training:
        if mask is not None:
            return x * mask * (1/(1-p))
        else:
            mask = np.random.rand(*x.shape) >= p
            return x * mask * (1/(1-p))
    else : return x 
    pass