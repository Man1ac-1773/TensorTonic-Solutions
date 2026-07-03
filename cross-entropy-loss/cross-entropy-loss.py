import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    s = 0
    for i in range(len(y_true)):
        s += np.log(y_pred[i][y_true[i]])
    return -s/len(y_true)
    pass