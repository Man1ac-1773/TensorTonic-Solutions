import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, Y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    w = np.zeros((X.shape[1],))
    b = 0.0
    for _ in range(steps):
        a = _sigmoid(X @ w + b)
        loss = -np.mean(Y * np.log(a) + (1-Y)*np.log(1-a))
        dw = (X.T @ (a - Y))/X.shape[0]
        db = np.mean(a-Y)
        w -= lr * dw
        b -= lr * db

    return (w,b)