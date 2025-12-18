import numpy as np

def binary_cross_entropy(y,y_hat):
    """
    Binary cross entropy loss
    """
    eps = 1e-15
    y_hat = np.clip(y_hat,eps,1-eps)
    loss = -((y * (np.log(y_hat))) + ((1-y) * (np.log(1-y_hat))))
    return np.mean(loss)