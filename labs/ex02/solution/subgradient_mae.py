import numpy as np

def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    
    e = y - tx @ w
    e_signs = np.sign(e)
    tx1 = np.multiply(tx, e_signs[:, np.newaxis])
    x_sum = np.sum(tx1, axis=0)
    gradient = (-1/y.shape[0]) * x_sum

    assert gradient.shape[0] == w.shape[0] 

    return gradient


