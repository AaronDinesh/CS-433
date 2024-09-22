# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_loss_mse(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    #MSE loss is (1/2*N)(e')(e)
    #where e = y-Xw
    e = y - np.dot(tx, w)
    return (1/(2*y.shape[0])) * (e.T @ e)

def compute_loss_mae(y, tx, w):
    """Calculate the Loss using MAE
    
    Args:
        y: shape=(N, )
        tx: shape=(N, 2)
        w: shape=(2, ). The vector of model parameters

    Returns:
        The value of the loss, corresponding to the input parameters w
    """
    e = y - tx @ w
    return (1/y.shape[0]) * np.sum(np.abs(e))