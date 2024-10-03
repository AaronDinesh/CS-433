# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""

    A = np.empty((x.shape[0], degree+1))
    for i in range(x.shape[0]):
        for j in range(degree+1):
            A[i, j] = np.pow(x[i], j)
    return A