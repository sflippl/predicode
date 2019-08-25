"""Create artificial data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.stats as stats

def decaying_multi_normal(dimensions, size, alpha=1):
    """Create multinormal data with exponentially decaying principal components.

    Creates a two-dimensional numpy array such that a PCA yields principal 
    components with exponentially decaying variance.
    
    Args:
        dimensions: How many dimensions should the data have?
        size: How many samples should be drawn?
        alpha: The exponential decay constant: how fast should the variance
            of the principal components decay (default: 1)? Only non-negative 
            values are allowed.
    
    Returns:
        A two-dimensional numpy array with one sample per row and one dimension
        per column.
    
    Raises:
        ValueError: alpha is negative.
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    pc_variance = np.exp(-alpha*np.array(range(dimensions)))
    rand_ortho = stats.ortho_group.rvs(dimensions)
    rand_normal = np.random.normal(scale=pc_variance, size=(size, dimensions))
    rand_input = np.matmul(rand_normal, rand_ortho)
    return rand_input
