"""Specifies weight initialization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import sklearn.decomposition as decomp
import scipy.stats as stats

def _validate_latent_dimensions(latent_dimensions, input_dimensions):
    if latent_dimensions is None:
        latent_dimensions = input_dimensions
    if latent_dimensions > input_dimensions:
        raise ValueError(
            ('Latent dimensions (%d) must be less '
             'than input dimensions (%d)') % (latent_dimensions,
                                              input_dimensions)
        )
    return latent_dimensions

def weight_init_pca():
    """Initialize weights as the first principal rotations of a PCA.

    Returns:
        Function that initializes weights.

        Args:
            latent_dimensions: Number of latent dimensions.
            input_data: Input data as a two dimensional numpy array. One sample
                per row and one dimension per column.

        Returns:
            Two dimensional numpy array with one principal rotation per
            column."""
    def initialize(latent_dimensions=None, input_data=None, **kwargs): # kwargs required to make arbitrary passing of one of the weight_init_* functions possible pylint:disable=unused-argument
        weights = decomp.PCA(
            n_components=latent_dimensions
        ).fit(input_data).components_.T
        return weights

    return initialize

def init_random(method='orthogonal'):
    """Initialize value randomly.

    Args:
        method: Which method should be used for the random initialization?
           Default is orthogonal.

           orthogonal: Creates a random orthogonal matrix.

    Returns:
        Function that initializes a matrix.

        Args:
            rows: Number of rows.
            columns: Number of columns.

        Returns:
            Two dimensional numpy array.

    Raises:
        NotImplementedError: If method is not implemented."""
    if method == 'orthogonal':
        def initialize(rows, columns,
                       **kwargs): # kwargs required to make arbitrary passing of one of the init_* functions possible pylint:disable=unused-argument
            """Initialize weights according to the specified method.

            Args:
                rows: Number of rows.
                columns: Number of latent dimensions to be extracted. Must
                    be less than the input dimensions.

            Returns:
                Two dimensional numpy array."""
            min_dim = min(rows, columns)
            max_dim = max(rows, columns)
            transpose = rows < columns
            matrix = stats.ortho_group.rvs(
                max_dim
            )[:, range(min_dim)]
            if transpose:
                matrix = matrix.T
            return matrix

    else:
        raise NotImplementedError('Method %s is not implemented.' % (method, ))

    return initialize

def weight_init_random(method='orthogonal'):
    """Initialize weights randomly.

    Args:
        method: Which method should be used for the random initialization?
           Default is orthogonal.

           orthogonal: Creates a random orthogonal matrix.

    Returns:
        Function that initializes a random rotation.

        Args:
            input_dimensions: Dimensions of the lower tier.
            latent_dimensions: Dimensions of the upper tier.

        Returns:
            Two dimensional numpy array.

    Raises:
        NotImplementedError: If method is not implemented."""
    _initialize = init_random(method)
    def initialize(input_dimensions=None, latent_dimensions=None,
                   **kwargs):
        latent_dimensions = _validate_latent_dimensions(latent_dimensions,
                                                        input_dimensions)
        return _initialize(rows=input_dimensions,
                           columns=latent_dimensions,
                           **kwargs)
    return initialize

def init(char, **kwargs):
    """Initialize matrix according to method.

    Args:
        char: Method, function, or initialized array.
        rows: Number of rows.
        columns: Number of columns.

    Returns:
        A two dimensional numpy array.

    Raises:
        NotImplementedError: If the method specified by char is not implemented.
        """
    if isinstance(char, np.ndarray):
        return char
    if isinstance(char, collections.Callable):
        return char(**kwargs)
    if char == 'random':
        return init_random('orthogonal')(**kwargs)
    raise NotImplementedError('Method %s is not implemented.' % (char, ))

def weight_init(char, **kwargs):
    """Initialize weights according to method.

    Initialize weights according to the specified method and the additional
    input arguments as a shortcut for `pc.weight_init_*(...)(...)`.

    If passed a numpy array or a function, it returns the object.

    Args:
        input_dimensions: Number of input dimensions.
        latent_dimensions: Number of latent dimensions. Must be less than the
            input dimensions.
        input_data: Input data as a two dimensional numpy array. One sample per
            row and one dimension per column.
        char: Which method?

    Returns:
        Two dimensional numpy array with one rotation per row."""
    if isinstance(char, np.ndarray):
        return char
    if isinstance(char, collections.Callable):
        return char(**kwargs)
    if char == 'pca':
        return weight_init_pca()(**kwargs)
    if char == 'random':
        return weight_init_random('orthogonal')(**kwargs)
    raise NotImplementedError('Method %s is not implemented.' % (char, ))
