"""Custom prediction errors."""

import collections

import tensorflow as tf

@tf.function
def difference(states, predictions):
    """Computes the prediction error as difference between state and prediction.

    Args:
        states: A tensor variable or constant specifying the states of a tier.
        predictions: A tensor variable or constant specifying the predictions of
            the same tier.

    Returns:
        The prediction error between the two.
    """
    return states - predictions

def get(identifier):
    """Retrieves a prediction error instance.

    Args:
        identifier: Either a string (specifying a prediction error function) or
            a function computing the prediction error with states and
            predictions as input.

    Returns:
        Prediction error instance.

    Raises:
        ValueError: If identifier does not specify a prediction error."""
    if isinstance(identifier, collections.Callable):
        return identifier
    if identifier == 'difference':
        return difference
    raise ValueError('Identifier {} cannot be identified.'.\
                     format((identifier, )))
