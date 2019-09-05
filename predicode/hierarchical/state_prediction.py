"""The class State prediction connects the predicted with the inferred state."""

import tensorflow.keras as keras

class StatePrediction():
    """A state prediction connects a predicted with an inferred state.

    Args:
        loss: A tensor function that computes the loss driving state and weight
            inference."""

    def __init__(self, loss=keras.losses.mean_squared_error):
        self.loss = loss

    def compute_loss(self, state, prediction):
        """Compute the loss that drives the state and weight inference."""
        _loss = self.loss(state, prediction)
        return _loss

    def summary(self):
        """Summary of StatePrediction. Used in the summary of Hierarchical."""
        print('Loss-driven state prediction.')
        print('Loss function: %s' % (self.loss.__str__()))
