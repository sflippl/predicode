"""Default tier connections.
"""

import abc

class TierConnection(abc.ABC):
    """General tier connections.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def predict(self, upper_tier, lower_tier):
        """Computes the prediction.

        Args:
            upper_tier: The tensor variable or constant corresponding to the
                states of the upper tier.
            lower_tier: The tensor variable or constant corresponding to the
                states of the lower tier.
        """
        pass

    @abc.abstractmethod
    def prediction_error(self, upper_tier, lower_tier, predictions):
        """Computes the prediction error.

        Args:
            upper_tier: The tensor variable or constant corresponding to the
                states of the upper tier.
            lower_tier: The tensor variable or constant corresponding to the
                states of the lower tier.
            predictions: The predictions compute by the connection.
        """
        pass

    @abc.abstractmethod
    def compute_loss(self, upper_tier, lower_tier, predictions):
        """Computes the loss based on prediction error.

        Args:
            prediction_error: The prediction error computed by the connection.
        """
        pass

    @abc.abstractmethod
    def summary(self):
        """Provides a summary of the tier connection.
        """
        pass

    @property
    def predictor_variables(self):
        """Returns the trainable predictor variables.
        """
        return []

class NoTierConnection(TierConnection): # This class is required to make the interface consistent pylint:disable=too-few-public-methods
    """Undefined tier connection.
    """

    def __init__(self):
        super().__init__()

    def predict(self, upper_tier, lower_tier):
        raise ValueError('NoTierConnection is only a placeholder. Define a '
                         'proper tier connection before using its methods.')

    def prediction_error(self, upper_tier, lower_tier, predictions):
        raise ValueError('NoTierConnection is only a placeholder. Define a '
                         'proper tier connection before using its methods.')

    def compute_loss(self, upper_tier, lower_tier, predictions):
        raise ValueError('NoTierConnection is only a placeholder. Define a '
                         'proper tier connection before using its methods.')

    def summary(self):
        return print('(No tier connection defined.)')
