"""Tier connections computing top-down predictions.
"""

import tensorflow as tf
import tensorflow.keras as keras

from predicode.connections.connections import TierConnection
import predicode.prediction_errors as prediction_errors

class TopDownPrediction(TierConnection):
    """Tier connections computing top-down predictions.

    Top-down predictions compute a prediction of the lower tier based on the
    values of the upper tier, determine a prediction error, and define a loss.

    The states of the upper tier are inferred by minimizing that loss.

    Args:
        model: Predictive Keras model, taking as input the upper tier and
            predicting the lower tier.
        prediction_error: String (specifying a prediction error) or function
            computing a prediction error.
        loss: String (specifying a keras loss) or function computing a loss.
            Compares the prediction error to zero.

    Raises:
        ValueError: If the identifier of the prediction_error or the loss does
            not yield an appropriate result.
    """

    def __init__(self, model,
                 prediction_error='difference',
                 loss='mean_squared_error'):
        super().__init__()
        self.model = model
        self._prediction_error = prediction_errors.get(prediction_error)
        self.loss = keras.losses.get(loss)

    @tf.function
    def predict(self, upper_tier, lower_tier=None): # pragma: no cover
        """Uses the model to predict the lower tier.
        """
        return self.model(upper_tier)

    @tf.function
    def prediction_error(self, upper_tier, lower_tier, predictions): # pragma: no cover
        """Computes the prediction error between the lower tier and its
        prediction specified in the initialization.
        """
        return self._prediction_error(lower_tier, predictions)

    @tf.function
    def compute_loss(self, upper_tier, lower_tier, predictions): # pragma: no cover
        """Computes the loss specified in the initialization.
        """
        return self.loss(lower_tier, predictions)

    def summary(self):
        print('Top-down prediction.')
        print('## Predictive model')
        self.model.summary()
        print('## Prediction error')
        print(self._prediction_error)
        print('## Loss function')
        print(self.loss)

    @property
    def predictor_variables(self):
        """Returns a list of all trainable variables in the model.
        """
        return self.model.trainable_variables

class TopDownSequential(TopDownPrediction):
    """Top-down prediction computed by a sequential model.

    Makes add method directly available for an easier interface."""

    def __init__(self, layers=None, name=None,
                 prediction_error='difference',
                 loss='mean_squared_error'):
        super().__init__(
            model=keras.Sequential(layers=layers, name=name),
            prediction_error=prediction_error,
            loss=loss
        )

    def add(self, layer):
        """Add a layer to the sequential model.
        """
        self.model.add(layer)
