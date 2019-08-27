"""Interface to the minimal model, linear predictive coding with a single hidden
layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import numpy as np
import tqdm

from predicode.hierarchical.weight_init import weight_init

class MinimalHierarchicalModel(): # pylint:disable=too-many-instance-attributes
    """Interface to the minimal hierarchical model.

    The minimal hierarchical model is given by a linear predictive coding model
    with one densely connected, linear layer.

    Args:
        input_data: A numpy array, where the columns correspond to the
            dimensions and the rows correspond to the number of observations.
        input_dimensions: Input dimensions.
        n_observations: Number of observations.
        latent_dimensions: Dimensions of the latent layer.
        weights: An object that can be fed through
            :fun:`~predicode.weight_init`.
        use_bias: Should the model use an additional bias layer?"""

    @staticmethod
    def _state_estimator_predict(mode, flipped_graph):
        return tf.estimator.EstimatorSpec(mode, predictions=flipped_graph)

    @staticmethod
    def _state_estimator_loss(flipped_graph, labels):
        loss = tf.losses.mean_squared_error(flipped_graph, labels)
        return loss

    @classmethod
    def _state_estimator_eval(cls, mode, flipped_graph, labels):
        loss = cls._state_estimator_loss(flipped_graph, labels)
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    def _state_estimator_train(self, mode, flipped_graph, labels,
                               learning_rate=1):
        loss = self._state_estimator_loss(flipped_graph, labels)
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate
        )
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    def _state_estimator(self, features, labels, mode, params):
        flipped_graph = tf.feature_column.input_layer(features,
                                                      params['latent_weights'])
        flipped_graph = tf.layers.dense(flipped_graph,
                                        units=params['n_observations'],
                                        use_bias=params['use_bias'])

        if mode == tf.estimator.ModeKeys.PREDICT:
            return self._state_estimator_predict(
                mode, flipped_graph
            )

        if mode == tf.estimator.ModeKeys.EVAL:
            return self._state_estimator_eval(
                mode, flipped_graph, labels
            )

        assert mode == tf.estimator.ModeKeys.TRAIN

        return self._state_estimator_train(
            mode, flipped_graph, labels,
            learning_rate=params['state_learning_rate']
        )

    def __init__(self, input_data, weights='pca', latent_dimensions=None, #pylint:disable=too-many-arguments
                 use_bias=False, state_learning_rate=1,
                 **kwargs):
        self.input_data = input_data
        self.input_dimensions = input_data.shape[1]
        self.n_observations = input_data.shape[0]
        self.latent_dimensions = latent_dimensions

        self._weights = weight_init(weights,
                                    input_data=input_data,
                                    latent_dimensions=latent_dimensions,
                                    input_dimensions=self.input_dimensions)
        self._dct_weights = {
            "theta1_%d"%(i): self.weights[:, i] for i in range(
                self.weights.shape[1]
            )
        }

        assert self.input_dimensions == self._weights.shape[0]
        assert self.latent_dimensions == self._weights.shape[1]

        kwargs['use_bias'] = use_bias
        kwargs['n_observations'] = self.n_observations
        kwargs['latent_weights'] = [
            tf.feature_column.numeric_column(
                key=key, dtype=tf.float64
            ) for key in self._dct_weights.keys()
        ]
        kwargs['state_learning_rate'] = state_learning_rate

        self._kwargs = kwargs

        self._state = tf.estimator.Estimator(
            model_fn=self._state_estimator,
            params=kwargs
        )
        self._state.config.replace(save_summary_steps=10)

        self.activate('state')

    @staticmethod
    def _validate_what(what):
        if what not in ['state', 'weight']:
            raise ValueError(
                'You must either activate "state" or "weight" estimation.'
            )
        return what

    def activate(self, what):
        """Activate state or weight estimation.

        Args:
            what: 'state' or 'weight'."""
        what = self._validate_what(what)
        self.what = what

    def train(self, steps=None, max_steps=None, learning_rate=None, **kwargs):
        """Train the network.

        Args:
            max_steps: Number of steps."""
        if self.what == 'state':
            if learning_rate is not None:
                if learning_rate != self._kwargs['state_learning_rate']:
                    self._kwargs['state_learning_rate'] = learning_rate
                    self._state = tf.estimator.Estimator(
                        model_fn=self._state_estimator,
                        params=self._kwargs
                    )

            return self._state.train(
                input_fn=lambda: (self._dct_weights, self.input_data.T),
                steps=steps,
                max_steps=max_steps,
                **kwargs
            )
        raise NotImplementedError()

    def evaluate(self, **kwargs):
        """Evaluate network.

        Args:
            kwargs: Additional parameters for tf.Estimator.evaluate().

        Returns:
            Statistics of the trained model."""
        if self.what == 'state':
            return self._state.evaluate(
                input_fn=lambda: (self._dct_weights, self.input_data.T),
                steps=1,
                **kwargs
            )
        raise NotImplementedError()

    def predict(self, **kwargs):
        """Predict the input data according to the model.

        Args:
            kwargs: Additional parameters for tf.Estimator.predict().

        Returns:
            Prediction as a two-dimensional numpy array with the same form as
            the input data."""
        if self.what == 'state':
            generator = self._state.predict(
                input_fn=lambda: (self._dct_weights, self.input_data.T),
                **kwargs)
            prediction_lst = [
                next(generator) for i in range(self.input_dimensions)
            ]
            prediction = np.array(prediction_lst).T
            return prediction
        raise NotImplementedError()

    def learning_curve(self, steps=1e4, resolution=1, learning_rate=None,
                       **kwargs):
        """Get the learning curve.

        Args:
            steps: For how many steps would you like the learning curve?
            resolution: In what resolution (in steps) would you like to
                determine the learning curve?
            learning_rate: You can optionally specify a learning rate.
            kwargs: Other arguments.

        Returns:
            Three dimensional numpy array: observations x latent dimensions x
            steps."""
        tsteps = math.ceil(steps/resolution)
        steps_per_tstep = steps/tsteps
        _learning_curve = np.empty(dtype=np.float64,
                                   shape=(self.n_observations,
                                          self.latent_dimensions,
                                          tsteps))
        prev_verbosity = tf.logging.get_verbosity()
        tf.logging.set_verbosity(tf.logging.ERROR)
        for i in tqdm.trange(tsteps):
            self.train(steps=steps_per_tstep, learning_rate=learning_rate,
                       **kwargs)
            _learning_curve[:, :, i] = self.latent_values
        tf.logging.set_verbosity(prev_verbosity)
        return _learning_curve

    @property
    def latent_values(self):
        """Get the latent values.

        Returns:
            The latent variables for the input data as a two-dimensional numpy
            array. Each observation is one row and each latent dimension one
            column."""
        return self._state.get_variable_value('dense/kernel').T

    @property
    def weights(self):
        """Get the weights.

        Returns:
            The weights of the minimal model with the rotation that returns one
            input dimension per row."""
        return self._weights
