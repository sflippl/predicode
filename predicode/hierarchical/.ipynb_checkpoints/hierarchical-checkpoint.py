"""Defines general hierarchical predictive coding models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np
import tensorflow as tf

class NoPredictor(): # This class is required to make the interface consistent pylint:disable=too-few-public-methods
    """Predictor class if no model has been defined so far."""

    def __init__(self):
        pass

    @staticmethod
    def summary():
        """Summary of NoPredictor."""
        return print('(No predictor defined.)')

class NoStatePrediction(): # This class is required to make the interface consistent pylint:disable=too-few-public-methods
    """State Prediction class if no state prediction has been defined so far."""

    def __init__(self):
        pass

    @staticmethod
    def summary():
        """Summary of NoStatePrediction."""
        return print('(No state prediction defined.)')

class Hierarchical(): #pylint:disable=too-many-instance-attributes
    """Defines a hierarchical predictive coding model."""

    n = 1
    def __init__(self, name=None):
        if not name:
            name = 'hierarchical_%d' % (Hierarchical.n, )
            Hierarchical.n += 1
        self.name = name
        self._n_tiers = 0
        self._tiers = []
        self._raw_tiers = []
        self._tier_names = []
        self._predictors = []
        self._state_predictions = []
        self._current_connection = None
        self._regimen = None

    def add_tier(self, shape, name=None,
                 initializer=tf.initializers.GlorotNormal()):
        """Add a tier to the hierarchical model.

        This method adds a tier to the hierarchical model. For now, this is only
        possible on the top, but future implementation might include adding such
        a tier at an arbitrary position -- namely below or above any other tier.

        Args:
            shape: Which shape does the tier have?
            name: Optional name of the tier.
            initializer: How should the tier be initialized?

        Returns:
            The updated hierarchical model to allow chained assignment.

        Raises:
            ValueError: If you do not provide an appropriate name or shape."""
        if not name:
            name = 'tier_%d' % (self._n_tiers, )
        if name in self._tier_names:
            raise ValueError('The name %s has already been used in the model. '
                             'All names should be unique.' % (name, ))
        tier_variable = self._create_tier_variable(shape=shape,
                                                   initializer=initializer,
                                                   name=name)
        self._tiers.append(tier_variable)
        self._raw_tiers.append(tier_variable)
        self._tier_names.append(name)
        if self._n_tiers > 0:
            self._predictors.append(NoPredictor())
            self._state_predictions.append(NoStatePrediction())
        self._n_tiers += 1
        # We want an automatic connection to be chosen -- as long as there is
        # already a possible connection.
        if self._n_tiers > 1:
            self.choose_connection(self._n_tiers-1)
        return self

    def choose_connection(self, tier, lower=True):
        """Choose the default connection to manipulate.

        Most manipulation of the hierarchical predictive coding model occurs by
        manipulating its connections. This operation chooses the connection that
        is currently manipulated.

        Args:
            tier: Choose a tier, either by its name or by its position.
            lower: Should the connection be below (True) or above (False) this
                tier? Default is True.

        Returns:
            The new Hierarchical object."""
        tier = self._get_tier_from_name(tier)
        if lower and tier == 0:
            raise ValueError('There is no connection below tier 0.')
        if not lower and tier == self._n_tiers-1:
            raise ValueError('Since it is the last tier, there is no connection'
                             ' above tier %d.' % (tier, ))
        if lower:
            tier -= 1
        self._current_connection = tier
        return self

    def tier(self, tier):
        """Inspect a particular tier.

        Args:
            tier: Choose a tier, either by its name or by its position.

        Returns:
            a tensor variable or constant.

        Raises:
            ValueError: if the tier does not exist.
            TypeError: if the tier cannot be interpreted as string or integer.
        """
        tier_nr = self._get_tier_from_name(tier)
        _tier = self._tiers[tier_nr]
        return _tier

    def _get_tier_from_name(self, name):
        if isinstance(name, str):
            tier_nr = np.where(
                [name == tier_name for tier_name in self._tier_names]
            )[0]
            assert len(tier_nr) <= 1
            if not tier_nr.size:
                raise ValueError('Tier %s does not exist.' % (name, ))
            return tier_nr[0]
        try:
            tier_nr = int(name)
        except TypeError as e:
            raise TypeError('Name must be string or convertible to integer.')\
                from e
        if tier_nr >= self._n_tiers:
            raise ValueError('Since there are only %d tiers, tier %d does not '
                             'exist.' % (self._n_tiers, tier_nr))
        if tier_nr < 0:
            raise ValueError('Negative tiers do not exist.')
        return tier_nr

    def _create_tier_variable(self, shape, initializer, name=None):
        """Creates the variable corresponding to a particular tier.

        Args:
            shape: Which shape does the variable have?
            initializer: Initial value.
            name: What is the name of the variable?

        Returns:
            A tensor variable or constant."""
        name = '%s_%s' % (self.name, name)
        new_shape = [None] + list(shape)
        initial_value = initializer(shape=[1] + list(shape))
        variable = tf.Variable(initial_value, name=name, dtype=tf.float32,
                               shape=new_shape)
        return variable

    @property
    def predictor(self):
        """Returns the current connection's predictor."""
        return self._predictors[self._current_connection]

    @predictor.setter
    def predictor(self, value):
        if not isinstance(self._predictors[self._current_connection],
                          NoPredictor):
            raise TypeError('Predictor below tier %d has already been assigned.'
                            ' If you truly want to create a new predictor, '
                            'first delete the old model using the method '
                            '"delete_predictor".'\
                            % (self._current_connection+1, ))
        self._predictors[self._current_connection] = value

    def delete_predictor(self):
        """Deletes current connection's predictor."""
        self._predictors[self._current_connection] = NoPredictor()

    @property
    def state_prediction(self):
        """Returns the current connection's state prediction."""
        return self._state_predictions[self._current_connection]

    @state_prediction.setter
    def state_prediction(self, value):
        if not isinstance(self._state_predictions[self._current_connection],
                          NoStatePrediction):
            raise TypeError('State prediction below tier %d has already been '
                            'assigned. If you truly want to create a new state '
                            'prediction, first '
                            ' delete the old state prediction using the method '
                            '"delete_state_prediction".'\
                            % (self._current_connection+1, ))
        self._state_predictions[self._current_connection] = value

    def delete_state_prediction(self):
        """Deletes current connection's state prediction."""
        self._state_predictions[self._current_connection] = NoStatePrediction()

    def summary(self):
        """Provides a summary of the hierarchical predictive coding model."""
        for i in range(self._n_tiers-1, 0, -1):
            print('# Tier %d: %s' % (i, self._tier_names[i]))
            print('## Connecting Predictor')
            self._predictors[i-1].summary()
            print('## Connecting State Prediction')
            self._state_predictions[i-1].summary()
        print('# Tier 0: %s' % (self._tier_names[0], ))

    def train(self, dataset, regimen, metrics=None, batch_size=10000):
        """Train a model on a given dataset.

        This model trains a hierarchical predictive coding model.

        Args:
            data: A State or an object that is interpretable as a state, e. g.
                a numpy array or a Dataset.
            regimen: A training regimen.
            metrics: A list of metrics.
            batch_size: In which batch sizes should training occur? Default is
                10000, as this creates a manageable size, but also means that
                small datasets are estimated together.

        Returns:
            The trained Hierarchical object."""
        self._is_ready()
        predictor_weights = []
        metrics = metrics or []
        for predictor in self._predictors:
            for pred in predictor.trainable_variables:
                predictor_weights.append(pred)
        dataset = self.as_dataset(dataset)
        batches = dataset.batch(batch_size)
        self._tiers = copy.deepcopy(self._raw_tiers)
        while not regimen.end():
            regimen.start_batch()
            for data in batches:
                self._tiers = self._setup_tiers(data)
                @tf.function
                def loss_fun(): # pragma: no cover
                    return self._setup_losses(self._tiers)
                regimen.training_step(loss_fun,
                                      state_variables=self._tiers,
                                      predictor_variables=predictor_weights,
                                      metrics=metrics)
            regimen.finish_batch()
        return self

    def _is_ready(self):
        for predictor, lower_tier, upper_tier in zip(self._predictors,
                                                     self._tier_names[:-1],
                                                     self._tier_names[1:]):
            if isinstance(predictor, NoPredictor):
                raise ValueError('You need to define the predictor between %s '
                                 'and %s.' % (lower_tier, upper_tier))
        for state_pred, lower_tier, upper_tier in zip(self._state_predictions,
                                                      self._tier_names[:-1],
                                                      self._tier_names[1:]):
            if isinstance(state_pred, NoStatePrediction):
                raise ValueError('You need to define the state prediction '
                                 'between %s and %s.' % (lower_tier, upper_tier))

    def _setup_tiers(self, data):
        tiers = self._tiers
        for key, value in data.items():
            tier_nr = self._get_tier_from_name(key)
            tiers[tier_nr] = value
        size = next(iter(data.items()))[1].shape[0]
        for i, tier in enumerate(tiers):
            if isinstance(tier, tf.Variable):
                if tier.shape[0] is None or tier.shape[0] != size:
                    initial_array = tier.numpy()
                    repeats = int(np.ceil(size/initial_array.shape[0]))
                    array_length = np.repeat(initial_array,
                                             repeats=repeats,
                                             axis=0)[:repeats]
                    new_shape = [size] + list(tier.shape[1:])
                    tiers[i] = tf.Variable(array_length,
                                           shape=new_shape,
                                           dtype=tier.dtype,
                                           name=self._tier_names[i])
        return tiers

    @tf.function
    def _setup_losses(self, tiers): # pragma: no cover (is only executed within Tensorflow call.)
        predictions = [
            predictor(tier) for predictor, tier in zip(self._predictors,
                                                       tiers[1:])
        ]
        losses = [
            state_prediction.compute_loss(tier, prediction) for \
            prediction, tier, state_prediction in zip(predictions,
                                                      tiers[:-1],
                                                      self._state_predictions)
        ]
        return [losses, predictions, tiers[:-1]]

    def as_dataset(self, dataset):
        """Parses observations into a full dataset and validates dataset.

        Args:
            dataset: Either a dataset or a numpy array that should  be turned
                into a dataset of observations.

        Returns:
            A tensorflow dataset.

        Raises:
            Value Error: if dataset is neither numpy array nor dictionary
                nor tensorflow datset; if an entry is provided that is not
                a tier of the model or if the shape of a provided tier
                does not conform to the shape of the tier in the model."""
        if isinstance(dataset, np.ndarray):
            dataset = {self._tier_names[0]: dataset}
        if isinstance(dataset, dict):
            dataset = tf.data.Dataset.from_tensor_slices(dataset)
        self._validate_dataset(dataset)
        return dataset

    def _validate_dataset(self, dataset):
        inspect = next(iter(dataset.prefetch(1)))
        if not isinstance(inspect, dict):
            raise ValueError('Dataset must be a dictionary pointing to the '
                             'different tiers of the hierarchical model.')
        for key, value in inspect.items():
            if key not in self._tier_names:
                raise ValueError('%s does not refer to a tiername' % (key, ))
            provided_shape = value.shape
            expected_shape = tf.shape(self.tier(key))[1:]
            if provided_shape != expected_shape:
                raise ValueError('%s does not have the correct shape. '
                                 'Provided shape is %s, but should be %s.'\
                                 % (key, provided_shape, expected_shape))
