"""Defines general hierarchical predictive coding models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import predicode.regimens as regimens
import predicode.connections as connections

class Hierarchical(): #pylint:disable=too-many-instance-attributes
    """Defines a hierarchical predictive coding model.

    Args:
        tiers: List of tier shapes, bottom-up.
        name: Name of the hierarchical model.
    """

    n = 1
    def __init__(self, tiers = [], name=None):
        if not name:
            name = 'hierarchical_%d' % (Hierarchical.n, )
            Hierarchical.n += 1
        self.name = name
        self._n_tiers = 0
        self._tiers = []
        self._predictors = []
        self._prediction_errors = []
        self._raw_tiers = []
        self._tier_names = []
        self._connections = []
        self._current_connection = None
        self._regimen = None
        for tier in tiers:
            self.add_tier(tier)

    def add_tier(self, shape, name=None,
                 initializer=keras.initializers.GlorotNormal()):
        """Add a tier to the hierarchical model.

        This method adds a tier to the hierarchical model, above the current top
        tier.

        Args:
            shape: Which shape does the tier have?
            name: Optional name of the tier.
            initializer: How should the tier be initialized?

        Returns:
            The updated hierarchical model to allow chained assignment.

        Raises:
            ValueError: If shape or name are not appropriate.
        """
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
        self._predictors.append(tier_variable)
        self._prediction_errors.append(tier_variable)
        self._tier_names.append(name)
        if self._n_tiers > 0:
            self._connections.append(connections.NoTierConnection())
        self._n_tiers += 1
        # We want an automatic connection to be chosen -- as long as there is
        # already a possible connection.
        if self._n_tiers > 1:
            self.activate_connection(self._n_tiers-1)
        return self

    def activate_connection(self, tier, lower=True):
        """Activate a tier connection to further modify.

        Most manipulation of the hierarchical predictive coding model occurs by
        manipulating its tier connections. This operation chooses the connection
        that is currently affected by these changes.

        Args:
            tier: Choose a tier, either by its name or by its position.
            lower: Should the connection be below (True) or above (False) this
                tier? Default is True.

        Returns:
            The new Hierarchical object.

        Raises:
            ValueError: if the activated connection does not exist.
        """
        tier = self._get_tier_from_name(tier)
        if lower and tier == 0:
            raise ValueError('There is no connection below tier 0.')
        if not lower and tier == self._n_tiers-1:
            raise ValueError('Since it is the last tier, there is no connection'
                             ' above tier %d.' % (tier, ))
        if lower:
            tier -= 1
        self._current_connection = tier
        print('Active connection: %s -> %s' % (self._tier_names[tier+1],
                                               self._tier_names[tier]))
        return self

    def tier(self, tier):
        """Inspect a particular tier.

        Args:
            tier: Choose a tier, either by its name or by its position.

        Returns:
            a tensor variable or constant.

        Raises:
            ValueError: if the tier does not exist.
            TypeError: if the tier cannot be interpreted as a string or integer.
        """
        tier_nr = self._get_tier_from_name(tier)
        _tier = self._tiers[tier_nr]
        return _tier

    def prediction(self, tier):
        """Inspect the prediction for a tier.

        Args:
            tier: Choose a tier, either by its name or by its position.

        Returns:
            A Tensor variable or constant.

        Raises:
            Value Error: if the tier does not exist.
            TypeError: if the tier cannot be interpreted as a string or integer.
        """
        tier_nr = self._get_tier_from_name(tier)
        _prediction = self._connections[tier_nr].predict(
            self._tiers[tier_nr+1], self._tiers[tier_nr]
        )
        return _prediction

    def prediction_error(self, tier):
        """Inspect the prediction error of a tier.

        Args:
            tier: Choose a tier, either by its name or by its position.

        Returns:
            A Tensor variable or constant.

        Raises:
            Value Error: if the tier does not exist.
            TypeError: if the tier cannot be interpreted as a string or integer.
        """
        tier_nr = self._get_tier_from_name(tier)
        _prediction_error = self._connections[tier_nr].prediction_error(
            self._tiers[tier_nr+1], self._tiers[tier_nr], self.prediction(tier)
        )
        return _prediction_error

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
            A tensor variable or constant.
        """
        name = '%s_%s' % (self.name, name)
        new_shape = [None] + list(shape)
        initial_value = initializer(shape=[1] + list(shape))
        variable = tf.Variable(initial_value, name=name, dtype=tf.float32,
                               shape=new_shape)
        return variable

    @property
    def connection(self):
        """The activated tier connection."""
        return self._connections[self._current_connection]

    @connection.setter
    def connection(self, value):
        if not isinstance(self._connections[self._current_connection],
                          connections.NoTierConnection):
            raise TypeError('Predictor below tier %d has already been assigned.'
                            ' If you truly want to create a new predictor, '
                            'first delete the old model using the method '
                            '"delete_predictor()".'\
                            % (self._current_connection+1, ))
        self._connections[self._current_connection] = value

    def delete_connections(self):
        """Deletes the activated tier connection."""
        self._connections[self._current_connection] =\
            connections.NoTierConnection()

    def summary(self):
        """Provides a summary of the model architecture.
        """
        for i in range(self._n_tiers-1, 0, -1):
            print('# Tier %d: %s\n' % (i, self._tier_names[i]))
            print('# Connection: %s -> %s' % (self._tier_names[i],
                                              self._tier_names[i-1]))
            self._connections[i-1].summary()
            print()
        print('# Tier 0: %s' % (self._tier_names[0], ))

    def compile(self, optimizer, metrics=[]):
        """Configures the model for training.

        Args:
            optimizer: String (name of optimizer), optimizer, or optimizer
                regimen.
            metrics: List of metrics for the model to be evaluated during
                training.

        Returns:
            Compiled model.
        """
        self._optimizer = regimens.get(optimizer)
        metrics = metrics or []
        metrics = [keras.metrics.get(metric) for metric in metrics]
        self._metrics = metrics
        self._is_compiled = True

    def train(self, dataset, batch_size=10000, logdir=None):
        """Train a model on a given dataset.

        This model trains a hierarchical predictive coding model.

        Args:
            data: A numpy array or a Tensorflow Dataset.
            batch_size: In which batch sizes should training occur? Default is
                10000, as this creates a manageable size, but also means that
                small datasets are estimated together.

        Returns:
            The trained Hierarchical object.
        """
        self._is_ready()
        predictor_weights = []
        for connection in self._connections:
            predictor_weights += connection.predictor_variables
        dataset = self.as_dataset(dataset)
        batches = dataset.batch(batch_size)
        self._tiers = copy.deepcopy(self._raw_tiers)
        self._predictions = copy.deepcopy(self._raw_tiers[:-1])
        self._prediction_errors = copy.deepcopy(self._raw_tiers[:-1])
        optimizer = copy.deepcopy(self._optimizer)
        if logdir:
            summary_writer = tf.summary.create_file_writer(logdir).as_default()
        while not optimizer.end():
            optimizer.start_batch()
            for data in batches:
                self._tiers = self._setup_tiers(data)
                @tf.function
                def loss_fun(): # pragma: no cover
                    return self._setup_losses(self._tiers)
                optimizer.training_step(loss_fun,
                                        state_variables=self._tiers,
                                        predictor_variables=predictor_weights,
                                        metrics=self._metrics)
            optimizer.finish_batch()
        if logdir:
            del summary_writer
        return self

    def _is_ready(self):
        for connection, lower_tier, upper_tier in zip(self._connections,
                                                      self._tier_names[:-1],
                                                      self._tier_names[1:]):
            if isinstance(connection, connections.NoTierConnection):
                raise ValueError('You need to define the tier connection '
                                 'between %s and %s.'\
                                 % (upper_tier, lower_tier))
        if not self._is_compiled:
            raise ValueError('You need to compile the model using the '
                             '"compile()" method before training.')

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
    def _setup_losses(self, tiers): # pragma: no cover (is only executed within Tensorflow call)
        predictions = []
        for i, connection in enumerate(self._connections):
            predictions.append(connection.predict(tiers[i+1], tiers[i]))
        losses = []
        for i, connection in enumerate(self._connections):
            losses.append(connection.compute_loss(tiers[i+1],
                                                  tiers[i],
                                                  predictions[i]))
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
