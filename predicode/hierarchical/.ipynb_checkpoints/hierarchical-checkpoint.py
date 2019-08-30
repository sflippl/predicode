"""Defines general hierarchical predictive coding models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

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

class Hierarchical():
    """Defines a hierarchical predictive coding model."""

    n = 1
    def __init__(self, name=None):
        if not name:
            name = 'hierarchical_%d' % (Hierarchical.n, )
            Hierarchical.n += 1
        self.name = name
        self._n_tiers = 0
        self._tiers = []
        self._predictors = []
        self._state_predictions = []
        self._current_connection = None

    def add_tier(self, shape, name=None, initializer=None):
        """Add a tier to the hierarchical model.

        This method adds a tier to the hierarchical model. For now, this is only
        possible on the top, but future implementation might include adding such
        a tier at an arbitrary position -- namely below or above any other tier.

        Args:
            shape: Which shape does the tier have?
            name: Optional name of the tier.
            initializer: How should the tier be initialized? Will be handled by
                ~:fun:`predicode.init`

        Returns:
            The updated hierarchical model to allow chained assignment.

        Raises:
            ValueError: If you do not provide an appropriate shape."""
        if not name:
            name = 'tier_%d' % (self._n_tiers, )
        if name in self._tier_names:
            raise ValueError('The name %s has already been used in the model. '
                             'All names should be unique.' % (name, ))
        tier_variable = self._create_tier_variable(shape=shape,
                                                   initializer=initializer,
                                                   name=name)
        self._tiers.append((name, tier_variable))
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
                tier? Default is True."""
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

    @property
    def _tier_names(self):
        """Returns the currently defined tier names."""
        names = [name for name, __ in self._tiers]
        return names

    @classmethod
    def _create_tier_variable(cls, shape, initializer, name):
        """Creates the variable corresponding to a particular tier.

        Currently a dummy function."""
        return 'DUMMY'

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
            print('## Connection State Prediction')
            self._state_predictions[i-1].summary()
        print('# Tier 0: %s' % (self._tier_names[0], ))
