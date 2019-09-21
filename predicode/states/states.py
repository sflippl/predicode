"""States provide the different states of the tiers in Hierarchical.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import copy

import numpy as np
import h5py
import tensorflow as tf
import tensorflow.keras as keras

def _key_test(key, dct):
    if key not in dct:
        raise ValueError('Key {} has not been found in the container {}'.\
                         format((key, dct)))

def _copy_operation(obj, dct, strict=False, pure_states=None):
    if pure_states is None:
        pure_states = obj._pure_states()
    if isinstance(dct, (DictWithAttrs, h5py.Group)):
        for key, value in dct.attrs.items():
            if strict:
                _key_test(key, obj.attrs)
            obj.attrs[key] = value
    for key, value in dct.items():
        if strict:
            _key_test(key, dct)
        if key not in dct:
            dct._create_group(key)
        if not isinstance(value, (dict, h5py.Group)):
            if pure_states:
                raise ValueError('You can not set values in {}.'.\
                                 format((obj, )))
            obj[key] = value
        else:
            obj._create_group(key)
            _copy_operation(obj[key], value,
                            strict=strict, pure_states=pure_states)

class DictWithAttrs(dict):
    """A dictionary with the attribute attrs to make it more compatible with
    hdf5 objects.
    """

    def __init__(self, dct=None, attrs=None):
        dct = dct or {}
        super().__init__(dct)
        self.attrs = attrs or {}

class States(DictWithAttrs, abc.ABC):
    """States specify the schematics of the relevant variables.
    """

    def __init__(self, dct=None, attrs=None):
        dct = dct or {}
        self._validate_dct(dct)
        super().__init__(dct, attrs)

    @classmethod
    def _validate_dct(cls, dct):
        pass

    @classmethod
    def _state_values(cls, **kwds):
        return StateValues(**kwds)

    @classmethod
    def _state_tensors(cls, **kwds):
        return StateTensors(**kwds)

    @classmethod
    def _hdf5(cls, **kwds):
        return HDF5States(**kwds)

    @classmethod
    def _pure_states(cls):
        return True

    @abc.abstractmethod
    def set_values(self, values=None, **kwds):
        """Set values for the given state schematic.

        Returns:
            A StateValues object."""

    def _create_group(self, name):
        dict.__setitem__(self, name, DictWithAttrs())

    def to_hdf5(self, name, **kwds):
        """Converts the state object to hdf5 object and returns the file handle.

        Args:
            name: Name of the file.
            kwds: Remaining arguments to h5py.File.__init__.

        Returns:
            pc.HDF5State object.
        """
        file = self._hdf5(name=name, **kwds)
        _copy_operation(file, self)
        return file

class HDF5Backend(h5py.File):
    """The HDF5 backend allows handling of different states from within an hdf5
    file.
    """

    def __init__(self, name, dct=None, **kwds):
        dct = dct or DictWithAttrs()
        super().__init__(name=name, **kwds)
        _copy_operation(self, dct)

    def _create_group(self, name):
        super().create_group(name)

class HDF5States(HDF5Backend, States):
    """Hdf5 states allow handling of state schemata within an hdf5 file.
    """

    @classmethod
    def _state_values(cls, **kwds):
        return HDF5StateValues(**kwds)

    def __init__(self, name, dct=None, attrs=None, **kwds):
        states = States(dct=dct, attrs=attrs)
        super().__init__(name=name, dct=states, **kwds)

class StateValues(States):
    """State values specify possible values of the relevant variables.
    """

    def __init__(self, dct=None, attrs=None):
        super().__init__(dct=dct, attrs=attrs)

    @classmethod
    def _pure_states(cls):
        return False

    @abc.abstractmethod
    def get_tensors(self):
        """Get StateTensors object for the specified values.

        Returns:
            A StateTensors object."""

class HDF5StateValues(HDF5Backend, StateValues):
    """HDF5 State Values allow handling the values from within an hdf5 file.
    """

    def __init__(self, name, dct=None, attrs=None, **kwds):
        state_values = self._state_values(dct=dct, attrs=attrs)
        super().__init__(name=name, dct=state_values, **kwds)

class StateTensors(States):
    """State tensors specify the tensors, which will be handled by the
    estimator.
    """

    def __init__(self, dct=None, attrs=None):
        super().__init__(dct=dct, attrs=attrs)

    @classmethod
    def _pure_states(cls):
        return False

class HierarchicalStates(States):
    """States specify the states of the relevant variables in the different
    tiers.
    """

    def __init__(self, dct=None, attrs=None, order=None):
        super().__init__(dct=dct, attrs=attrs)
        order = order or []
        self.attrs['order'] = np.array(order, dtype=bytes)

    @classmethod
    def _validate_dct(cls, dct):
        for key, value in dct.items():
            for key_2, value_2 in value.items():
                if isinstance(value_2, (dict, h5py.Group)):
                    raise ValueError('Provided dictionary-like object has more '
                                     'than two specification levels.')

    @classmethod
    def _state_values(cls, **kwds):
        return HierarchicalStateValues(**kwds)

    @classmethod
    def _state_tensors(cls, **kwds):
        return HierarchicalStateTensors(**kwds)

    @classmethod
    def _hdf5(cls, **kwds):
        return HDF5HierarchicalStates(**kwds)

    @property
    def n_tiers(self):
        """The number of tiers.
        """
        return self.order.shape[0]

    @property
    def order(self):
        """The order of the tiers as a list.
        """
        return self.attrs['order'].astype(str)

    def __getitem__(self, index):
        """Get a particular state given tier and a name.

        Args:
            index: The tier may be specified by
                a tier number counting bottom-up, where negative values count
                top-down, or the tier name.

        Returns:
            A numpy array.
        """
        index = self._get_tier_name(index)
        return super().__getitem__(index)

    def set_state(self, tier, state, value):
        raise NotImplementedError('HierarchicalStates cannot obtain values.')

    def _get_tier_name(self, tier):
        """Get the name of a tier.

        Args:
            tier: Either the tier number counting bottom-up, where negative
                values count top-down, or the tier name.

        Returns:
            A string that corresponds to a tier name.
        """
        if isinstance(tier, (str, )):
            return tier
        return self.attrs['order'].astype(str)[tier]

    def add_tier(self, shape, name=None):
        """Adds an empty tier.

        Args:
            shape: Specify the shape of the tier.
            tier_name: Specify the tier name. By default, tier names are given
                as tier_0, tier_1, and so on."""
        name = name or 'tier_%d' % (self.n_tiers, )
        if name in self.order.astype(str):
            raise ValueError('Tier name %s is already taken.' % (name, ))
        self.attrs['order'] = np.append(self.attrs['order'], [name])\
                                .astype(bytes)
        self._create_group(name)
        self[name].attrs['shape'] = shape

    def set_values(self, values=None, **kwds):
        state_values = self._state_values(dct=copy.deepcopy(dict(self)),
                                          attrs=copy.deepcopy(self.attrs),
                                          order=copy.deepcopy(self.order),
                                          **kwds)
        for tier, tier_value in values.items():
            for state, state_value in tier_value.items():
                state_values.set_state(tier, state, state_value)
        return state_values

    def __setitem__(self, index, value):
        if not self._pure_states():
            raise NotImplementedError('Set states directly via ".set_state()".')

    def set_state(self, tier, state, value):
        """Set the state of a tier.

        Args:
            tier: The tier may be specified by
                a tier number counting bottom-up, where negative values count
                top-down, or the tier name.
            state: Which state do you wish to specify?
            value: The new value of the tier.

        Raises:
            ValueError: If new value is not a tier.
        """
        if self._pure_states():
            raise NotImplementedError('You cannot change the states of {}.'.\
                                      format(self))
        tier = self._get_tier_name(tier)
        self._validate_setitem(tier, state, value)
        self[tier][state] = value

    def _validate_setitem(self, tier, state, value):
        tier_shape = self[tier].attrs['shape']
        value_shape = value.shape[-len(tier_shape):]
        value_n_obs = value.shape[-len(tier_shape)-1]
        if 'n_obs' not in self.attrs:
            self.attrs['n_obs'] = value_n_obs
        if value_n_obs != self.attrs['n_obs']:
            raise ValueError('Wrong number of observations.')
        if len(tier_shape) != len(value_shape):
            raise ValueError('Wrong number of dimensions.')
        for tier_s, value_s in zip(tier_shape, value_shape):
            if tier_s != value_s:
                raise ValueError('Wrong shape.')

    def get_tensors(self):
        state_tensors = self._state_tensors(dct=copy.deepcopy(dict(self)),
                                            attrs=copy.deepcopy(self.attrs),
                                            order=copy.deepcopy(self.order))
        for tier in state_tensors.order:
            for state, state_value in tier.items():
                if state_tensors.is_constant(tier, state):
                    value = tf.constant(state_value)
                else:
                    value = tf.Variable(state_value)
                state_tensors.set_value(tier, state, value)
        return state_tensors

class HDF5HierarchicalStates(HDF5Backend, HierarchicalStates):
    """Handle hierarchical states from within an hdf5 file.
    """

    def __init__(self, name, dct=None, attrs=None, order=None, **kwds):
        hierarchical_states = HierarchicalStates(dct=dct, attrs=attrs,
                                                 order=order)
        super().__init__(name=name, dct=hierarchical_states, **kwds)

class HierarchicalStateValues(StateValues, HierarchicalStates):
    """Set values in hierarchical states.
    """

    def __init__(self, dct=None, attrs=None, order=None):
        HierarchicalStates.__init__(self, dct=dct, attrs=attrs, order=order)

    @classmethod
    def _state_tensors(cls, **kwds):
        return HierarchicalStateTensors(**kwds)

    @classmethod
    def _hdf5(cls, **kwds):
        return HDF5HierarchicalStateValues(**kwds)

    def __setitem__(self, index, value):
        raise NotImplementedError('Set states directly via ".set_state()".')

    def is_constant(self, tier, state):
        state_dct = self[tier][state]
        try:
            return state_dct.attrs['shape']
        except AttributeError:
            return True

    def get_tensors(self):
        state_tensors = self._state_tensors(dct=copy.deepcopy(dict(self)),
                                            attrs=copy.deepcopy(self.attrs),
                                            order=copy.deepcopy(self.order))
        for tier in state_tensors.order:
            for state, state_value in self[tier].items():
                if state_tensors.is_constant(tier, state):
                    state_tensors.set_value

class HDF5HierarchicalStateValues(HDF5Backend, HierarchicalStateValues):
    """Handle hierarchical state values from within an hdf5 file.
    """

    def __init__(self, name, dct=None, attrs=None, order=None, **kwds):
        hierarchical_state_values = HierarchicalStateValues(dct=dct,
                                                            attrs=attrs,
                                                            order=order)
        super().__init__(name=name, dct=hierarchical_state_values, **kwds)

class HierarchicalStateTensors(StateTensors, HierarchicalStates):
    """Handle hierarchical state tensors.
    """

    def __init__(self, dct=None, attrs=None, order=None):
        HierarchicalStates.__init__(self, dct=dct, attrs=attrs, order=order)