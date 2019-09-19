"""States provide the different states of the tiers in Hierarchical.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import h5py

import predicode.connections as con

class States(dict):
    """States specify the states of the relevant variables in the different
    tiers.
    """

    def __init__(self):
        super().__init__({})
        self.attrs = {'order': np.array([], dtype=bytes)}

    @property
    def n_tiers(self):
        """The number of tiers.
        """
        return self.order.shape[0]

    @property
    def order(self):
        """The order of the tiers as a list.
        """
        return self.attrs['order']

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

    def __setitem__(self, index, value):
        """Set the state of a tier.

        Args:
            index: The tier may be specified by
                a tier number counting bottom-up, where negative values count
                top-down, or the tier name.
            value: The new value of the tier.

        Raises:
            ValueError: If new value is not a tier.
        """
        self._validate_setitem(value)
        index = self._get_tier_name(index)
        super().__setitem__(index, value)

    def _validate_setitem(self, value):
        if not isinstance(value, con.Tier):
            raise ValueError('New value must be a tier, but is a {}.'.\
                             format(type(value)))

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

    def add_tier(self, shape, tier_name=None):
        """Adds an empty tier.

        Args:
            shape: Specify the shape of the tier.
            tier_name: Specify the tier name. By default, tier names are given
                as tier_0, tier_1, and so on."""
        tier_name = tier_name or 'tier_%d' % (self.n_tiers, )
        if tier_name in self.order.astype(str):
            raise ValueError('Tier name %s is already taken.' % (tier_name, ))
        self.attrs['order'] = np.append(self.attrs['order'], [tier_name])\
                                .astype(bytes)
        self[tier_name] = con.Tier(shape=shape)

    def to_hdf5(self, name, **kwds):
        """Converts the state object to hdf5 object and returns the file handle.

        Args:
            name: Name of the file.
            kwds: Remaining arguments to h5py.File.__init__.

        Returns:
            pc.HDF5State object.
        """
        file = HDF5States(name=name, **kwds)
        for key, value in self.items():
            group = file.create_group(key)
            value.to_hdf5(group)
        for key, value in self.attrs.items():
            file.attrs[key] = value
        return file

class HDF5States(h5py.File, States):
    """Inspect a state within an hdf5 file.

    This class allows you to only load the chunks currently required into
    memory, while still retaining the API from the State class.

    Args:
        name: Name of the file.
        order: Order of the specified tiers.
    """

    def __init__(self, name, **kwds):
        super().__init__(name=name, **kwds)
        self.attrs['order'] = np.array([], dtype=bytes)

    def __getitem__(self, index):
        index = self._get_tier_name(index)
        return super().__getitem__(index)

    def __setitem__(self, index, value):
        self._validate_setitem(value)
        index = super()._get_tier_name(index)
        value.to_hdf5(self[index])

    def add_tier(self, shape, tier_name=None):
        tier_name = tier_name or 'tier_%d' % (self.n_tiers, )
        self.create_group(tier_name)
        super().add_tier(shape, tier_name)

    def to_hdf5(self, name, **kwds):
        raise NotImplementedError('Attempt to save hdf5 object as another '
                                  'hdf5 file. Use h5py functionality directly.')
