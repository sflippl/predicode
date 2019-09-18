"""Default tiers.
"""

import h5py

class Tier(dict):
    """A tier consists of a particular shape and different possible states.

    Args:
        shape: The shape of one observation of the tier.
    """

    def __init__(self, shape):
        super().__init__({})
        self.attrs = {'shape': shape}

    def __setitem__(self, index, value):
        self._validate_shape(value)
        super().__setitem__(index, value)

    def _validate_shape(self, value):
        shape = value.shape[1:]
        if shape != self.shape:
            raise ValueError('New array has shape {}, not {}'.\
                             format(shape, self.shape))

    @property
    def shape(self):
        """The shape of any one observation from the tier.
        """
        return self.attrs['shape']

    def to_hdf5(self, group):
        """Converts the tier object to hdf5 object and returns the file handle.

        Args:
            name: Name of the file.
            kwds: Remaining arguments to h5py.File.__init__.

        Returns:
            pc.HDF5Tier object.
        """
        if isinstance(group, h5py.Group):
            group = group.id
        group = h5py.Group(group)
        for key, value in self.items():
            group[key] = value
        for key, value in self.attrs.items():
            group.attrs[key] = value
        return group
