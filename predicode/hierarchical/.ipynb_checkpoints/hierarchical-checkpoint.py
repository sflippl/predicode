"""Defines general hierarchical predictive coding models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Hierarchical():
    """Defines a hierarchical predictive coding model."""

    n = 1
    def __init__(self, layers=None, name=None):
        if name is None:
            name='hierarchical_%d' % (n, )
            n += 1
        self.name = name