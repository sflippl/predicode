import keras

class Hierarchical(keras.Sequential):
    """
    Define a hierarchical predictive coding model in a sequential fashion.
    """
    def __init__(self, layers = None, name = None):
        super().__init__(layers = layers, name = name)