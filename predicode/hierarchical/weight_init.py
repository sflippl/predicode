from abc import ABC
import numpy as np
import sklearn.decomposition as decomp
import scipy.stats as stats

class WeightInit(ABC):
    pass

class WeightInitPCA(WeightInit):
    def __init__(self):
        return
    
    def initialize(self, latent_dimensions = None, input_data = None, **kwargs):
        weights = decomp.PCA(n_components = latent_dimensions).fit(input_data).components_.T
        return weights

class WeightInitRandom(WeightInit):
    def validate_method(self, method):
        assert method in ['orthogonal']
        return method
    
    def __init__(self, method = 'orthogonal'):
        method = self.validate_method(method)
        self.method = method
    
    def initialize(self, input_dimensions = None, latent_dimensions = None, **kwargs):
        if latent_dimensions is None:
            latent_dimensions = input_dimensions
        if method == 'orthogonal':
            weights = stats.ortho_group(input_dimensions)[:,range(latent_dimensions)]
            return weights

def weight_init(char, **kwargs):
    if type(char) is np.ndarray:
        return char
    if char == 'pca':
        return WeightInitPCA().initialize(**kwargs)
    if char == 'random':
        return WeightInitRandom('orthogonal').initialize(**kwargs)