import numpy as np
import scipy.stats as stats

class ArtificialData:
    """Arbitrary artificial data
    
    Arguments:
        data (numpy.ndarray): Dataset with one sample per row and one dimension per column
    """
    def __init__(self, data):
        self.data = data
        
class DecayingMultiNormal(ArtificialData):
    def __init__(self, dimensions, samples, alpha = 1):
        assert alpha >= 0
        pc_variance = np.exp(-alpha*np.array(range(dimensions)))
        rand_ortho = stats.ortho_group.rvs(dimensions)
        rand_normal = np.random.normal(scale = pc_variance, size = (samples,dimensions))
        rand_input = np.matmul(rand_normal, rand_ortho)
        super().__init__(rand_input)