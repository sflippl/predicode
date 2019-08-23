import tensorflow as tf
import numpy as np
from predicode.hierarchical.weight_init import WeightInit, weight_init

class MinimalHierarchicalModel():
    def _state_estimator_predict(mode, flipped_graph):
        return tf.estimator.EstimatorSpec(mode, predictions = flipped_graph)
    
    def _state_estimator_loss(flipped_graph, labels):
        loss = tf.losses.mean_squared_error(flipped_graph, labels)
        return loss
    
    def _state_estimator_eval(mode, flipped_graph, labels):
        loss = MinimalHierarchicalModel._state_estimator_loss(flipped_graph, labels)
        return tf.estimator.EstimatorSpec(mode, loss = loss)
    
    def _state_estimator_train(mode, flipped_graph, labels):
        loss = MinimalHierarchicalModel._state_estimator_loss(flipped_graph, labels)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    
    def _state_estimator(features, labels, mode, params):
        flipped_graph = tf.feature_column.input_layer(features, params['latent_weights'])
        flipped_graph = tf.layers.dense(flipped_graph, 
                                        units = params['n_observations'], 
                                        use_bias = params['use_bias'])
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            return MinimalHierarchicalModel._state_estimator_predict(mode, flipped_graph)

        if mode == tf.estimator.ModeKeys.EVAL:
            return MinimalHierarchicalModel._state_estimator_eval(mode, flipped_graph, labels)

        assert mode == tf.estimator.ModeKeys.TRAIN
        
        return MinimalHierarchicalModel._state_estimator_train(mode, flipped_graph, labels)
    
    def __init__(self, input_data, weights = 'pca', latent_dimensions = None, use_bias = False, **kwargs):
        """
        Initializes a minimal hierarchical model. This model consists of an input layer and one latent layer without a prior that are linearly and densely connected.
        
        Parameters
        -----------
        
        input_data: a :class:`np.Array`, where the columns correspond to the data dimensions and the rows correspond to occurences.
        weights: either:
            - a :class:`np.Array`, where the rows correspond to the input dimensions, in order of their appearance in the input_data 
            and the columns should correspond to the latent dimensions
            - a :class:`WeightInit`
            - a character corresponding to a :class:`WeightInit`
        latent_dimensions (optional): Number of latent dimensions
        use_bias: Does the model allow for an intercept?
        **kwargs: Optional parameters
        """
        
        self.input_data = input_data
        self.input_dimensions = input_data.shape[1]
        self.n_observations = input_data.shape[0]    
        self.latent_dimensions = latent_dimensions  
        
        self.weights = weight_init(weights, 
                                  input_data = input_data, 
                                  latent_dimensions = latent_dimensions, 
                                  input_dimensions = self.input_dimensions)
        self.dct_weights = {"theta1_%d"%(i): self.weights[:,i] for i in range(self.weights.shape[1])}
        
        
        
        assert self.input_dimensions == self.weights.shape[0]
        assert self.latent_dimensions == self.weights.shape[1]
        
        kwargs['use_bias'] = use_bias
        kwargs['n_observations'] = self.n_observations
        kwargs['latent_weights'] = [
            tf.feature_column.numeric_column(key = key, dtype = tf.float64) for key in self.dct_weights.keys()
        ]
        
        self.state = tf.estimator.Estimator(model_fn = MinimalHierarchicalModel._state_estimator, params = kwargs)
        
        self.activate('state')
    
    def validate_what(self, what):
        assert what in ['state', 'weight']
        return what
    
    def activate(self, what):
        what = self.validate_what(what)
        self.what = what
    
    def train(self, steps = 1e4, **kwargs):
        if self.what == 'state':
            return self.state.train(input_fn = lambda: (self.dct_weights, self.input_data.T),
                                    steps = steps,
                                    **kwargs)
        raise NotImplementedError()
    
    def evaluate(self, **kwargs):
        if self.what == 'state':
            return self.state.evaluate(input_fn = lambda: (self.dct_weights, self.input_data.T),
                                       steps = 1,
                                       **kwargs)
        raise NotImplementedError()
    
    def predict(self, **kwargs):
        if self.what == 'state':
            generator = self.state.predict(input_fn = lambda: (self.dct_weights, self.input_data.T),
                                           **kwargs)
            prediction_lst = [next(generator) for i in range(self.input_dimensions)]
            prediction = np.array(prediction_lst).T
            return prediction
        raise NotImplementedError()
    
    def latent_values(self):
        return self.state.get_variable_value('dense/kernel').T