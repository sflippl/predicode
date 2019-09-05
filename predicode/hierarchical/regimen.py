"""Defines possible training regimens."""

import abc

import numpy as np
import tensorflow as tf

class Regimen(abc.ABC):
    """A training regimen defines how values change over the course of training
    and when training stops."""

    def start_batch(self):
        """This function indicates the start of a batch in the training regimen.
        """
        pass

    def finish_batch(self):
        """This function indicates the end of a batch in the training regimen.
        """

    @property
    def end(self):
        """This method determines whether training should stop.

        Returns:
            Either true if training should continue or false if it should not.
        """
        pass

    def training_step(self):
        """Trains a regimen as defined by its step-and-switch."""
        pass

class ConstantRegimen(Regimen):
    """This class defines a regimen that remains constant and does not affect
    the corresponding variables."""
    def __init__(self):
        pass

    def start_batch(self):
        """Starts the batch."""
        pass

    def finish_batch(self):
        pass

    def training_step(self, loss_fun, variables, metrics=[]):
        pass

    def end(self):
        return True

    def steps_until_convergence(self):
        return 0

    def restart(self):
        pass

    def train(self):
        pass

class SimpleOptimizerRegimen(Regimen):
    """This class defines a simple regimen using one optimizer and one set of
    values.

    Often, we wish to optimize a set of values by simply minimizing them up to
    a certain accuracy or at most for a number of steps. This simple class makes
    such a task easier.

    Args:
        optimizer: Any object that can handle a tensor with its 'minimize'
            method.
        eps: If the mean squared difference between the last two estimates
            is below this value, the regimen stops.
        max_steps: If this number of steps has been exceeded, the regimen stops.
    """

    def __init__(self, optimizer, eps=1e-5, max_steps=1e5):
        self.optimizer = optimizer
        self.eps = tf.constant(eps)
        self.max_steps = max_steps
        self.n_steps = 0
        self._grads = False
        self.metrics = None

    def start_batch(self):
        """Saves the estimated variables as the old values and increments the
        step number."""
        self._grads = True
        self.n_steps += 1

    def finish_batch(self):
        """This function takes care of cleaning up the metrics."""
        return
    
    def training_step(self, loss_fun, variables, metrics=[]):
        """Take a minimizing step for a tensor or a set of tensors.

        This function uses the optimizer to minimize the 'loss'.

        Args:
            gradients: The gradient the step should follow.
            variables: The variables that should be changed.
        """
        self._grads = self._training_step(loss_fun=loss_fun,
                                          variables=variables,
                                          _grads=self._grads,
                                          eps=self.eps,
                                          metrics=metrics)

    @tf.function
    def _training_step(self, loss_fun, variables, _grads, eps, metrics=[]):
        with tf.GradientTape() as tape:
            losses = loss_fun()
        gradients = tape.gradient(losses, variables)
        gen = []
        for grad, var in zip(gradients, variables):
            if grad is not None:
                gen.append((grad, var))
                _grads = _grads and (
                    tf.math.reduce_all(
                        tf.pow(tf.reshape(grad, [-1]), 2) < eps
                    )
                )
        self.optimizer.apply_gradients(gen)
        return _grads

    def end(self):
        """The regimen ends when the variables do not change by a significant
        amount anymore or the number of steps have been exceeded."""
        if self.n_steps >= self.max_steps:
            return True
        if (self.n_steps == 0):
            return False
        return self._grads

    def train(self, loss_fun, variables, metrics=[]):
        """Trains a model until convergence or the maximum number of steps
        have been exceeded."""
        while not self.end():
            self.start_batch()
            self.training_step(loss_fun, variables, metrics=metrics)
            self.finish_batch()

    def steps_until_convergence(self):
        """Returns the number of steps until convergence.

        If the regimen has converged, returns the step after which the gradient
        was below the threshold. If it hasn't returns NA."""
        if self._grads:
            return self.n_steps-1
        else:
            return np.nan

    def restart(self):
        """Restarts a regimen.

        This categorizes the metrics accordingly, and sets the steps back
        to zero."""
        self.n_steps = 0

class ExpectationMaximizationRegimen(Regimen):
    """This regimen implements an expectation maximization algorithm.

    The expectation maximization algorithm can be used to infer latent states
    and estimate the weights connecting these latent states with observations
    at the same time (see Dempster et al., 1987). This regimen accordingly
    consists of a state regimen and a weight regimen.

    Since the states are lost if there are several batches that are being
    iterated over, as a general convergence criterion, weight convergence is
    being used, with state convergence being the logical implication.

    Args:
        state_regimen: Which regimen should be used for the state?
        predictor_regimen: Which regimen should be used for the predictors?
        max_steps: How many EM-steps should at most be taken?"""

    def __init__(self, state_regimen, predictor_regimen, max_steps = 1000):
        self.state_regimen = state_regimen
        self.predictor_regimen = predictor_regimen
        self.max_steps = max_steps
        self.n_steps = 0
        self.metrics = None
        self._sut = [False]

    def start_batch(self):
        """Starts batch by restarting the regimens and incrementing the number
        of steps."""
        self.state_regimen.restart()
        self.predictor_regimen.restart()
        self.n_steps += 1
        self._sut = []

    def finish_batch(self):
        """Finishes batch."""
        return

    def end(self):
        """The regimen ends when the weight regimen had immediately converged.

        This means that even the very first gradient was below the threshold."""
        return all(self._sut)

    def training_step(self, loss_fun, state_variables, predictor_variables,
                      metrics=[]):
        """Takes one training steps by first learning the states and then
        learning the weights."""
        self.state_regimen.train(loss_fun, state_variables, metrics=metrics)
        self.predictor_regimen.train(loss_fun, predictor_variables,
                                     metrics=metrics)
        self._sut.append(self.predictor_regimen.steps_until_convergence() == 0)

    def train(self, loss_fun, state_variables, predictor_variables, metrics=[]):
        """Trains a model until convergence or the maximum number of steps
        have been exceeded."""
        while not self.end():
            self.start_batch()
            self.training_step(loss_fun, state_variables, predictor_variables,
                               metrics=metrics)
            self.finish_batch()

    def reset(self):
        """Restarts a regimen."""
        self.n_steps = 1
        self._sut = [False]
