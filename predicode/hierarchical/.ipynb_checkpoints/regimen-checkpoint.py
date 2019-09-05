"""Defines possible training regimens."""

import numpy as np
import tensorflow as tf

class SimpleOptimizerRegimen:
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

    def training_step(self, loss_fun, variables, metrics=None):
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
    def _training_step(self, loss_fun, variables, _grads, eps, metrics=None): # pragma: no cover
        # (is only executed within Tensorflow call.)
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
        if self.n_steps == 0 and self.max_steps > 0:
            return False
        return self._grads or (self.n_steps >= self.max_steps)

    def train(self, loss_fun, variables, metrics=None):
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
        return np.nan

    def restart(self):
        """Restarts a regimen.

        This categorizes the metrics accordingly, and sets the steps back
        to zero."""
        self.n_steps = 0

class ConstantRegimen(SimpleOptimizerRegimen):
    """This class defines a regimen that remains constant and does not affect
    the corresponding variables."""
    def __init__(self):
        super().__init__(None, max_steps=-1, eps=np.inf)

    def start_batch(self):
        """Indicates the start of a batch in the training regimen.
        """
        return

    def finish_batch(self):
        """Indicates the end of a patch in the training regimen."""
        return

    def training_step(self, loss_fun, variables, metrics=None):
        """A single training step."""
        return

    def end(self):
        """Always indicates that the regimen has ended."""
        return True

    def steps_until_convergence(self):
        """Indicates that it took 0 steps until convergence."""
        return 0

    def restart(self):
        """Restarts the regimen."""
        return

    def train(self, loss_fun, variables, metrics=None):
        """A training session."""
        return

class ExpectationMaximizationRegimen:
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

    def __init__(self, state_regimen, predictor_regimen, max_steps=1000):
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
        if self.n_steps == 0 and self.max_steps > 0:
            return False
        return all(self._sut) or (self.n_steps >= self.max_steps)

    def training_step(self, loss_fun, state_variables, predictor_variables,
                      metrics=None):
        """Takes one training steps by first learning the states and then
        learning the weights."""
        self.state_regimen.train(loss_fun, state_variables, metrics=metrics)
        self.predictor_regimen.train(loss_fun, predictor_variables,
                                     metrics=metrics)
        self._sut.append(self.predictor_regimen.steps_until_convergence() == 0)

    def train(self, loss_fun, state_variables, predictor_variables,
              metrics=None):
        """Trains a model until convergence or the maximum number of steps
        have been exceeded."""
        while not self.end():
            self.start_batch()
            self.training_step(loss_fun, state_variables, predictor_variables,
                               metrics=metrics)
            self.finish_batch()

    def restart(self):
        """Restarts a regimen."""
        self.n_steps = 0
        self._sut = [False]
