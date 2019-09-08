"""Defines training regimens."""

import copy

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

class OptimizerRegimen:
    """This class defines a simple regimen using one optimizer and one set of
    values.

    Often, we wish to optimize a set of values by simply minimizing them up to
    a certain accuracy or at most for a number of steps. This simple class makes
    such a task easier.

    Args:
        optimizer: Any object that can handle a tensor with its 'minimize'
            method.
        eps: If the maximal difference between the last two estimates
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

    def finish_batch(self, metrics=None, it_baseline=0):
        """This function takes care of cleaning up the metrics."""
        metrics = metrics or []
        for metric in metrics:
            tf.summary.scalar(
                metric.name, metric.result(),
                step=it_baseline + self.optimizer.iterations
            )
            metric.reset_states()

    def training_step(self, loss_fun, variables, metrics=None):
        """Take a minimizing step for a tensor or a set of tensors.

        This function uses the optimizer to minimize the 'loss'.

        Args:
            gradients: The gradient the step should follow.
            variables: The variables that should be changed.
        """
        metrics = metrics or []
        self._grads = self._training_step(loss_fun=loss_fun,
                                          variables=variables,
                                          _grads=self._grads,
                                          eps=self.eps,
                                          metrics=metrics)

    @tf.function
    def _training_step(self, loss_fun, variables, _grads, eps, metrics): # pragma: no cover
        # (is only executed within Tensorflow call.)
        with tf.GradientTape() as tape:
            [losses, predictions, values] = loss_fun()
        gradients = tape.gradient(losses, variables)
        gen = []
        for metric in metrics:
            metric.update_state(values, predictions)
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
        amount anymore or the number of steps have been exceeded.
        """
        if self.n_steps == 0 and self.max_steps > 0:
            return False
        return self._grads or (self.n_steps >= self.max_steps)

    def train(self, loss_fun, variables, metrics=None, it_baseline=0):
        """Trains a model until convergence or the maximum number of steps
        have been exceeded.
        """
        metrics = metrics or []
        while not self.end():
            self.start_batch()
            self.training_step(loss_fun, variables, metrics=metrics)
            self.finish_batch(metrics=metrics, it_baseline=it_baseline)

    def steps_until_convergence(self):
        """Returns the number of steps until convergence.

        If the regimen has converged, returns the step after which the gradient
        was below the threshold. If not, it returns numpy.nan.
        """
        if self._grads:
            return self.n_steps-1
        return np.nan

    def restart(self):
        """Restarts a regimen.

        This categorizes the metrics accordingly, and sets the steps back
        to zero."""
        self.n_steps = 0

    @property
    def iterations(self):
        """The optimizer's iterations."""
        return self.optimizer.iterations

class ConstantRegimen(OptimizerRegimen):
    """This class defines a regimen that remains constant and does not affect
    the corresponding variables."""
    def __init__(self):
        super().__init__(None, max_steps=-1, eps=np.inf)

    def start_batch(self):
        """Indicates the start of a batch in the training regimen.
        """

    def finish_batch(self, metrics=None, it_baseline=0):
        """Indicates the end of a patch in the training regimen."""

    def training_step(self, loss_fun, variables, metrics=None):
        """A single training step."""

    def end(self):
        """Always indicates that the regimen has ended."""
        return True

    def steps_until_convergence(self):
        """Indicates that it took 0 steps until convergence."""
        return 0

    def restart(self):
        """Restarts the regimen."""

    def train(self, loss_fun, variables, metrics=None, it_baseline=0):
        """A training session."""

    @property
    def iterations(self):
        """Returns constant 0, since there are not iterations."""
        return 0

class EMRegimen:
    """This regimen implements an expectation maximization algorithm.

    The expectation maximization algorithm can be used to infer latent states
    and estimate the weights connecting these latent states with observations
    at the same time (see Dempster et al., 1987). This regimen accordingly
    consists of a state regimen and a predictor regimen.

    Since the states are lost if there are several batches that are being
    iterated over, predictor convergence is used as the general convergence
    criterion (with state convergence being the logical implication).

    Args:
        state_regimen: The regimen used for the state estimation.
        predictor_regimen: The regimen used for the predictor estimation.
    """

    def __init__(self, state_regimen, predictor_regimen):
        self.state_regimen = state_regimen
        self.predictor_regimen = predictor_regimen
        self.n_steps = 0
        self.metrics = None
        self._sut = [False]
        self._state_baseline = state_regimen.iterations
        self._predictor_baseline = predictor_regimen.iterations

    def start_batch(self):
        """Starts batch by restarting the regimens and incrementing the number
        of steps.
        """
        self.state_regimen.restart()
        self.predictor_regimen.restart()
        self.n_steps += 1
        self._sut = []

    def finish_batch(self):
        """Finishes batch.
        """

    def end(self, epochs):
        """The regimen ends when the weight regimen had immediately converged.

        This means that even the very first gradient was below the threshold.
        """
        if (self.n_steps == 0) and (epochs > 0):
            return False
        return all(self._sut) or (self.n_steps >= epochs)

    def training_step(self, loss_fun, state_variables, predictor_variables,
                      metrics=None):
        """Takes one training steps by first learning the states and then
        learning the weights.
        """
        metrics = metrics or []
        self.state_regimen.train(
            loss_fun, variables=state_variables,
            metrics=metrics,
            it_baseline=self._predictor_baseline
        )
        self.predictor_regimen.train(
            loss_fun, variables=predictor_variables,
            metrics=metrics,
            it_baseline=self._state_baseline
        )
        self._sut.append(self.predictor_regimen.steps_until_convergence() == 0)

    def train(self, loss_fun, state_variables, predictor_variables, # (mostly internal method) pylint:disable=too-many-arguments
              metrics=None, epochs=1):
        """Trains a model until convergence or the maximum number of steps
        have been exceeded.
        """
        metrics = metrics or []
        while not self.end(epochs):
            self.start_batch()
            self.training_step(loss_fun, state_variables, predictor_variables,
                               metrics=metrics)
            self.finish_batch()

    def restart(self):
        """Restarts a regimen.
        """
        self.n_steps = 0
        self._sut = [False]

def _get_sor(identifier):
    if isinstance(identifier, OptimizerRegimen):
        return copy.deepcopy(identifier)
    try:
        identifier = keras.optimizers.get(identifier)
    except ValueError as e:
        raise ValueError('Could not interpret regimen identifier.') from e
    return OptimizerRegimen(optimizer=copy.deepcopy(identifier))

def get(identifier):
    """Retrieves a EMRegimen instance.

    Args:
        identifier: Regimen identifier, one of:
            - String: Name of a keras optimizer
            - Dictionary: Regimen specified independently for states and
                predictors
            - EMRegimen: Returned unchanged
            - Keras optimizer instance: The optimizer used for state and
                predictor optimization
            - Regimen instance: The regimen used for state and predictor
                optimization

    Returns:
        An EMRegimen instance.

    Raises:
        ValueError: if 'identifier' cannot be interpreted.
    """
    if isinstance(identifier, OptimizerRegimen):
        return EMRegimen(
            state_regimen=copy.deepcopy(identifier),
            predictor_regimen=copy.deepcopy(identifier)
        )
    if isinstance(identifier, EMRegimen):
        return identifier
    if isinstance(identifier, dict):
        if not set(identifier.keys()).issubset({'states', 'predictors'}):
            raise ValueError('You can only specify "states" and "predictors" '
                             'in a dictionary.')
        for key in ['states', 'predictors']:
            if key not in identifier:
                identifier[key] = ConstantRegimen()
        return EMRegimen(
            state_regimen=_get_sor(identifier['states']),
            predictor_regimen=_get_sor(identifier['predictors'])
        )
    return EMRegimen(
        state_regimen=_get_sor(identifier),
        predictor_regimen=_get_sor(identifier)
    )
