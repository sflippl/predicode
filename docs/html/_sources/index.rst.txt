.. predicode documentation master file, created by
   sphinx-quickstart on Mon Aug 19 21:21:43 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Predicode: Hierarchical Predictive Coding in Python
================================================================================
.. image:: https://travis-ci.org/sflippl/predicode.svg?branch=master
    :target: https://travis-ci.org/sflippl/predicode
.. image:: https://coveralls.io/repos/github/sflippl/predicode/badge.svg?branch=master
    :target: https://coveralls.io/github/sflippl/predicode?branch=master
.. image:: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.7--dev-blue


Predicode is a high-level API for predictive coding algorithms in Python, written on top of `Tensorflow 2.0 <https://www.tensorflow.org/beta/>`_. It was written with the guiding principles of `Keras <https://keras.io>`_ in mind. In particular, it allows for the integration of arbitrary Keras models in the predictive coding hierarchy. You can declare your models completely in Python and easily extend the functionality.

What is hierarchical predictive coding?
--------------------------------------------------------------------------------

A hierarchical predictive coding model consists of layers of latent variables (*tiers*). Each tier attempts to predict the adjacent lower tier, resulting in a predicted state and a prediction error. By minimizing the prediction error, both the latent variables and the predictors of these variables are estimated.

This principle is often complemented by more general variations that will be supported in future versions of the package.

A predictive coding model in 60 seconds
--------------------------------------------------------------------------------

You can create a new predictive coding model using the class :class:`pc.Hierarchical`.

>>> import predicode as pc
>>> hpc = pc.Hierarchical()

As a next step, you can add tiers increasing in the hierarchy.

>>> # This represents a ten-dimensional input layer.
>>> hpc.add_tier(shape=(10, ))
>>> # This represents a four-dimensional latent layer.
>>> hpc.add_tier(shape=(4, ))
>>> # Active connection: tier_1 -> tier_0

You then need to specify how the tiers are connected. In this case, we specify that the higher tier (tier 1) should predict the lower tier (tier 0), the prediction error being the difference between the prediction and the observed value and the loss function driving the estimation being the mean squared error.

:class:`pc.connections.TopDownSequential` allows you to configure a sequential model predicting tier 0 from tier 1.

>>> import tensorflow.keras as keras
>>> hpc.connection = pc.connections.TopDownSequential([
>>>     keras.layers.Dense(10)
>>> ])

Modifying the connection works in the same way as in Keras:

>>> hpc.connection.add(keras.layers.Activation('relu'))
>>> hpc.connection.add(keras.layers.Dense(10))

Once your model looks good, you can configure the learning process:

>>> hpc.compile(regimen='adam', metrics=['mean_squared_error'])

If you need to, you can further configure the optimization regimen, for instance by specifying different optimizers for state and predictor estimation.

>>> hpc.compile(regimen=pc.EMRegimen(
>>>     state_regimen='adam',
>>>     predictor_regimen=keras.optimizers.SGD(learning_rate=1)
>>> ))

Finally, you can train the model on your dataset.

>>> # dataset is some ten-dimensional dataset.
>>> hpc.train(dataset)

Evaluate your performance (or inspect the inferred tiers, predictions, or prediction errors) in one line:

>>> metrics = hpc.evaluate()
>>> # The inferred values of the latent tier_1
>>> tier_1 = hpc.tier(1)
>>> # The prediction error in the observed tier_0
>>> prediction_error = hpc.prediction_error(0)

In this way, you can create arbitarily complex predictive coding models with several tiers, complex predictors, and elaborate optimization mechanisms, utilizing the close integration of Keras and Tensorflow.

Learning more
--------------------------------------------------------------------------------

The following chapters contain a more in-depth introduction to predicode starting with simple examples before explaining how to tweak your optimization regimen and how to take advantage of Tensorboard. Each of the chapters can be downloaded as a Jupyter notebook or -- even simpler -- be opened online in Google Colab, where you change lines of code in an interactive session without any preparation.

The :ref:`modindex` provides with more detailed documentation of the software itself.

Finally, I would recommend the `Keras documentation <https://keras.io>`_ for more resources on how to define predictors.

Installation
--------------------------------------------------------------------------------

Get predicode 0.1.0-beta on PyPi now:

>>> pip install predicode

Alternatively, you can download the latest development version from Github:

>>> pip install git+https://github.com/sflippl/predicode

Support and future development
--------------------------------------------------------------------------------

Predicode 0.2.0 is currently under `active development <https://github.com/sflippl/predicode/milestone/2>`_. Stay tuned for autopredictive coding tiers, metrics that are more customized to predictive coding, and state and predictor traces!

If you would like to file a bug, submit a feature request, or contribute to the development, please file an issue on `Github <https://github.com/sflippl/predicode/issues>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:
   
   usage/get_started
   usage/installation
   usage/datasets
   usage/minimal_model
