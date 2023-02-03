Loss and objective functions
============================

Loss functions
--------------

Regression
~~~~~~~~~~

.. autosummary::
  :toctree: _autosummary

    jaxopt.loss.huber_loss

Regression losses are of the form ``loss(float: target, float: pred) -> float``,
where ``target`` is the ground-truth and ``pred`` is the model's output.

Binary classification
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: _autosummary

    jaxopt.loss.binary_logistic_loss
    jaxopt.loss.binary_sparsemax_loss
    jaxopt.loss.binary_hinge_loss
    jaxopt.loss.binary_perceptron_loss

Binary classification losses are of the form ``loss(int: label, float: score) -> float``,
where ``label`` is the ground-truth (``0`` or ``1``) and ``score`` is the model's output.

Multiclass classification
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: _autosummary

    jaxopt.loss.multiclass_logistic_loss
    jaxopt.loss.multiclass_sparsemax_loss
    jaxopt.loss.multiclass_hinge_loss
    jaxopt.loss.multiclass_perceptron_loss

Multiclass classification losses are of the form ``loss(int: label, jnp.ndarray: scores) -> float``,
where ``label`` is the ground-truth (between ``0`` and ``n_classes - 1``) and
``scores`` is an array of size ``n_classes``.

Applying loss functions on a batch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All loss functions above are pointwise, meaning that they operate on a single sample. Use ``jax.vmap(loss)``
followed by a reduction such as ``jnp.mean`` or ``jnp.sum`` to use on a batch.

Objective functions
-------------------

.. _composite_linear_functions:

Composite linear functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: _autosummary

    jaxopt.objective.least_squares
    jaxopt.objective.binary_logreg
    jaxopt.objective.multiclass_logreg
    jaxopt.objective.multiclass_linear_svm_dual

Composite linear objective functions can be used with
:ref:`block coordinate descent <block_coordinate_descent>`.

Other functions
~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: _autosummary

    jaxopt.objective.ridge_regression
    jaxopt.objective.multiclass_logreg_with_intercept
    jaxopt.objective.l2_multiclass_logreg
    jaxopt.objective.l2_multiclass_logreg_with_intercept
    jaxopt.loss.sparse_plus
    jaxopt.loss.sparse_sigmoid
