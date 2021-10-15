API at a glance
===============

Optimization
------------

Unconstrained
~~~~~~~~~~~~~

.. autosummary::
  :toctree: _autosummary

    jaxopt.GradientDescent
    jaxopt.ScipyMinimize

Constrained
~~~~~~~~~~~

.. autosummary::
  :toctree: _autosummary

    jaxopt.ProjectedGradient
    jaxopt.MirrorDescent
    jaxopt.QuadraticProgramming
    jaxopt.ScipyBoundedMinimize

Non-smooth
~~~~~~~~~~

.. autosummary::
  :toctree: _autosummary

    jaxopt.ProximalGradient
    jaxopt.BlockCoordinateDescent

Stochastic
~~~~~~~~~~

.. autosummary::
  :toctree: _autosummary

    jaxopt.ArmijoSGD
    jaxopt.OptaxSolver
    jaxopt.PolyakSGD

Loss functions
~~~~~~~~~~~~~~

.. autosummary::
  :toctree: _autosummary

    jaxopt.loss.binary_logistic_loss
    jaxopt.loss.huber_loss
    jaxopt.loss.multiclass_logistic_loss
    jaxopt.loss.multiclass_sparsemax_loss

Root finding
------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.Bisection
    jaxopt.ScipyRootFinding

Fixed point resolution
----------------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.FixedPointIteration
    jaxopt.AndersonAcceleration
    jaxopt.AndersonWrapper

Implicit differentiation
------------------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.implicit_diff.custom_root
    jaxopt.implicit_diff.custom_fixed_point
    jaxopt.implicit_diff.root_jvp
    jaxopt.implicit_diff.root_vjp
