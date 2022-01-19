API at a glance
===============

Optimization
------------

Unconstrained
~~~~~~~~~~~~~

.. autosummary::
  :toctree: _autosummary

    jaxopt.GradientDescent
    jaxopt.LBFGS
    jaxopt.ScipyMinimize

Constrained
~~~~~~~~~~~

.. autosummary::
  :toctree: _autosummary

    jaxopt.ProjectedGradient
    jaxopt.MirrorDescent
    jaxopt.ScipyBoundedMinimize

Quadratic programming
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: _autosummary

    jaxopt.EqualityConstrainedQP
    jaxopt.CvxpyQP
    jaxopt.BoxOSQP
    jaxopt.OSQP

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

Linear system solving
---------------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.linear_solve.solve_lu
    jaxopt.linear_solve.solve_cholesky
    jaxopt.linear_solve.solve_cg
    jaxopt.linear_solve.solve_normal_cg
    jaxopt.linear_solve.solve_gmres
    jaxopt.linear_solve.solve_bicgstab
    jaxopt.IterativeRefinement

Nonlinear least squares
-----------------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.GaussNewton

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

Line search
-----------

.. autosummary::
  :toctree: _autosummary

    jaxopt.BacktrackingLineSearch
