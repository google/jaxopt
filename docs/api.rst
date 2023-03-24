API at a glance
===============

Optimization
------------

Unconstrained
~~~~~~~~~~~~~

.. autosummary::
  :toctree: _autosummary

    jaxopt.BFGS
    jaxopt.GradientDescent
    jaxopt.LBFGS
    jaxopt.ScipyMinimize
    jaxopt.NonlinearCG

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

    jaxopt.BoxCDQP
    jaxopt.BoxOSQP
    jaxopt.CvxpyQP
    jaxopt.EqualityConstrainedQP
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
    jaxopt.loss.binary_sparsemax_loss
    jaxopt.loss.binary_hinge_loss
    jaxopt.loss.binary_perceptron_loss
    jaxopt.loss.sparse_plus
    jaxopt.loss.sparse_sigmoid
    jaxopt.loss.huber_loss
    jaxopt.loss.multiclass_logistic_loss
    jaxopt.loss.multiclass_sparsemax_loss
    jaxopt.loss.multiclass_hinge_loss
    jaxopt.loss.multiclass_perceptron_loss

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
    jaxopt.LevenbergMarquardt

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
    jaxopt.HagerZhangLineSearch


Perturbed optimizers
--------------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.perturbations.make_perturbed_argmax
    jaxopt.perturbations.make_perturbed_max
    jaxopt.perturbations.make_perturbed_fun
    jaxopt.perturbations.Gumbel
    jaxopt.perturbations.Normal



Isotonic regression
-------------------

.. autosummary::
  :toctree: _autosummary


    jaxopt.isotonic.isotonic_l2_pav


Tree utilities
--------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.tree_util.tree_add
    jaxopt.tree_util.tree_sub
    jaxopt.tree_util.tree_mul
    jaxopt.tree_util.tree_div
    jaxopt.tree_util.tree_scalar_mul
    jaxopt.tree_util.tree_add_scalar_mul
    jaxopt.tree_util.tree_vdot
    jaxopt.tree_util.tree_sum
    jaxopt.tree_util.tree_l2_norm
    jaxopt.tree_util.tree_zeros_like

