Linear system solving
=====================

This section is concerned with solving problems of the form

.. math::

    Ax = b

with unknown :math:`x` for a linear operator :math:`A` and vector :math:`b`.

Indirect solvers
----------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.linear_solve.solve_cg
    jaxopt.linear_solve.solve_normal_cg
    jaxopt.linear_solve.solve_gmres
    jaxopt.linear_solve.solve_bicgstab


Indirect solvers are wrappers on top of ``jax.scipy.sparse.linalg`` that support
ridge regularization wirth ``ridge`` hyper-parameter.
They can be *warm started* using optional ``init`` parameter.
Other hyper-parameters (such as ``tol`` or ``maxiter``) are forwarded::


  import jaxopt.linear_solve as linear_solve
  import numpy as onp

  onp.random.seed(42)
  A = onp.random.randn(3, 3)
  b = onp.random.randn(3)

  ridge = 0.5
  init = onp.random.randn(3)
  tol = 1e-5

  def matvec_A(x):
    return  jnp.dot(A, x)

  sol = linear_solve.solve_normal_cg(matvec_A, b, ridge=ridge, init=init, tol=tol)
  print(sol)

  sol = linear_solve.solve_gmres(matvec_A, b, ridge=ridge, init=init, tol=tol)
  print(sol)

  sol = linear_solve.solve_bicgstab(matvec_A, b, ridge=ridge, init=init, tol=tol)
  print(sol)


Direct solvers
--------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.linear_solve.solve_lu
    jaxopt.linear_solve.solve_cholesky


Direct solvers are wrappers over ``jax.scipy.linalg``.
They support matvecs (but not pytrees).

Example of usage::

  import jaxopt.linear_solve as linear_solve
  import numpy as onp

  onp.random.seed(42)
  A = onp.random.randn(3, 3)
  b = onp.random.randn(3)

  init = onp.random.randn(3)

  def matvec_A(x):
    return jnp.dot(A, x)

  sol = linear_solve.solve_lu(matvec_A, b)
  print(sol)


Iterative refinement
--------------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.IterativeRefinement


Instantiating and running the solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The class :class:`IterativeRefinement` aims to improve accuracy of linear solvers
from ``jaxopt.linear_solve``.

This is a meta-algorithm for solving the linear system ``Ax = b`` based on
a provided linear system solver. Our implementation is a slight generalization
of the `standard algorithm <https://en.wikipedia.org/wiki/Iterative_refinement>`_.
It starts with :math:`(r_0, x_0) = (b, 0)` and
iterates

.. math::

  \begin{aligned}
  x &= \text{solution of } \bar{A} x = r_{t-1}\\
  x_t &= x_{t-1} + x\\
  r_t &= b - A x_t
  \end{aligned}

where :math:`\bar{A}` is some approximation of A, with preferably
better preconditonning than A. By default, we use
:math:`\bar{A} = A`, which is the standard iterative refinement algorithm.

This method has the advantage of converging even if the solve step is
inaccurate.  This is particularly useful for ill-posed problems::

  from functools import partial
  import jax.numpy as jnp
  import numpy as onp
  from jaxopt import IterativeRefinement
  from jaxopt.linear_solve import solve_gmres

  # ill-conditioned linear system
  A = jnp.array([[3.9, 1.65], [6.845, 2.9]])
  b = jnp.array([5.5, 9.7])
  print(f"Condition number: {onp.linalg.cond(A):.0f}")
  # Condition number: 4647

  ridge = 1e-2
  tol = 1e-7

  x = solve_gmres(lambda x: jnp.dot(A, x), b, tol=tol)
  print(f"GMRES only error: {jnp.linalg.norm(A @ x - b):.7f}")
  # GMRES only error: nan

  solve_gmres_ridge = partial(solve_gmres, ridge=ridge)

  x_ridge = solve_gmres_ridge(lambda x: jnp.dot(A, x), b, tol=tol, ridge=ridge)
  print(f"GMRES+ridge error: {jnp.linalg.norm(A @ x_ridge - b):.7f}")
  # GMRES+ridge error: 0.0333328

  solver = IterativeRefinement(solve=solve_gmres_ridge,
                              tol=tol, maxiter=100)
  x_refined, state = solver.run(init_params=None, params_A=A, b=b)
  print(f"Iterativement Refinement error: {jnp.linalg.norm(A @ x_refined - b):.7f}")
  # Iterativement Refinement error: 0.0000000
