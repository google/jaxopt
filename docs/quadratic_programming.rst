Quadratic programming
=====================

This section is concerned with minimizing quadratic functions subject
to equality and/or inequality constraints, also known as
`quadratic programming <https://en.wikipedia.org/wiki/Quadratic_programming>`_.

JAXopt supports several solvers for quadratic programming.
The solver specificities are summarized in the table below.
The best choice will depend on the usage.

.. list-table:: Quadratic Solvers
   :widths: 45, 15, 20, 15, 15, 15, 22
   :header-rows: 1

   * - Name
     - jit
     - matvec
     - precision
     - stability
     - speed
     - input format
   * - :class:`EqualityConstrainedQP`
     - yes
     - yes
     - ++
     - \+
     - +++
     - (Q, c), (A, b)
   * - :class:`CvxpyQP`
     - no
     - no
     - +++
     - +++
     - \+
     - (Q, c), (A, b), (G, h)
   * - :class:`OSQP`
     - yes
     - yes
     - \+
     - ++
     - ++
     - (Q, c), (A, b), (G, h)
   * - :class:`BoxOSQP`
     - yes
     - yes
     - \+
     - ++
     - ++
     - (Q, c), A, (l, u)

- *jit*: the algorithm can be used with jit or vmap, on GPU/TPU.
- *matvec*: the input can be given as matvec instead of dense matrices.
- *precision*: accuracy expected when the solver succeeds to converge.
- *stability*: capacity to handle badly scaled problems and matrices with poor conditioning.
- *speed*: typical speed on big instances to each its maximum accuracy.
- *input format*: see subsections below.


This table is given as rule of thumb only; on some particular instances
some solvers may behave unexpectedly better than others.
In case of difficulties, we suggest to test different combinations of
algorithms, ``maxiter`` and ``tol`` values.

.. warning::

  Those algorithms are guaranteed to converge on **convex problems** only.
  Hence, the matrix :math:`Q` *must* be positive semi-definite (PSD).

Equality-constrained QPs
------------------------

The problem takes the form:

.. math::

    \min_{x} \frac{1}{2} x^\top Q x + c^\top x \textrm{ subject to } A x = b

.. autosummary::
  :toctree: _autosummary

    jaxopt.EqualityConstrainedQP

This class is optimized for QPs with equality constraints only: it supports jit, pytrees and matvec.

Example::

  from jaxopt import EqualityConstrainedQP

  Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
  c = jnp.array([1.0, 1.0])
  A = jnp.array([[1.0, 1.0]])
  b = jnp.array([1.0])

  qp = EqualityConstrainedQP()
  sol = qp.run(params_obj=(Q, c), params_eq=(A, b)).params

  print(sol.primal)
  print(sol.dual_eq)

Ill-posed problems
~~~~~~~~~~~~~~~~~~

This solver is the fastest for well-posed problems, but can behave poorly on badly scaled matrices,
or with redundant constraints.

If the solver struggles to converge,
it is possible to enable
`iterative refinement <https://en.wikipedia.org/wiki/Iterative_refinement>`_.
This can be done by setting ``refine_regularization`` and ``refine_maxiter``::

  from jaxopt._src.eq_qp import EqualityConstrainedQP

  Q = 2 * jnp.array([[3000., 0.5], [0.5, 1]])
  c = jnp.array([1.0, 1.0])
  A = jnp.array([[1.0, 1.0]])
  b = jnp.array([1.0])

  qp = EqualityConstrainedQP(tol=1e-5, refine_regularization=3., refine_maxiter=50)
  sol = qp.run(params_obj=(Q, c), params_eq=(A, b)).params

  print(sol.primal)
  print(sol.dual_eq)
  print(qp.l2_optimality_error(sol, params_obj=(Q, c), params_eq=(A, b)))


General QPs
-----------

The problem takes the form:

.. math::

    \min_{x} \frac{1}{2} x^\top Q x + c^\top x \textrm{ subject to } A x = b, G x \le h

CvxpyQP
~~~~~~~

The wrapper over
`CVXPY <https://www.cvxpy.org>`_
is a solver that runs in ``float64`` precision.
However, it is not jittable, and does not support matvec and pytrees.

.. autosummary::
  :toctree: _autosummary

    jaxopt.CvxpyQP

Example::

  from jaxopt import CvxpyQP

  Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
  c = jnp.array([1.0, 1.0])
  A = jnp.array([[1.0, 1.0]])
  b = jnp.array([1.0])
  G = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
  h = jnp.array([0.0, 0.0])

  qp = CvxpyWrapper()
  sol = qp.run(params_obj=(Q, c), params_eq=(A, b), params_ineq=(G, h)).params

  print(sol.primal)
  print(sol.dual_eq)
  print(sol.dual_ineq)

It is also possible to specify only equality constraints or only inequality
constraints by setting ``params_eq`` or ``params_ineq`` to ``None``.

OSQP
~~~~

This solver is a pure JAX re-implementation of the OSQP algorithm.
It is jittable, supports pytrees and matvecs, but the precision is usually
lower than :class:`CvxpyQP` when run in float32 precision.

.. autosummary::
  :toctree: _autosummary

    jaxopt.OSQP

Example::

  from jaxopt import OSQP

  Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
  c = jnp.array([1.0, 1.0])
  A = jnp.array([[1.0, 1.0]])
  b = jnp.array([1.0])
  G = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
  h = jnp.array([0.0, 0.0])

  qp = OSQP()
  sol = qp.run(params_obj=(Q, c), params_eq=(A, b), params_ineq=(G, h)).params

  print(sol.primal)
  print(sol.dual_eq)
  print(sol.dual_ineq)

See :class:`BoxOSQP` for a full description of the parameters.

Box-constrained QPs
-------------------

The problem takes the form:

.. math::

    \min_{x,z} \frac{1}{2} x^\top Q x + c^\top x \textrm{ subject to } A x = z, l \le z \le u

.. autosummary::
  :toctree: _autosummary

    jaxopt.BoxOSQP

:class:`BoxOSQP` is similar to :class:`OSQP` but accepts problems in the above box-constrained format instead.
The bounds ``u`` (resp. ``l``) can be set to ``inf`` (resp. ``-inf``) if required.

Example::

  from jaxopt import BoxOSQP

  Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
  c = jnp.array([1.0, 1.0])
  A = jnp.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
  l = jnp.array([1.0, -jnp.inf, -jnp.inf])
  u = jnp.array([1.0, 0.0, 0.0])

  qp = BoxOSQP()
  sol = qp.run(params_obj=(Q, c), params_eq=A, params_ineq=(l, u)).params

  print(sol.primal)
  print(sol.dual_eq)
  print(sol.dual_ineq)

If required the algorithm can be sped up by setting
``check_primal_dual_infeasability`` to ``False``, and by setting
``eq_qp_preconditioner`` to ``"jacobi"`` (when possible).

.. note::

  The ``tol`` parameter controls the tolerance of the stopping criterion, which
  is based on the primal and dual residuals.  For over-constrained problems, or
  badly-scaled matrices, the residuals can be high, and it may be difficult to
  set ``tol`` appropriately.  In this case, it is better to tune ``maxiter``
  instead.

Unconstrained QPs
-----------------

For completeness, we also briefly describe how to solve unconstrained
quadratics of the form:

.. math::

    \min_{x} \frac{1}{2} x^\top Q x + c^\top x

The optimality condition rewrites :math:`\nabla \frac{1}{2} x^\top Q x + c^\top
x=Qx+c=0`.  Therefore, this is equivalent to solving the linear system
:math:`Qx=-c`.  Since the matrix :math:`Q` is assumed PSD, one of the best
algorithms is *conjugate gradient*.  In JAXopt, this can be done as follows::

  from jaxopt.linear_solve import solve_cg

  Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
  c = jnp.array([1.0, 1.0])
  matvec = lambda x: jnp.dot(Q, x)

  sol = solve_cg(matvec, b=-c)

  print(sol)
