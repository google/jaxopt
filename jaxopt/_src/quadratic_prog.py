# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quadratic programming in JAX."""

from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple

import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt._src import implicit_diff as idf
from jaxopt._src import linear_solve
from jaxopt._src import tree_util


ArrayPair = Tuple[jnp.ndarray, jnp.ndarray]


def _check_params(params_obj, params_eq=None, params_ineq=None):
  if params_obj is None:
    raise ValueError("params_obj should be a tuple (Q, c)")
  Q, c = params_obj
  if Q.shape[0] != Q.shape[1]:
    raise ValueError("Q must be a square matrix.")
  if Q.shape[1] != c.shape[0]:
    raise ValueError("Q.shape[1] != c.shape[0]")

  if params_eq is not None:
    A, b = params_eq
    if A.shape[0] != b.shape[0]:
      raise ValueError("A.shape[0] != b.shape[0]")
    if A.shape[1] != Q.shape[1]:
      raise ValueError("Q.shape[1] != A.shape[1]")

  if params_ineq is not None:
    G, h = params_ineq
    if G.shape[0] != h.shape[0]:
      raise ValueError("G.shape[0] != h.shape[0]")
    if G.shape[1] != Q.shape[1]:
      raise ValueError("G.shape[1] != Q.shape[1]")


def _matvec_and_rmatvec(matvec, x, y):
  """Returns both matvec(x) = dot(A, x) and rmatvec(y) = dot(A.T, y)."""
  matvec_x, vjp = jax.vjp(matvec, x)
  rmatvec_y, = vjp(y)
  return matvec_x, rmatvec_y


def _solve_eq_constrained_qp(init_params,
                             matvec_Q,
                             c,
                             matvec_A,
                             b,
                             maxiter,
                             tol=1e-5,
                             solve=linear_solve.solve_gmres):
  """Solves 0.5 * x^T Q x + c^T x subject to Ax = b.

  This solver returns both the primal solution (x) and the dual solution.
  """

  def matvec(u):
    primal_u, dual_u = u
    mv_A, rmv_A = _matvec_and_rmatvec(matvec_A, primal_u, dual_u)
    return (tree_util.tree_add(matvec_Q(primal_u), rmv_A), mv_A)

  minus_c = tree_util.tree_scalar_mul(-1, c)

  # Solves the following linear system:
  # [[Q A^T]  [primal_var = [-c
  #  [A 0  ]]  dual_var  ]    b]
  return solve(matvec, (minus_c, b), tol=tol, maxiter=maxiter)


def _solve_constrained_qp_cvxpy(params_obj, params_eq, params_ineq):
  """Solve 0.5 * x^T Q x + c^T x subject to Gx <= h, Ax = b."""

  # CVXPY runs on CPU. Hopefully, we can implement our own pure JAX solvers
  # and remove this dependency in the future.
  # TODO(frostig,mblondel): experiment with `jax.experimental.host_callback`
  # to "support" other devices (GPU/TPU) in the interim, by calling into the
  # host CPU and running cvxpy there.
  import cvxpy as cp

  Q, c = params_obj
  A, b = params_eq
  G, h = params_ineq

  x = cp.Variable(len(c))
  objective = 0.5 * cp.quad_form(x, Q) + c.T @ x
  constraints = [A @ x == b, G @ x <= h]
  pb = cp.Problem(cp.Minimize(objective), constraints)
  pb.solve(solver='OSQP')

  if pb.status in ["infeasible", "unbounded"]:
    raise ValueError("The problem is %s." % pb.status)

  return (jnp.array(x.value), jnp.array(pb.constraints[0].dual_value),
          jnp.array(pb.constraints[1].dual_value))


def _create_matvec(matvec, M):
  if matvec is not None:
    # M = params_M
    return lambda u: matvec(M, u)
  else:
    return lambda u: jnp.dot(M, u)


def _make_quadratic_prog_optimality_fun(matvec_Q, matvec_A):
  """Makes the optimality function for quadratic programming.

  Returns:
    optimality_fun(params, params_obj, params_eq, params_ineq) where
      params = (primal_var, eq_dual_var, ineq_dual_var)
      params_obj = (Q, c)
      params_eq = (A, b)
      params_ineq = (G, h) or None
  """
  def obj_fun(primal_var, params_obj):
    Q, c = params_obj
    _matvec_Q = _create_matvec(matvec_Q, Q)
    return (0.5 * tree_util.tree_vdot(primal_var, _matvec_Q(primal_var)) +
            tree_util.tree_vdot(primal_var, c))

  def eq_fun(primal_var, params_eq):
    A, b = params_eq
    _matvec_A = _create_matvec(matvec_A, A)
    return tree_util.tree_sub(_matvec_A(primal_var), b)

  def ineq_fun(primal_var, params_ineq):
    G, h = params_ineq
    # TODO(mblondel): Add support for matvec_G when we implement our own QP
    # solver.
    return jnp.dot(G, primal_var) - h

  return idf.make_kkt_optimality_fun(obj_fun, eq_fun, ineq_fun)


@dataclass(eq=False)
class QuadraticProgramming(base.Solver):
  """Deprecated: will be removed in v0.4.

  Use :class:`jaxopt.CvxpyQP`, :class:`jaxopt.OSQP`, :class:`jaxopt.BoxOSQP` and
  :class:`jaxopt.EqualityConstrainedQP` instead.

  The objective function is::

    0.5 * x^T Q x + c^T x subject to Gx <= h, Ax = b.

  Attributes:
    matvec_Q: a Callable matvec_Q(params_Q, u).
      By default, matvec_Q(Q, u) = dot(Q, u), where Q = params_Q.
    matvec_A: a Callable matvec_A(params_A, u).
      By default, matvec_A(A, u) = dot(A, u), where A = params_A.
    maxiter: maximum number of iterations.
    implicit_diff_solve: the linear system solver to use.
  """

  # TODO(mblondel): add matvec_G when we implement our own QP solvers.
  matvec_Q: Optional[Callable] = None
  matvec_A: Optional[Callable] = None
  maxiter: int = 1000
  tol: float = 1e-5
  implicit_diff_solve: Optional[Callable] = None

  def run(self,
          init_params: Optional[Tuple] = None,
          params_obj: Optional[ArrayPair] = None,
          params_eq: Optional[ArrayPair] = None,
          params_ineq: Optional[ArrayPair] = None) -> base.OptStep:
    """Runs the quadratic programming solver in CVXPY.

    The returned params contains both the primal and dual solutions.

    Args:
      init_params: ignored.
      params_obj: (Q, c) or (params_Q, c) if matvec_Q is provided.
      params_eq: (A, b) or (params_A, b) if matvec_A is provided.
      params_ineq: = (G, h) or None if no inequality constraints.
    Returns:
      (params, state), ``params = (primal_var, dual_var_eq, dual_var_ineq)``
    """
    if self.matvec_Q is None and self.matvec_A is None:
      _check_params(params_obj, params_eq, params_ineq)

    Q, c = params_obj
    A, b = params_eq

    matvec_Q = _create_matvec(self.matvec_Q, Q)
    matvec_A = _create_matvec(self.matvec_A, A)

    if params_ineq is None:
      sol = base.KKTSolution(*_solve_eq_constrained_qp(init_params,
                                                       matvec_Q, c,
                                                       matvec_A, b,
                                                       self.maxiter, tol=self.tol))
    else:
      sol = base.KKTSolution(*_solve_constrained_qp_cvxpy(params_obj,
                                                          params_eq,
                                                          params_ineq))

    # No state needed currently as we use CVXPY.
    return base.OptStep(params=sol, state=None)

  def l2_optimality_error(
      self,
      params: Any,
      params_obj: ArrayPair,
      params_eq: ArrayPair,
      params_ineq: Optional[ArrayPair] = None) -> base.OptStep:
    """Computes the L2 norm of the KKT residuals."""
    pytree = self.optimality_fun(params, params_obj, params_eq, params_ineq)
    return tree_util.tree_l2_norm(pytree)

  def __post_init__(self):
    warnings.warn("Class 'QuadraticProgramming' will be removed in v0.4. "
                  "Use 'EqualityConstraintsQP' if you want the same behavior as "
                  "'QuadraticProgramming' for QPs with equality constraints only. "
                  "Use 'CVXPY_QP' if you want the same behavior as "
                  "'QuadraticProgramming' for QPs with inequality constraints. "
                  "Use 'OSQP' if you want a solver that supports pytrees, matvec, jit and vmap "
                  "for QPs with inequality constraints.", FutureWarning)

    self.optimality_fun = _make_quadratic_prog_optimality_fun(self.matvec_Q,
                                                              self.matvec_A)

    # Set up implicit diff.
    decorator = idf.custom_root(self.optimality_fun, has_aux=True,
                                solve=self.implicit_diff_solve)
    # pylint: disable=g-missing-from-attributes
    self.run = decorator(self.run)
