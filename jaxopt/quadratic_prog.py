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
from typing import Optional
from typing import Tuple

from dataclasses import dataclass

import jax.numpy as jnp

from jaxopt import base
from jaxopt import implicit_diff3 as idf
from jaxopt import linear_solve
from jaxopt import tree_util


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


def _solve_eq_constrained_qp(params_obj, params_eq):
  """Solve 0.5 * x^T Q x + c^T x subject to Ax = b."""
  Q, c = params_obj
  A, b = params_eq

  def matvec(u):
    primal_u, dual_u = u

    return (jnp.dot(Q, primal_u) + jnp.dot(A.T, dual_u), jnp.dot(A, primal_u))

  return linear_solve.solve_cg(matvec, (-c, b))


def _solve_constrained_qp_cvxpy(params_obj, params_eq, params_ineq):
  """Solve 0.5 * x^T Q x + c^T x subject to Gx <= h, Ax = b."""

  # CVXPY runs on CPU. Hopefully, we can implement our own pure JAX solvers
  # and remove this dependency in the future.

  import cvxpy as cp

  Q, c = params_obj
  A, b = params_eq
  G, h = params_ineq

  x = cp.Variable(len(c))
  objective = 0.5 * cp.quad_form(x, Q) + c.T @ x
  constraints = [A @ x == b, G @ x <= h]
  pb = cp.Problem(cp.Minimize(objective), constraints)
  pb.solve()
  return (jnp.array(x.value), jnp.array(pb.constraints[0].dual_value),
          jnp.array(pb.constraints[1].dual_value))


def _make_quadratic_prog_optimality_fun():
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
    return (0.5 * jnp.dot(primal_var, jnp.dot(Q, primal_var)) +
            jnp.dot(primal_var, c))

  def eq_fun(primal_var, params_eq):
    A, b = params_eq
    return jnp.dot(A, primal_var) - b

  def ineq_fun(primal_var, params_ineq):
    G, h = params_ineq
    return jnp.dot(G, primal_var) - h

  return idf.make_kkt_optimality_fun(obj_fun, eq_fun, ineq_fun)


@dataclass
class QuadraticProgramming:
  """Quadratic programming solver.

  The objective function is::

    0.5 * x^T Q x + c^T x subject to Gx <= h, Ax = b.
  """

  def run(self,
          init_params: Optional[Tuple] = None,
          params_obj: Optional[ArrayPair] = None,
          params_eq: Optional[ArrayPair] = None,
          params_ineq: Optional[ArrayPair] = None) -> base.OptStep:
    """Runs the quadratic programming solver in CVXPY.

    The returned params contains both the primal and dual solutions.

    Args:
      init_params: ignored.
      params_obj: (Q, c)
      params_eq: (A, b)
      params_ineq: = (G, h) or None if no inequality constraints.
    Return type:
      base.OptStep
    Returns:
      (params, state), ``params = (primal_var, dual_var_eq, dual_var_ineq)``
    """
    # Not used at the moment but it will be when we implement our own solvers.
    del init_params

    _check_params(params_obj, params_eq, params_ineq)

    if params_ineq is None:
      primal, dual_eq = _solve_eq_constrained_qp(params_obj, params_eq)
      params = (primal, dual_eq, None)
    else:
      params = _solve_constrained_qp_cvxpy(params_obj, params_eq, params_ineq)

    # No state needed currently as we use CVXPY.
    return base.OptStep(params=params, state=None)

  def l2_optimality_error(self,
                          params: Any,
                          params_obj: Tuple,
                          params_eq: Tuple,
                          params_ineq: Optional[Tuple]) -> base.OptStep:
    """Computes the L2 norm of the KKT residuals."""
    pytree = self.optimality_fun(params, params_obj, params_eq, params_ineq)
    return tree_util.tree_l2_norm(pytree)

  def __post_init__(self):
    self.optimality_fun = _make_quadratic_prog_optimality_fun()

    # Set up implicit diff.
    decorator = idf.custom_root(self.optimality_fun, has_aux=True)
    # pylint: disable=g-missing-from-attributes
    self.run = decorator(self.run)
