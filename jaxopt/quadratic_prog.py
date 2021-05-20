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

import jax.numpy as jnp

from jaxopt import implicit_diff
from jaxopt import linear_solve


def _check_params(params_obj, params_eq=None, params_ineq=None):
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


def make_solver_fun():
  """Creates a quadratic programming solver.

  The objective function is::

    0.5 * x^T Q x + c^T x subject to Gx <= h, Ax = b.

  Returns:
    sol = solver_fun(params_obj, params_eq, params_ineq=None) where
      params_obj = (Q, c)
      params_eq = (A, b)
      params_ineq = (G, h) or None
      sol = (primal_var, dual_var_eq, dual_var_ineq)
  """
  def solver_fun(params_obj, params_eq, params_ineq=None):
    _check_params(params_obj, params_eq=params_eq, params_ineq=params_ineq)

    if params_ineq is None:
      primal, dual_eq = _solve_eq_constrained_qp(params_obj, params_eq)
      return primal, dual_eq, None
    else:
      return _solve_constrained_qp_cvxpy(params_obj, params_eq, params_ineq)

  fun = implicit_diff.make_quadratic_prog_optimality_fun()
  return implicit_diff.custom_root(fun, unpack_params=True)(solver_fun)
