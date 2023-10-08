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
"""CVXPY wrappers."""

from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple

from dataclasses import dataclass

import jax
import jax.numpy as jnp 
from jaxopt._src import base
from jaxopt._src import implicit_diff as idf
from jaxopt._src import linear_solve
from jaxopt._src import tree_util


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


def _make_cvxpy_qp_optimality_fun():
  """Makes the optimality function for CVXPY quadratic programming.

  Returns:
    optimality_fun(params, params_obj, params_eq, params_ineq) where
      params = (primal_var, eq_dual_var, ineq_dual_var)
      params_obj = (Q, c)
      params_eq = (A, b) or None
      params_ineq = (G, h) or None
  """
  def obj_fun(primal_var, params_obj):
    Q, c = params_obj
    return 0.5 * jnp.vdot(primal_var, jnp.dot(Q, primal_var)) + jnp.vdot(primal_var, c)

  def eq_fun(primal_var, params_eq):
    if params_eq is None:
      return None
    A, b = params_eq
    return jnp.dot(A, primal_var) - b

  def ineq_fun(primal_var, params_ineq):
    if params_ineq is None:
      return None
    G, h = params_ineq
    return jnp.dot(G, primal_var) - h

  return idf.make_kkt_optimality_fun(obj_fun, eq_fun, ineq_fun)


@dataclass(eq=False)
class CvxpyQP(base.Solver):
  """Wraps CVXPY's quadratic solver with implicit diff support.

  No support for matvec, pytrees, jit and vmap.
  Meant to be run on CPU. Provide high precision solutions.

  The objective function is::

    0.5 * x^T Q x + c^T x subject to Gx <= h, Ax = b.
  
  Attributes:
    solver: string specifying the underlying solver used by Cvxpy, in ``"OSQP", "ECOS", "SCS"`` (default: ``"OSQP"``).
  """
  solver: str = 'OSQP'  #TODO(lbethune): "True" original OSQP implementation (not the one written in Jax). Confusing for user ?
  implicit_diff_solve: Optional[Callable] = None

  def run(self,
          init_params: Optional[jnp.ndarray],  # unused
          params_obj: base.ArrayPair,
          params_eq: Optional[base.ArrayPair] = None,
          params_ineq: Optional[base.ArrayPair] = None) -> base.OptStep:
    """Runs the quadratic programming solver in CVXPY.

    The returned params contains both the primal and dual solutions.

    Args:
      init_params: ignored.
      params_obj: (Q, c).
      params_eq: (A, b) or None if no equality constraints.
      params_ineq: (G, h) or None if no inequality constraints.
    Returns:
      (params, state), ``params = (primal_var, dual_var_eq, dual_var_ineq)``
    """
    # TODO(frostig,mblondel): experiment with `jax.experimental.host_callback`
    # to "support" other devices (GPU/TPU) in the interim, by calling into the
    # host CPU and running cvxpy there.
    #
    # TODO(lbethune): the interface of CVXPY could easily allow pytrees for constraints,
    # by populating `constraints` list with different Ai x = bi and Gi x <= hi.
    # Pytree support for x could be possible by creating a cp.Variable for each leaf in the pytree c.
    import cvxpy as cp

    del init_params  # no warm start
    _check_params(params_obj, params_eq, params_ineq)

    Q, c = params_obj
    x = cp.Variable(len(c))
    objective = 0.5 * cp.quad_form(x, Q) + c.T @ x

    constraints = []
    if params_eq is not None:
      A, b = params_eq
      constraints.append(A @ x == b)
    if params_ineq is not None:
      G, h = params_ineq
      constraints.append(G @ x <= h)

    pb = cp.Problem(cp.Minimize(objective), constraints)
    pb.solve(solver=self.solver)

    if pb.status in ["infeasible", "unbounded"]:
      raise ValueError("The problem is %s." % pb.status)

    dual_eq = None if params_eq is None else jnp.array(pb.constraints[0].dual_value)
    dual_ineq = None if params_ineq is None else jnp.array(pb.constraints[-1].dual_value)

    sol = base.KKTSolution(primal=jnp.array(x.value),
                           dual_eq=dual_eq,
                           dual_ineq=dual_ineq)

    # TODO(lbethune): pb.solver_stats is a "state" the user might be interested in.
    return base.OptStep(params=sol, state=None)

  def l2_optimality_error(
      self,
      params: jnp.ndarray,
      params_obj: base.ArrayPair,
      params_eq: Optional[base.ArrayPair],
      params_ineq: Optional[base.ArrayPair],
  ) -> base.OptStep:
    """Computes the L2 norm of the KKT residuals."""
    pytree = self.optimality_fun(params, params_obj, params_eq, params_ineq)
    return tree_util.tree_l2_norm(pytree)

  def __post_init__(self):
    self.optimality_fun = _make_cvxpy_qp_optimality_fun()

    # Set up implicit diff.
    decorator = idf.custom_root(self.optimality_fun, has_aux=True,
                                solve=self.implicit_diff_solve)
    # pylint: disable=g-missing-from-attributes
    self.run = decorator(self.run)
