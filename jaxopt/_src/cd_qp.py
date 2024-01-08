# Copyright 2022 Google LLC
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

"""Implementation of coordinate descent for box-constrained QPs."""

from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Union

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt._src import projection
from jaxopt._src import tree_util


class BoxCDQPState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  error: float


def fori_loop_body_fun(i, tup):
  x, Q, c, l, u, error = tup
  # i-th element of the gradient
  g_i = jnp.dot(Q[i], x) + c[i]
  # i-th diagonal element of the Hessian
  h_i = Q[i, i]
  # Newton-update and avoid division by zero
  update = jnp.where(h_i == 0, 0, g_i / h_i)
  # Newton-update + clipping to satisfy the box constraint
  x_i_new = jnp.clip(x[i] - update, l[i], u[i])
  delta_i = x_i_new - x[i]
  # Cumulated error
  error += jnp.abs(delta_i)
  x = x.at[i].set(x_i_new)
  return x, Q, c, l, u, error


@dataclass(eq=False)
class BoxCDQP(base.IterativeSolver):
  """Coordinate descent solver for box-constrained QPs.

  This solver minimizes::

    0.5 <x, Qx> + <c, x> subject to l <= x <= u

  Attributes:
    maxiter: maximum number of coordinate descent iterations.
    tol: tolerance to use.
    verbose: whether to print information on every iteration or not.

    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.

    jit: whether to JIT-compile the optimization loop (default: True).
    unroll: whether to unroll the optimization loop (default: "auto").
  """
  maxiter: int = 500
  tol: float = 1e-4
  verbose: Union[bool, int] = False
  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None
  jit: bool = True
  unroll: base.AutoOrBoolean = "auto"

  def init_state(self,
                 init_params: jnp.ndarray,
                 params_obj: Optional[base.ArrayPair] = None,
                 params_ineq: Optional[base.ArrayPair] = None) -> BoxCDQPState:
    """Initialize the solver state.

    Args:
      init_params: array containing the initial parameters.
      params_obj: Tuple of arrays ``(Q, c)``.
      params_ineq: Tuple of arrays ``(l, u)``.
    Returns:
      state
    """
    del params_obj, params_ineq  # Not used.
    return BoxCDQPState(iter_num=jnp.asarray(0),
                        error=jnp.asarray(jnp.inf))

  def update(self,
             params: jnp.ndarray,
             state: NamedTuple,
             params_obj: base.ArrayPair,
             params_ineq: base.ArrayPair) -> base.OptStep:
    """Performs one epoch of coordinate descent.

    Args:
      params: array containing the parameters.
      state: named tuple containing the solver state.
      params_obj: Tuple of arrays ``(Q, c)``.
      params_ineq: Tuple of arrays ``(l, u)``.
    Returns:
      (params, state)
    """
    Q, c = params_obj
    l, u = params_ineq

    init = (params, Q, c, l, u, 0)

    # todo: ability to permute coordinate order.
    params, _, _, _, _, error = jax.lax.fori_loop(lower=0,
                                                  upper=params.shape[0],
                                                  body_fun=fori_loop_body_fun,
                                                  init_val=init)

    state = BoxCDQPState(iter_num=state.iter_num + 1, error=error)

    if self.verbose:
      self.log_info(state)
    return base.OptStep(params=params, state=state)

  def _fixed_point_fun(self,
                       sol: jnp.ndarray,
                       params_obj: base.ArrayPair,
                       params_ineq: base.ArrayPair) -> jnp.ndarray:
    Q, c = params_obj
    l, u = params_ineq
    grad = jnp.dot(Q, sol) + c
    return projection.projection_box(sol - grad, (l, u))

  def optimality_fun(self,
                     sol: jnp.ndarray,
                     params_obj: base.ArrayPair,
                     params_ineq: base.ArrayPair) -> jnp.ndarray:
    return self._fixed_point_fun(sol, params_obj, params_ineq) - sol

  def __post_init__(self):
    super().__post_init__()
