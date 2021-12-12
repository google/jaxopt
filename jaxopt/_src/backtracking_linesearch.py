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

"""Bactracking line search algorithm."""

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_vdot


class BacktrackingLineSearchState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  value: float
  error: float
  params: Any
  grad: Any


@dataclass(eq=False)
class BacktrackingLineSearch(base.IterativeLineSearch):
  """Backtracking line search.

  Attributes:
    fun: a function of the form ``fun(params, *args, **kwargs)``, where
      ``params`` are parameters of the model,
      ``*args`` and ``**kwargs`` are additional arguments.
    value_and_grad: if ``False``, ``fun`` should return the function value only.
      If ``True``, ``fun`` should return both the function value and the
      gradient.

    condition: either "strong-wolfe" or "wolfe".
    c1: constant used by the (strong) Wolfe condition.
    c2: constant strictly less than 1 used by the (strong) Wolfe condition.
    decrease_factor: factor by which to decrease the stepsize during line search
      (default: 0.8).

    maxiter: maximum number of line search iterations.
    tol: tolerance of the stopping criterion.

    verbose: whether to print error on every iteration or not. verbose=True will
      automatically disable jit.

    jit: whether to JIT-compile the optimization loop (default: "auto").
    unroll: whether to unroll the optimization loop (default: "auto").
  """
  fun: Callable
  value_and_grad: bool = False

  maxiter: int = 30
  tol: float = 0.0

  condition: str = "strong-wolfe"
  c1: float = 1e-4
  c2: float = 0.9
  decrease_factor: float = 0.8

  verbose: int = 0
  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"

  def init_state(self,
                 init_stepsize: float,
                 params: Any,
                 value: Optional[float] = None,
                 grad: Optional[Any] = None,
                 descent_direction: Optional[Any] = None,
                 *args,
                 **kwargs) -> BacktrackingLineSearchState:
    """Initialize the line search state.

    Args:
      init_stepsize: initial step size value.
      params: current parameters.
      value: current function value (recomputed if None).
      grad: current gradient (recomputed if None).
      descent_direction: ignored.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      state
    """
    del descent_direction  # Not used.

    if value is None or grad is None:
      value, grad = self._value_and_grad_fun(params, *args, **kwargs)

    return BacktrackingLineSearchState(iter_num=jnp.asarray(0),
                                       value=value,
                                       error=jnp.asarray(jnp.inf),
                                       params=params,
                                       grad=grad)

  def update(self,
             stepsize: float,
             state: NamedTuple,
             params: Any,
             value: Optional[float] = None,
             grad: Optional[Any] = None,
             descent_direction: Optional[Any] = None,
             *args,
             **kwargs) -> base.LineSearchStep:
    """Performs one iteration of backtracking line search.

    Args:
      stepsize: current estimate of the step size.
      state: named tuple containing the line search state.
      params: current parameters.
      value: current function value (recomputed if None).
      grad: current gradient (recomputed if None).
      descent_direction: descent direction (negative gradient if None).
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      (params, state)
    """
    if value is None or grad is None:
      value, grad = self._value_and_grad_fun(params, *args, **kwargs)

    if descent_direction is None:
      descent_direction = tree_scalar_mul(-1, grad)

    gd_vdot = tree_vdot(grad, descent_direction)
    new_params = tree_add_scalar_mul(params, stepsize, descent_direction)
    new_value, new_grad = self._value_and_grad_fun(new_params, *args, **kwargs)
    new_gd_vdot = tree_vdot(new_grad, descent_direction)

    # Armijo condition (upper bound on admissible step size).
    # cond1 = new_value <= value + self.c1 * stepsize * gd_vdot
    # See equation (3.6a), Numerical Optimization, Second edition.
    diff_cond1 = new_value - value + self.c1 * stepsize * gd_vdot
    error_cond1 = jax.lax.max(diff_cond1, 0.0)

    if self.condition == "strong-wolfe":
      # cond2 = abs(new_gd_vdot) <= c2 * abs(gd_vdot)
      # See equation (3.7b), Numerical Optimization, Second edition.
      diff_cond2 = jnp.abs(new_gd_vdot) - self.c2 * jnp.abs(gd_vdot)
      error_cond2 = jax.lax.max(diff_cond2, 0.0)

    elif self.condition == "wolfe":
      # cond2 = new_gd_vdot >= c2 * gd_vdot
      # See equation (3.6b), Numerical Optimization, Second edition.
      diff_cond2 = self.c2 * gd_vdot - new_gd_vdot
      error_cond2 = jax.lax.max(diff_cond2, 0.0)

    else:
      raise ValueError("condition should be 'strong-wolfe' or 'wolfe'.")

    error = jax.lax.max(error_cond1, error_cond2)

    new_stepsize = jnp.where(error <= self.tol,
                             stepsize,
                             stepsize * self.decrease_factor)

    new_state = BacktrackingLineSearchState(iter_num=state.iter_num + 1,
                                            value=new_value,
                                            grad=new_grad,
                                            params=new_params,
                                            error=error)

    return base.LineSearchStep(stepsize=new_stepsize, state=new_state)

  def __post_init__(self):
    if self.value_and_grad:
      self._value_and_grad_fun = self.fun
    else:
      self._value_and_grad_fun = jax.value_and_grad(self.fun)
