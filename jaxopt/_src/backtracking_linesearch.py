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
  done: bool
  params: Any
  grad: Any
  num_fun_eval: int
  num_grad_eval: int
  failed: bool
  aux: Optional[Any] = None


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
    has_aux: if ``False``, ``fun`` should return the function value only.
      If ``True``, ``fun`` should return a pair ``(value, aux)`` where ``aux``
      is a pytree of auxiliary values.

    condition: either "armijo", "goldstein", "strong-wolfe" or "wolfe".
    c1: constant used by the (strong) Wolfe condition.
    c2: constant strictly less than 1 used by the (strong) Wolfe condition.
    decrease_factor: factor by which to decrease the stepsize during line search
      (default: 0.8).
    max_stepsize: upper bound on stepsize.

    maxiter: maximum number of line search iterations.
    tol: tolerance of the stopping criterion.

    verbose: whether to print error on every iteration or not. verbose=True will
      automatically disable jit.

    jit: whether to JIT-compile the optimization loop (default: "auto").
    unroll: whether to unroll the optimization loop (default: "auto").
  """
  fun: Callable
  value_and_grad: bool = False
  has_aux: bool = False

  maxiter: int = 30
  tol: float = 0.0

  condition: str = "strong-wolfe"
  c1: float = 1e-4
  c2: float = 0.9
  decrease_factor: float = 0.8
  max_stepsize: float = 1.0

  verbose: int = 0
  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"

  def init_state(
      self,
      init_stepsize: float,
      params: Any,
      value: Optional[float] = None,
      grad: Optional[Any] = None,
      descent_direction: Optional[Any] = None,
      fun_args: list = [],
      fun_kwargs: dict = {},
  ) -> BacktrackingLineSearchState:
    """Initialize the line search state.

    Args:
      init_stepsize: initial step size value.
      params: current parameters.
      value: current function value (recomputed if None).
      grad: current gradient (recomputed if None).
      descent_direction: ignored.
      fun_args: additional positional arguments to be passed to ``fun``.
      fun_kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      state
    """
    del descent_direction  # Not used.

    num_fun_eval = 0
    num_grad_eval = 0
    if value is None or grad is None:
      if self.has_aux:
        (value, _), grad = self._value_and_grad_fun(
            params, *fun_args, **fun_kwargs
        )
      else:
        value, grad = self._value_and_grad_fun(params, *fun_args, **fun_kwargs)
      num_fun_eval += 1
      num_grad_eval += 1

    return BacktrackingLineSearchState(iter_num=jnp.asarray(0),
                                       value=value,
                                       aux=None,  # we do not need to have aux
                                       # in the initial state
                                       error=jnp.asarray(jnp.inf),
                                       params=params,
                                       num_fun_eval=num_fun_eval,
                                       num_grad_eval=num_grad_eval,
                                       done=jnp.asarray(False),
                                       grad=grad,
                                       failed=jnp.asarray(False))

  def update(
      self,
      stepsize: float,
      state: NamedTuple,
      params: Any,
      value: Optional[float] = None,
      grad: Optional[Any] = None,
      descent_direction: Optional[Any] = None,
      fun_args: list = [],
      fun_kwargs: dict = {},
  ) -> base.LineSearchStep:
    """Performs one iteration of backtracking line search.

    Args:
      stepsize: current estimate of the step size.
      state: named tuple containing the line search state.
      params: current parameters.
      value: current function value (recomputed if None).
      grad: current gradient (recomputed if None).
      descent_direction: descent direction (negative gradient if None).
      fun_args: additional positional arguments to be passed to ``fun``.
      fun_kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      (params, state)
    """
    # Ensure that stepsize does not exceed upper bound.
    stepsize = jnp.minimum(self.max_stepsize, stepsize)
    num_fun_eval = state.num_fun_eval
    num_grad_eval = state.num_grad_eval

    if value is None or grad is None:
      if self.has_aux:
        (value, _), grad = self._value_and_grad_fun(params, *fun_args, **fun_kwargs)
      else:
        value, grad = self._value_and_grad_fun(params, *fun_args, **fun_kwargs)
      num_fun_eval += 1
      num_grad_eval += 1

    if descent_direction is None:
      descent_direction = tree_scalar_mul(-1, grad)

    gd_vdot = tree_vdot(grad, descent_direction)
    # For backtracking linesearches, we want to compute the next point
    # from the basepoint. i.e. x_i  = x_0 + s_i * p
    new_params = tree_add_scalar_mul(params, stepsize, descent_direction)
    if self.has_aux:
      (new_value, new_aux), new_grad = self._value_and_grad_fun(
          new_params, *fun_args, **fun_kwargs
      )
    else:
      new_value, new_grad = self._value_and_grad_fun(
          new_params, *fun_args, **fun_kwargs
      )
      new_aux = None
    new_gd_vdot = tree_vdot(new_grad, descent_direction)

    # Every condition requires the new function value, but not every one
    # requires the new gradient value (we'll assume that this code is called
    # under `jit`).
    num_fun_eval += 1

    # Armijo condition (upper bound on admissible step size).
    # cond1 = new_value <= value + self.c1 * stepsize * gd_vdot
    # See equation (3.6a), Numerical Optimization, Second edition.
    diff_cond1 = new_value - (value + self.c1 * stepsize * gd_vdot)
    error_cond1 = jnp.maximum(diff_cond1, 0.0)

    error = error_cond1

    if self.condition == "armijo":
      # We don't need to do any extra work, since this is covered by the
      # sufficient decrease condition above.
      pass

    elif self.condition == "strong-wolfe":
      # cond2 = abs(new_gd_vdot) <= c2 * abs(gd_vdot)
      # See equation (3.7b), Numerical Optimization, Second edition.
      diff_cond2 = jnp.abs(new_gd_vdot) - self.c2 * jnp.abs(gd_vdot)
      error_cond2 = jnp.maximum(diff_cond2, 0.0)
      error = jnp.maximum(error_cond1, error_cond2)
      num_grad_eval += 1

    elif self.condition == "wolfe":
      # cond2 = new_gd_vdot >= c2 * gd_vdot
      # See equation (3.6b), Numerical Optimization, Second edition.
      diff_cond2 = self.c2 * gd_vdot - new_gd_vdot
      error_cond2 = jnp.maximum(diff_cond2, 0.0)
      error = jnp.maximum(error_cond1, error_cond2)
      num_grad_eval += 1

    elif self.condition == "goldstein":
      # cond2 = new_value >= value + (1 - self.c1) * stepsize * gd_vdot
      diff_cond2 = value + (1 - self.c1) * stepsize * gd_vdot - new_value
      error_cond2 = jnp.maximum(diff_cond2, 0.0)
      error = jnp.maximum(error_cond1, error_cond2)

    else:
      raise ValueError("condition should be one of "
                       "'armijo', 'goldstein', 'strong-wolfe' or 'wolfe'.")

    new_stepsize = jnp.where(error <= self.tol,
                             stepsize,
                             stepsize * self.decrease_factor)
    done = state.done | (error <= self.tol)
    failed = state.failed | ((state.iter_num + 1 == self.maxiter) & ~done)

    new_state = BacktrackingLineSearchState(iter_num=state.iter_num + 1,
                                            value=new_value,
                                            aux=new_aux,
                                            grad=new_grad,
                                            params=new_params,
                                            num_fun_eval=num_fun_eval,
                                            num_grad_eval=num_grad_eval,
                                            done=done,
                                            error=error,
                                            failed=failed)

    return base.LineSearchStep(stepsize=new_stepsize, state=new_state)

  def __post_init__(self):
    if self.value_and_grad:
      self._value_and_grad_fun = self.fun
    else:
      self._value_and_grad_fun = jax.value_and_grad(
          self.fun, has_aux=self.has_aux
      )
