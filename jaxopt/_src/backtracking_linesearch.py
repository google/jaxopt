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
from typing import Union

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt._src.cond import cond
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_conj
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_vdot_real


class BacktrackingLineSearchState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  params: Any
  value: float
  grad: Any  # either initial or final for armijo or glodstein
  value_init: float 
  grad_init: Any
  error: float
  done: bool
  failed: bool
  num_fun_eval: int
  num_grad_eval: int
  aux: Optional[Any] = None


@dataclass(eq=False)
class BacktrackingLineSearch(base.IterativeLineSearch):
  """Backtracking line search.

  Supports complex variables.

  Attributes:
    fun: a function of the form ``fun(params, *args, **kwargs)``, where
      ``params`` are parameters of the model,
      ``*args`` and ``**kwargs`` are additional arguments.
    value_and_grad: if ``False``, ``fun`` should return the function value only.
      If ``True``, ``fun`` should return both the function value and the
      gradient. If it is a Callable, fun should return the value while value_and_grad
      returns value and gradient of the objective.
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

    verbose: whether to print information on every iteration or not.

    jit: whether to JIT-compile the optimization loop (default: "auto").
    unroll: whether to unroll the optimization loop (default: "auto").
  """
  fun: Callable
  value_and_grad: Union[bool, Callable] = False
  has_aux: bool = False

  maxiter: int = 30
  tol: float = 0.0

  condition: str = "strong-wolfe"
  c1: float = 1e-4
  c2: float = 0.9
  decrease_factor: float = 0.8
  max_stepsize: float = 1.0

  verbose: Union[bool, int] = False
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
    del init_stepsize  # Not used.

    num_fun_eval = jnp.asarray(0, base.NUM_EVAL_DTYPE)
    num_grad_eval = jnp.asarray(0, base.NUM_EVAL_DTYPE)
    if value is None or grad is None:
      (value, _), grad = self._value_and_grad_fun_with_aux(
          params, *fun_args, **fun_kwargs
      )
      num_fun_eval += 1
      num_grad_eval += 1

    return BacktrackingLineSearchState(iter_num=jnp.asarray(0),
                                       params=params,
                                       value=value,
                                       grad=grad,
                                       value_init=value,
                                       grad_init=grad,
                                       aux=None,  # we do not need to have aux
                                       # in the initial state
                                       error=jnp.asarray(jnp.inf),
                                       done=jnp.asarray(False),
                                       failed=jnp.asarray(False),
                                       num_fun_eval=num_fun_eval,
                                       num_grad_eval=num_grad_eval)

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
      value: current function value (computed at initialization, unused here).
      grad: current gradient (computed at initialization, unused here).
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

    # Grab value and grad from initialization and avoid recomputing them
    del value
    del grad
    value = state.value_init
    grad = state.grad_init

    if descent_direction is None:
      descent_direction = tree_scalar_mul(-1, tree_conj(grad))

    slope = tree_vdot_real(tree_conj(grad), descent_direction)

    # For backtracking linesearches, we want to compute the next point
    # from the basepoint. i.e. x_i  = x_0 + s_i * p
    new_params = tree_add_scalar_mul(params, stepsize, descent_direction)
    # Every condition requires the new function value, but not every one
    # requires the new gradient value (we'll assume that this code is called
    # under `jit`).
    num_fun_eval += 1
    if self.condition in ["armijo", "goldstein"]:
      new_value, new_aux = self._fun_with_aux(
          new_params, *fun_args, **fun_kwargs
      )
      # For those conditions, no need to compute a new grad
      # We recompute a new_grad only once we have found the right stepsize,
      # see below
      new_grad, new_slope = grad, slope
    else:
      (new_value, new_aux), new_grad = self._value_and_grad_fun_with_aux(
          new_params, *fun_args, **fun_kwargs
      )
      new_slope = tree_vdot_real(tree_conj(new_grad), descent_direction)
      num_grad_eval += 1

    # Armijo condition (upper bound on admissible step size).
    # cond1 = new_value <= value + self.c1 * stepsize * slope
    # See equation (3.6a), Numerical Optimization, Second edition.
    diff_cond1 = new_value - (value + self.c1 * stepsize * slope)
    error_cond1 = jnp.where(jnp.isnan(diff_cond1), jnp.inf, diff_cond1)
    error_cond1 = jnp.maximum(error_cond1, 0.0)
    error = error_cond1

    if self.condition == "armijo":
      # We don't need to do any extra work, since this is covered by the
      # sufficient decrease condition above.
      pass

    elif self.condition == "strong-wolfe":
      # cond2 = abs(new_slope) <= c2 * abs(slope)
      # See equation (3.7b), Numerical Optimization, Second edition.
      diff_cond2 = jnp.abs(new_slope) - self.c2 * jnp.abs(slope)
      error_cond2 = jnp.where(jnp.isnan(diff_cond2), jnp.inf, diff_cond2)
      error_cond2 = jnp.maximum(error_cond2, 0.0)
      error = jnp.maximum(error_cond1, error_cond2)

    elif self.condition == "wolfe":
      # cond2 = new_slope >= c2 * slope
      # See equation (3.6b), Numerical Optimization, Second edition.
      diff_cond2 = self.c2 * slope - new_slope
      error_cond2 = jnp.where(jnp.isnan(diff_cond2), jnp.inf, diff_cond2)
      error_cond2 = jnp.maximum(error_cond2, 0.0)
      error = jnp.maximum(error_cond1, error_cond2)

    elif self.condition == "goldstein":
      # cond2 = new_value >= value + (1 - self.c1) * stepsize * slope
      diff_cond2 = value + (1 - self.c1) * stepsize * slope - new_value
      error_cond2 = jnp.where(jnp.isnan(diff_cond2), jnp.inf, diff_cond2)
      error_cond2 = jnp.maximum(error_cond2, 0.0)
      error = jnp.maximum(error_cond1, error_cond2)

    else:
      raise ValueError("condition should be one of "
                       "'armijo', 'goldstein', 'strong-wolfe' or 'wolfe'.")

    done = state.done | (error <= self.tol)
    failed = state.failed | ((state.iter_num + 1 == self.maxiter) & ~done)

    new_stepsize = jnp.where(done | failed,
                             stepsize,
                             stepsize * self.decrease_factor)

    if self.condition in ["armijo", "goldstein"]:
      # If we are done for the armijo or the goldstein conditions,
      # we compute the final gradient (we had not computed it before since 
      # these conditions did not require it)
      new_grad = cond(done | failed,
                      self._compute_final_grad,
                      lambda *_: grad, 
                      new_params, fun_args, fun_kwargs,
                      jit=self.jit)
      maybe_additional_eval = jnp.asarray(done | failed, dtype=base.NUM_EVAL_DTYPE)
      num_grad_eval = num_grad_eval + maybe_additional_eval
      # We a priori always access the function value when computing the gradient
      num_fun_eval = num_fun_eval + maybe_additional_eval

    new_state = state._replace(iter_num=state.iter_num + 1,
                               params=new_params,
                               value=new_value,
                               grad=new_grad,
                               aux=new_aux,
                               done=done,
                               error=error,
                               failed=failed,
                               num_fun_eval=num_fun_eval,
                               num_grad_eval=num_grad_eval)

    if self.verbose:
      additional_info = {'Stepsize': stepsize, 'Objective Value': new_value}
      if self.condition != 'armijo':
        error_name = "Minimum Decrease & Curvature Errors"
        additional_info.update({'Decrease Error': error_cond1})
      else:
        error_name = "Decrease Error"
      self.log_info(
          new_state,
          error_name=error_name,
          additional_info=additional_info
      )
    return base.LineSearchStep(stepsize=new_stepsize, state=new_state)
  
  def _compute_final_grad(self, params, fun_args, fun_kwargs):
    return self._grad_with_aux(params, *fun_args, **fun_kwargs)[0]

  def __post_init__(self):
    self._fun_with_aux, self._grad_with_aux, self._value_and_grad_fun_with_aux = \
        base._make_funs_with_aux(fun=self.fun,
                                 value_and_grad=self.value_and_grad,
                                 has_aux=self.has_aux)
