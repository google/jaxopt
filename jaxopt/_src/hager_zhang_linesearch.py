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

"""Hager-Zhang line search algorithm."""

# This is based on:
# [1] W. Hager, H. Zhang, A new conjugate gradient method with guaranteed
# descent and an efficient line search. SIAM J. Optim., Vol 16. 1, pp. 170-172.
# 2005. https://www.math.lsu.edu/~hozhang/papers/cg_descent.pdf
#
# Algorithm details are from
# [2] W. Hager, H. Zhang, Algorithm 851: CG_DESCENT, a Conjugate Gradient Method
# with Guaranteed Descent.
# https://www.math.lsu.edu/~hozhang/papers/cg_compare.pdf

from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Union

import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_vdot_real
from jaxopt.tree_util import tree_conj


def _failed_nan(value, grad):
  return jnp.isnan(value) | jnp.isnan(grad)


class HagerZhangLineSearchState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  done: bool
  low: float
  high: float
  value: float
  error: float
  params: Any
  grad: Any
  failed: bool
  aux: Optional[Any] = None

  num_fun_eval: int = 0
  num_grad_eval: int = 0


@dataclass(eq=False)
class HagerZhangLineSearch(base.IterativeLineSearch):
  """Hager-Zhang line search.

  Supports complex variables.

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

    c1: constant used by the Wolfe and Approximate Wolfe condition.
    c2: constant strictly less than 1 used by the Wolfe and Approximate Wolfe
      condition.
    max_stepsize: upper bound on stepsize (unused).

    maxiter: maximum number of line search iterations.
    tol: tolerance of the stopping criterion.

    verbose: whether to print information on every iteration or not.

    jit: whether to JIT-compile the optimization loop (default: "auto").
    unroll: whether to unroll the optimization loop (default: "auto").
  """
  fun: Callable  # pylint:disable=g-bare-generic
  value_and_grad: Union[bool, Callable] = False
  has_aux: bool = False

  maxiter: int = 30
  tol: float = 0.

  c1: float = 0.1
  c2: float = 0.9
  expansion_factor: float = 5.0
  shrinkage_factor: float = 0.66
  approximate_wolfe_threshold = 1e-6
  # TODO(vroulet): remove max_stepsize argument as it is not used
  max_stepsize: float = 1.0 

  verbose: Union[bool, int] = False
  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"

  def _value_and_grad_on_line(self, x, c, descent_direction, *args, **kwargs):
    z = tree_add_scalar_mul(x, c, descent_direction)
    (value, _), grad = self._value_and_grad_fun_with_aux(z, *args, **kwargs)
    return value, tree_vdot_real(tree_conj(grad), descent_direction)

  def _satisfies_wolfe_and_approx_wolfe(
      self,
      c,
      value_c,
      gd_vdot_c,
      value_initial,
      grad_initial,
      approx_wolfe_threshold_value,
      descent_direction):
    gd_vdot = tree_vdot_real(tree_conj(grad_initial), descent_direction)

    # Armijo condition
    # armijo = value_c <= value_initial + self.c1 * c * gd_vdot
    armijo = value_c - (value_initial + self.c1 * c * gd_vdot)
    armijo_error = jax.lax.max(armijo, 0.)
    # Curvature condition
    # curvature = gd_vdot_c >= self.c2 * gd_vdot
    curvature = self.c2 * gd_vdot - gd_vdot_c
    wolfe_error = jax.lax.max(armijo_error, curvature)

    # Approximate Wolfe
    # approx_wolfe = (2 * self.c1 - 1.) * gd_vdot >= gd_vdot_c
    approx_wolfe = gd_vdot_c - (2 * self.c1 - 1.) * gd_vdot
    approx_wolfe_error = jax.lax.max(approx_wolfe, 0.)
    approx_wolfe_error = jax.lax.max(approx_wolfe_error, curvature)
    # Finally only enable the approximate wolfe conditions when we are close in
    # value.
    approx_wolfe_error = jax.lax.max(
        approx_wolfe_error,
        (value_c - approx_wolfe_threshold_value))
    # We succeed if we either satisfy the Wolfe conditions or the approximate
    # Wolfe conditions.
    return jax.lax.min(wolfe_error, approx_wolfe_error)

  def _update(
      self, x, low, high, middle, approx_wolfe_threshold_value,
      descent_direction, fun_args, fun_kwargs):

    value_middle, grad_middle = self._value_and_grad_on_line(
        x, middle, descent_direction, *fun_args, **fun_kwargs)

    # Corresponds to the `update` subroutine in the paper.
    # This tries to create a smaller interval contained in `[low, high]`
    # from the point `middle` that satisfies the opposite slope condition, where
    # the left end point is equal to within tolerance of the initial value.

    def cond_fn(state):
      done, failed, low, middle, high, *_ = state
      return jnp.any((middle < high) & (middle > low) & ~done & ~failed)

    def body_fn(state):
      done, failed, low, middle, high, value_middle, grad_middle, it = state
      # Correspond to U1 in the paper.
      update_right_endpoint = grad_middle >= 0.
      new_high = jnp.where(~done & update_right_endpoint, middle, high)
      done = done | update_right_endpoint

      # Correspond to U2 in the paper.
      update_left_endpoint = value_middle <= approx_wolfe_threshold_value
      # Note that ~done implies grad_middle < 0. which is necessary for this
      # check.
      new_low = jnp.where(~done & update_left_endpoint, middle, low)
      done = done | update_left_endpoint

      # Correspond to U3 in the paper.
      new_high = jnp.where(~done, middle, new_high)
      done = done | jnp.isneginf(value_middle)

      # TODO(srvasude): Allow this parameter to be varied.
      new_middle = jnp.where(~done, (low + high) / 2., middle)

      new_value_middle, new_grad_middle = self._value_and_grad_on_line(
          x, new_middle, descent_direction, *fun_args, **fun_kwargs)
      new_value_middle = jnp.where(~done, new_value_middle, value_middle)
      new_grad_middle = jnp.where(~done, new_grad_middle, grad_middle)
      failed = failed | _failed_nan(new_value_middle, new_grad_middle)

      return (done,
              failed,
              new_low,
              new_middle,
              new_high,
              new_value_middle,
              new_grad_middle,
              it + 1)

    _, failed, final_low, _, final_high, _, _, nit = jax.lax.while_loop(
        cond_fn,
        body_fn,
        ((middle >= high) | (middle <= low),
         _failed_nan(value_middle, grad_middle),
         low,
         middle,
         high,
         value_middle,
         grad_middle,
         0))
    num_fun_grad_calls = nit + 1
    return failed, final_low, final_high, num_fun_grad_calls

  def _secant(self, x, low, high, descent_direction, *args, **kwargs):
    _, dlow = self._value_and_grad_on_line(
        x, low, descent_direction, *args, **kwargs)
    _, dhigh = self._value_and_grad_on_line(
        x, high, descent_direction, *args, **kwargs)
    return (low * dhigh - high * dlow) / (dhigh  - dlow)

  def _secant2(
      self, x, low, high,
      approx_wolfe_threshold_value, descent_direction, *args, **kwargs):

    # Corresponds to the secant^2 routine in the paper.

    c = self._secant(x, low, high, descent_direction, *args, **kwargs)
    num_fun_grad_calls = 2
    failed, new_low, new_high, num_fun_grad_calls_update = self._update(
        x, low, high, c, approx_wolfe_threshold_value,
        descent_direction, args, kwargs)
    num_fun_grad_calls += num_fun_grad_calls_update
    on_left_boundary = jnp.equal(c, new_low)
    on_right_boundary = jnp.equal(c, new_high)
    c = jnp.where(on_right_boundary, self._secant(
        x, high, new_high, descent_direction, *args, **kwargs), c)
    c = jnp.where(on_left_boundary, self._secant(
        x, low, new_low, descent_direction, *args, **kwargs), c)
    num_fun_grad_calls += 4

    def _reupdate():
      return self._update(
          x, new_low, new_high, c, approx_wolfe_threshold_value,
          descent_direction, args, kwargs)

    failed, new_low, new_high, num_fun_grad_calls_update = jax.lax.cond(
        on_left_boundary | on_right_boundary,
        _reupdate, lambda: (failed, new_low, new_high, 0))
    num_fun_grad_calls += num_fun_grad_calls_update
    return failed, new_low, new_high, num_fun_grad_calls

  def _bracket(
      self, x, c, approx_wolfe_threshold_value,
      descent_direction, *args, **kwargs):
    # Initial interval that satisfies the opposite slope condition.

    def cond_fn(state):
      return jnp.any(~state[0]) & ~jnp.all(state[1])

    def body_fn(state):
      (done,
       failed,
       low,
       middle,
       high,
       value_middle,
       grad_middle,
       best_middle,
       num_fun_grad_calls) = state
      # Correspond to B1 in the paper.
      update_right_endpoint = grad_middle >= 0.
      new_high = jnp.where(~done & update_right_endpoint, middle, high)
      new_low = jnp.where(~done & update_right_endpoint, best_middle, low)
      done = done | update_right_endpoint

      # Correspond to B2 in the paper.
      # Note that ~done implies grad_middle < 0. at this point so we omit
      # checking that.
      reupdate = ~done & (value_middle > approx_wolfe_threshold_value)

      def _update_interval():
        return self._update(
            x,
            0,
            middle,
            middle / 2.,
            approx_wolfe_threshold_value,
            descent_direction, args, kwargs)

      new_failed, new_low, new_high, new_num_fun_grad_calls = jax.lax.cond(
          reupdate, _update_interval, lambda: (failed, new_low, new_high, 0))
      failed = failed | new_failed
      done = done | reupdate

      # This corresponds to the largest middle value that we have probed
      # so far, that also is 'valid' (decreases the function sufficiently).
      best_middle = jnp.where(
          ~done & (value_middle <= approx_wolfe_threshold_value),
          middle, best_middle)

      # Corresponds to B3 in the paper. Increase the point and recompute.
      new_middle = jnp.where(~done, self.expansion_factor * middle, middle)

      new_value_middle, new_grad_middle = self._value_and_grad_on_line(
          x, new_middle, descent_direction, *args, **kwargs)
      new_num_fun_grad_calls += 1
      num_fun_grad_calls += new_num_fun_grad_calls

      # Terminate on encountering NaNs to avoid an infinite loop.
      failed = failed  | _failed_nan(new_value_middle, new_grad_middle)
      return (done,
              failed,
              new_low,
              new_middle,
              new_high,
              new_value_middle,
              new_grad_middle,
              best_middle,
              num_fun_grad_calls)

    value_c, grad_c = self._value_and_grad_on_line(
        x, c, descent_direction, *args, **kwargs)
    num_fun_grad_calls = 1

    # We have failed if there is a NaN at the right endpoint, or the gradient is
    # NaN at the right endpoint (when there is a finite value).
    failed = _failed_nan(value_c, grad_c)
    # If the right endpoint is -inf, then we are done as this is a minima.
    done = jnp.isneginf(value_c)

    _, failed, final_low, _, final_high, _, _, _, new_num_fun_grad_calls = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (done,
         failed,
         jnp.array(0.),
         c,
         c,
         value_c,
         grad_c,
         jnp.array(0.),
         0))
    num_fun_grad_calls += new_num_fun_grad_calls
    return failed, final_low, final_high, num_fun_grad_calls

  def init_state(
      self,
      init_stepsize: float,
      params: Any,
      value: Optional[float] = None,
      grad: Optional[Any] = None,
      descent_direction: Optional[Any] = None,
      fun_args: list = [],
      fun_kwargs: dict = {},
  ) -> HagerZhangLineSearchState:
    """Initialize the line search state.

    Args:
      init_stepsize: initial step size value. This is ignored by the linesearch.
      params: current parameters.
      value: current function value (recomputed if None).
      grad: current gradient (recomputed if None).
      descent_direction: ignored.
      fun_args: additional positional arguments to be passed to ``fun``.
      fun_kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      state
    """
    del init_stepsize

    if value is None or grad is None:
      (value, _), grad = self._value_and_grad_fun_with_aux(
          params, *fun_args, **fun_kwargs
      )
      num_fun_eval = 1
      num_grad_eval = 1
    else:
      num_fun_eval = 0
      num_grad_eval = 0


    if descent_direction is None:
      descent_direction = tree_scalar_mul(-1, tree_conj(grad))

    approx_wolfe_threshold_value = (
        value + self.approximate_wolfe_threshold * jnp.abs(value))

    # Create initial interval.
    failed, low, high, num_fun_grad_calls = self._bracket(
        params,
        jnp.ones_like(value),
        approx_wolfe_threshold_value,
        descent_direction,
        *fun_args,
        **fun_kwargs
    )
    num_fun_eval += num_fun_grad_calls
    num_grad_eval += num_fun_grad_calls

    value_low, grad_low = self._value_and_grad_on_line(
        params, low, descent_direction, *fun_args, **fun_kwargs)

    value_high, grad_high = self._value_and_grad_on_line(
        params, high, descent_direction, *fun_args, **fun_kwargs)
    num_fun_eval += 2
    num_grad_eval +=  2

    best_point = jnp.where(value_low < value_high, low, high)
    gd_vdot_best_point = jnp.where(value_low < value_high, grad_low, grad_high)
    value_best_point = jnp.minimum(value_low, value_high)

    error = self._satisfies_wolfe_and_approx_wolfe(
        best_point,
        value_best_point,
        gd_vdot_best_point,
        value,
        grad,
        approx_wolfe_threshold_value,
        descent_direction)
    done = error <= self.tol

    return HagerZhangLineSearchState(
        iter_num=jnp.asarray(0),
        low=low,
        high=high,
        error=error,
        done=done,
        value=value,
        aux=None,  # we do not need to have aux in the initial state
        params=params,
        grad=grad,
        failed=failed,
        num_fun_eval=jnp.array(num_fun_eval, base.NUM_EVAL_DTYPE),
        num_grad_eval=jnp.array(num_grad_eval, base.NUM_EVAL_DTYPE))

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
    """Performs one iteration of Hager-Zhang line search.

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
    new_num_fun_eval = state.num_fun_eval
    new_num_grad_eval = state.num_grad_eval
    if value is None or grad is None:
      (value, _), grad = self._value_and_grad_fun_with_aux(
          params, *fun_args, **fun_kwargs
      )
      new_num_fun_eval = new_num_fun_eval + 1
      new_num_grad_eval = new_num_grad_eval + 1

    if descent_direction is None:
      descent_direction = tree_scalar_mul(-1, tree_conj(grad))

    approx_wolfe_threshold_value = (
        value + self.approximate_wolfe_threshold * jnp.abs(value))

    failed, new_low, new_high, num_fun_grad_calls = self._secant2(
        params,
        state.low,
        state.high,
        approx_wolfe_threshold_value,
        descent_direction,
        *fun_args, **fun_kwargs)
    new_num_fun_eval += num_fun_grad_calls
    new_num_grad_eval += num_fun_grad_calls

    failed = state.failed | failed

    new_low = jnp.where(state.done, state.low, new_low)
    new_high = jnp.where(state.done, state.high, new_high)

    def _reupdate():
      c = (new_low + new_high) / 2.
      return self._update(
          params, new_low, new_high, c, approx_wolfe_threshold_value,
          descent_direction, fun_args, fun_kwargs)

    failed, new_low, new_high, num_fun_grad_calls = jax.lax.cond(
        ~state.done & ((new_high - new_low) >
                       (self.shrinkage_factor * (state.high - state.low))),
        _reupdate, lambda: (failed, new_low, new_high, 0))
    new_num_fun_eval += num_fun_grad_calls
    new_num_grad_eval += num_fun_grad_calls

    # Check wolfe and approximate wolfe conditions and update them.

    value_low, grad_low = self._value_and_grad_on_line(
        params, new_low, descent_direction, *fun_args, **fun_kwargs)
    value_high, grad_high = self._value_and_grad_on_line(
        params, new_high, descent_direction, *fun_args, **fun_kwargs)
    new_num_fun_eval += 2
    new_num_grad_eval += 2

    best_point = jnp.where(value_low < value_high, new_low, new_high)
    gd_vdot_best_point = jnp.where(value_low < value_high, grad_low, grad_high)

    value_best_point = jnp.minimum(value_low, value_high)

    new_stepsize = jnp.where(state.done, stepsize, best_point)
    new_params = tree_add_scalar_mul(params, best_point, descent_direction)
    (new_value, new_aux), new_grad = self._value_and_grad_fun_with_aux(
        new_params, *fun_args, **fun_kwargs
    )
    new_num_fun_eval += 1
    new_num_grad_eval += 1

    error = jnp.where(state.done, state.error,
                      self._satisfies_wolfe_and_approx_wolfe(
                          best_point,
                          value_best_point,
                          gd_vdot_best_point,
                          value,
                          grad,
                          approx_wolfe_threshold_value,
                          descent_direction))
    done = state.done | (error <= self.tol)
    failed = failed | ((state.iter_num + 1 == self.maxiter) & ~done)

    new_state = HagerZhangLineSearchState(
        iter_num=state.iter_num + 1,
        value=new_value,
        grad=new_grad,
        aux=new_aux,
        params=new_params,
        low=new_low,
        high=new_high,
        error=error,
        done=done,
        failed=failed,
        num_fun_eval=new_num_fun_eval,
        num_grad_eval=new_num_grad_eval)

    if self.verbose:
      self.log_info(
        new_state,
        error_name="Minimum Decrease & Curvature Errors",
        additional_info={
            "Stepsize": new_stepsize,
            "Objective Value": new_value
        }
      )

    return base.LineSearchStep(stepsize=new_stepsize, state=new_state)

  def __post_init__(self):
    _, _, self._value_and_grad_fun_with_aux = base._make_funs_with_aux(self.fun, self.value_and_grad, self.has_aux)
