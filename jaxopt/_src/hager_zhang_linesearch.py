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

import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_vdot


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
  aux: Optional[Any] = None


@dataclass(eq=False)
class HagerZhangLineSearch(base.IterativeLineSearch):
  """Hager-Zhang line search.

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
    max_stepsize: upper bound on stepsize.

    maxiter: maximum number of line search iterations.
    tol: tolerance of the stopping criterion.

    verbose: whether to print error on every iteration or not. verbose=True will
      automatically disable jit.

    jit: whether to JIT-compile the optimization loop (default: "auto").
    unroll: whether to unroll the optimization loop (default: "auto").
  """
  fun: Callable  # pylint:disable=g-bare-generic
  value_and_grad: bool = False
  has_aux: bool = False

  maxiter: int = 30
  tol: float = 0.

  c1: float = 0.1
  c2: float = 0.9
  expansion_factor: float = 5.0
  shrinkage_factor: float = 0.66
  approximate_wolfe_threshold = 1e-6
  max_stepsize: float = 1.0

  verbose: int = 0
  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"

  def _value_and_grad_on_line(self, x, c, descent_direction, *args, **kwargs):
    z = tree_add_scalar_mul(x, c, descent_direction)
    if self.has_aux:
      (value, _), grad = self._value_and_grad_fun(z, *args, **kwargs)
    else:
      value, grad = self._value_and_grad_fun(z, *args, **kwargs)
    return value, tree_vdot(grad, descent_direction)

  def _satisfies_wolfe_and_approx_wolfe(
      self,
      c,
      value_c,
      gd_vdot_c,
      value_initial,
      grad_initial,
      approx_wolfe_threshold_value,
      descent_direction):
    gd_vdot = tree_vdot(grad_initial, descent_direction)

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
      descent_direction, *args, **kwargs):

    value_middle, grad_middle = self._value_and_grad_on_line(
        x, middle, descent_direction, *args, **kwargs)

    # Corresponds to the `update` subroutine in the paper.
    # This tries to create a smaller interval contained in `[low, high]`
    # from the point `middle` that satisfies the opposite slope condition, where
    # the left end point is equal to within tolerance of the initial value.

    def cond_fn(state):
      done, low, middle, high, _, _ = state
      return jnp.any((middle < high) & (middle > low) & ~done)

    def body_fn(state):
      done, low, middle, high, value_middle, grad_middle = state
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
      # TODO(srvasude): Allow this parameter to be varied.
      new_middle = jnp.where(~done, (low + high) / 2., middle)

      new_value_middle, new_grad_middle = self._value_and_grad_on_line(
          x, new_middle, descent_direction, *args, **kwargs)
      new_value_middle = jnp.where(~done, new_value_middle, value_middle)
      new_grad_middle = jnp.where(~done, new_grad_middle, grad_middle)

      return (done,
              new_low,
              new_middle,
              new_high,
              new_value_middle,
              new_grad_middle)

    _, final_low, _, final_high, _, _ = jax.lax.while_loop(
        cond_fn,
        body_fn,
        ((middle >= high) | (middle <= low),
         low,
         middle,
         high,
         value_middle,
         grad_middle))
    return final_low, final_high

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
    new_low, new_high = self._update(
        x, low, high, c, approx_wolfe_threshold_value,
        descent_direction, *args, **kwargs)
    on_left_boundary = jnp.equal(c, new_low)
    on_right_boundary = jnp.equal(c, new_high)
    c = jnp.where(on_right_boundary, self._secant(
        x, high, new_high, descent_direction, *args, **kwargs), c)
    c = jnp.where(on_left_boundary, self._secant(
        x, low, new_low, descent_direction, *args, **kwargs), c)

    def _reupdate():
      return self._update(
          x, new_low, new_high, c, approx_wolfe_threshold_value,
          descent_direction, *args, **kwargs)

    new_low, new_high = jax.lax.cond(
        on_left_boundary | on_right_boundary,
        _reupdate, lambda: (new_low, new_high))
    return new_low, new_high

  def _bracket(
      self, x, c, approx_wolfe_threshold_value,
      descent_direction, *args, **kwargs):
    # Initial interval that satisfies the opposite slope condition.

    def cond_fn(state):
      return jnp.any(~state[0])

    def body_fn(state):
      (done,
       low,
       middle,
       high,
       value_middle,
       grad_middle,
       best_middle) = state
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
            descent_direction, *args, **kwargs)

      new_low, new_high = jax.lax.cond(
          reupdate, _update_interval, lambda: (new_low, new_high))
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

      return (done,
              new_low,
              new_middle,
              new_high,
              new_value_middle,
              new_grad_middle,
              best_middle)

    value_c, grad_c = self._value_and_grad_on_line(
        x, c, descent_direction, *args, **kwargs)

    _, final_low, _, final_high, _, _, _ = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (jnp.array(False),
         jnp.array(0.),
         c,
         c,
         value_c,
         grad_c,
         jnp.array(0.)))
    return final_low, final_high

  def init_state(self,  # pylint:disable=keyword-arg-before-vararg
                 init_stepsize: float,
                 params: Any,
                 value: Optional[float] = None,
                 grad: Optional[Any] = None,
                 descent_direction: Optional[Any] = None,
                 *args,
                 **kwargs) -> HagerZhangLineSearchState:
    """Initialize the line search state.

    Args:
      init_stepsize: initial step size value. This is ignored by the linesearch.
      params: current parameters.
      value: current function value (recomputed if None).
      grad: current gradient (recomputed if None).
      descent_direction: ignored.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      state
    """
    del init_stepsize

    if value is None or grad is None:
      if self.has_aux:
        (value, _), grad = self._value_and_grad_fun(params, *args, **kwargs)
      else:
        value, grad = self._value_and_grad_fun(params, *args, **kwargs)


    if descent_direction is None:
      descent_direction = tree_scalar_mul(-1, grad)

    approx_wolfe_threshold_value = (
        value + self.approximate_wolfe_threshold * jnp.abs(value))

    # Create initial interval.
    low, high = self._bracket(
        params, jnp.ones_like(value),
        approx_wolfe_threshold_value, descent_direction, *args, **kwargs)

    value_low, grad_low = self._value_and_grad_on_line(
        params, low, descent_direction, *args, **kwargs)

    value_high, grad_high = self._value_and_grad_on_line(
        params, high, descent_direction, *args, **kwargs)

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
        grad=grad)

  def update(self,  # pylint:disable=keyword-arg-before-vararg
             stepsize: float,
             state: NamedTuple,
             params: Any,
             value: Optional[float] = None,
             grad: Optional[Any] = None,
             descent_direction: Optional[Any] = None,
             *args,
             **kwargs) -> base.LineSearchStep:
    """Performs one iteration of Hager-Zhang line search.

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
      if self.has_aux:
        (value, _), grad = self._value_and_grad_fun(params, *args, **kwargs)
      else:
        value, grad = self._value_and_grad_fun(params, *args, **kwargs)


    if descent_direction is None:
      descent_direction = tree_scalar_mul(-1, grad)

    approx_wolfe_threshold_value = (
        value + self.approximate_wolfe_threshold * jnp.abs(value))

    new_low, new_high = self._secant2(
        params,
        state.low,
        state.high,
        approx_wolfe_threshold_value,
        descent_direction,
        *args, **kwargs)

    new_low = jnp.where(state.done, state.low, new_low)
    new_high = jnp.where(state.done, state.high, new_high)

    def _reupdate():
      c = (new_low + new_high) / 2.
      return self._update(
          params, new_low, new_high, c, approx_wolfe_threshold_value,
          descent_direction, *args, **kwargs)

    new_low, new_high = jax.lax.cond(
        ~state.done & ((new_high - new_low) >
                       (self.shrinkage_factor * (state.high - state.low))),
        _reupdate, lambda: (new_low, new_high))

    # Check wolfe and approximate wolfe conditions and update them.

    value_low, grad_low = self._value_and_grad_on_line(
        params, new_low, descent_direction, *args, **kwargs)
    value_high, grad_high = self._value_and_grad_on_line(
        params, new_high, descent_direction, *args, **kwargs)

    best_point = jnp.where(value_low < value_high, new_low, new_high)
    gd_vdot_best_point = jnp.where(value_low < value_high, grad_low, grad_high)

    value_best_point = jnp.minimum(value_low, value_high)

    new_stepsize = jnp.where(state.done, stepsize, best_point)
    new_params = tree_add_scalar_mul(params, best_point, descent_direction)
    if self.has_aux:
      (new_value, new_aux), new_grad = self._value_and_grad_fun(
          new_params, *args, **kwargs)
    else:
      new_value, new_grad = self._value_and_grad_fun(
          new_params, *args, **kwargs)
      new_aux = None

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

    new_state = HagerZhangLineSearchState(
        iter_num=state.iter_num + 1,
        value=new_value,
        grad=new_grad,
        aux=new_aux,
        params=new_params,
        low=new_low,
        high=new_high,
        error=error,
        done=done)

    return base.LineSearchStep(stepsize=new_stepsize, state=new_state)

  def __post_init__(self):
    if self.value_and_grad:
      self._value_and_grad_fun = self.fun
    else:
      self._value_and_grad_fun = jax.value_and_grad(self.fun, has_aux=self.has_aux)
