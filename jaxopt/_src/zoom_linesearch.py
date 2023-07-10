# Copyright 2023 Google LLC
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

"""Zoom line search algorithm."""

import dataclasses
import functools
from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional

import jax
from jax import lax
import jax.numpy as jnp
from jaxopt._src import base
from jaxopt._src.base import _make_funs_with_aux
from jaxopt._src.cond import cond
from jaxopt._src.tree_util import tree_single_dtype
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_vdot
# pylint: disable=g-bare-generic
# pylint: disable=invalid-name

_dot = functools.partial(jnp.dot, precision=lax.Precision.HIGHEST)


def _cubicmin(a, fa, fpa, b, fb, c, fc):
  """Cubic interpolation.

  Finds a critical point of a cubic polynomial
  p(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D, that goes through
  the points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.
  May return NaN (if radical<0), in that case, the point will be ignored.
  Taken from scipy.optimize._linesearch.py.

  Args:
    a: scalar
    fa: value of a function f at a
    fpa: slope of a function f at a
    b: scalar
    fb: value of a function f at b
    c: scalar
    fc: value of a function f at c

  Returns:
    xmin: point at which p'(xmin) = 0
  """
  C = fpa
  db = b - a
  dc = c - a
  denom = (db * dc) ** 2 * (db - dc)
  d1 = jnp.array([[dc**2, -(db**2)], [-(dc**3), db**3]])
  A, B = _dot(d1, jnp.array([fb - fa - C * db, fc - fa - C * dc])) / denom

  radical = B * B - 3.0 * A * C
  xmin = a + (-B + jnp.sqrt(radical)) / (3.0 * A)

  return xmin


def _quadmin(a, fa, fpa, b, fb):
  """Quadratic interpolation.

  Finds a critical point of a quadratic polynomial
  p(x) = B*(x-a)^2 + C*(x-a) + D, that goes through
  the points (a,fa), (b,fb) with derivative at a of fpa.
  Taken from scipy.optimize._linesearch.py.

  Args:
    a: scalar
    fa: value of a function f at a
    fpa: slope of a function f at a
    b: scalar
    fb: value of a function f at b

  Returns:
    xmin: point at which p'(xmin) = 0
  """
  D = fa
  C = fpa
  db = b - a
  B = (fb - D - C * db) / (db**2)
  xmin = a - C / (2.0 * B)
  return xmin


def _set_values(cond, candidate, default):
  def _set_val(x, y):
    return jnp.where(cond, x, y)

  return jax.tree_util.tree_map(_set_val, candidate, default)


def _check_failure_status(fail_code):
  """Print failure reason according to fail value."""
  if fail_code == 1:
    print("Provided descent direction is not a descent direction.")
  elif fail_code == 2:
    print("Maximal stepsize reached.")
  elif fail_code == 3:
    print("Maximal number of line search iterations reached.")
  elif fail_code == 4:
    print(
        "Length of searched interval has been reduced below machine precision."
    )
  elif fail_code == 5:
    print("NaN or Inf values encountered in function values.")


class ZoomLineSearchState(NamedTuple):
  """Named tuple containing state information for core loop."""

  iter_num: int
  params: Any  # either initial or final
  value: float  # either initial or final
  grad: Any  # either initial or final

  # unchanged after initialization
  value_init: float  #  (redundant with value, left for readability)
  slope_init: float
  descent_direction: Any

  num_fun_eval: int
  num_grad_eval: int

  error: float
  done: bool
  fail_code: int  # encode failure status, see _check_status
  failed: bool  # comply to semantic used by other line searches

  # Used only during the interval search
  interval_found: bool
  prev_stepsize: float
  prev_value_step: float
  prev_slope_step: float

  # Set up after interval search done, modified during zoom
  low: float
  value_low: float
  slope_low: float
  high: float
  value_high: float
  slope_high: float
  cubic_ref: float
  value_cubic_ref: float

  # Safeguard point: we may not be able to satisfy the curvature condition
  # but we can still return a point that satisfies the decrease condition
  safe_stepsize: float

  aux: Optional[Any] = None  # either initial or final


@dataclasses.dataclass(eq=False)
class ZoomLineSearch(base.IterativeLineSearch):
  """Inexact line search that satisfies sufficient decrease (Armijo) and small curvature (strong Wolfe) conditions.

  Algorithms 3.5, 3.6 from [1], pages 60-62.

  The sufficient decrease condition may be impossible to satisfy close to a
  minimum, in that case, we switch to an approximate sufficient decrease
  condition (approximate Wolfe) taken from [2].

  [1] J. Nocedal and S. Wright, 'Numerical Optimization', 2nd edition, 2006.
  [2] W. Hager, H. Zhang, Algorithm 851: CG_DESCENT, a Conjugate Gradient Method
    with Guaranteed Descent.

  Attributes:
    fun: a function of the form ``fun(params, *args, **kwargs)``, where
      ``params`` are parameters of the model, ``*args`` and ``**kwargs`` are
      additional arguments.
    value_and_grad: if ``False``, ``fun`` should return the function value only.
      If ``True``, ``fun`` should return both the function value and the
      gradient.
    has_aux: if ``False``, ``fun`` should return the function value only. If
      ``True``, ``fun`` should return a pair ``(value, aux)`` where ``aux`` is a
      pytree of auxiliary values. (default: False)
    c1: constant used to check if a sufficient decrease has been found (Armijo)
      (default: 1e-4)
    c2: constant used to check if a small curvature has been found (strong
      Wolfe) (default: 0.9)
    c3: constant used to check if an approximate sufficient decrease
      (approximate Wolfe) has been found (default: 1e-6)
    rel_tol_cubic: point computed by cubic interpolation accepted if inside
      rel_tol_cubic*interval_size (default: 0.2)
    rel_tol_quad: point computed by quadratic interpolation accepted if inside
      rel_tol_quad*interval_size (default: 0.1)
    increase_factor: factor to mutliply stepsize at initialization until finding
      interval satisfying curvature condition (default: 2.)
    max_stepsize: maximal possible stepsize. (default: 2**30)
    tol: tolerance of the stopping criterion. (default: 0.0)
    maxiter: maximum number of line search iterations. (default: 30)
    verbose: whether to print error on every iteration or not. verbose=True will
      automatically disable jit. (default: False)
    jit: whether to JIT-compile the optimization loop (default: "auto").
    unroll: whether to unroll the optimization loop (default: "auto").
  """

  fun: Callable
  value_and_grad: bool = False
  has_aux: bool = False

  c1: float = 1e-4
  c2: float = 0.9
  c3: float = 1e-6
  rel_tol_cubic: float = 0.2
  rel_tol_quad: float = 0.1
  increase_factor: float = 2.0

  tol: float = 0.0
  maxiter: int = 30
  # max_stepsize needs to be large enough for the linesearch to be able
  # to find a good stepsize
  max_stepsize: float = 2**30

  verbose: bool = False
  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"

  def _value_and_slope_on_line(
      self, params, stepsize, descent_direction, args, kwargs
  ):
    step = tree_add_scalar_mul(params, stepsize, descent_direction)
    (value_step, aux_step), grad_step = self._value_and_grad_fun_with_aux(
        step, *args, **kwargs
    )
    slope_step = tree_vdot(grad_step, descent_direction)
    return value_step, slope_step, step, grad_step, aux_step

  def _decrease_error(
      self, stepsize, value_step, slope_step, value_init, slope_init
  ):
    # We consider either the usual sufficient decrease (Armijo condition), see
    # equation (3.7a) of [1]
    exact_decrease_error = (
        value_step - value_init - self.c1 * stepsize * slope_init
    )
    # or an approximate decrease condition, see equation (23) of [2]
    approx_decrease_error_ = slope_step - (2 * self.c1 - 1.0) * slope_init

    # The classical Armijo condition may fail to be satisfied if we are too
    # close to a minimum, causing the optimizer to fail as explained in [2]

    # We switch to approximate Wolfe conditions only if we are close enough to
    # the minimizer which is captured by the following criterion.
    delta_values = value_step - value_init - self.c3 * jnp.abs(value_init)
    approx_decrease_error = jnp.maximum(approx_decrease_error_, delta_values)
    # We take then the *minimum* of both errors.
    return jnp.minimum(approx_decrease_error, exact_decrease_error)

  def _curvature_error(self, slope_step, slope_init):
    # See equation (3.7b) of [1].
    return jnp.abs(slope_step) - self.c2 * jnp.abs(slope_init)

  def _make_safe_step(self, _, state, args, kwargs):
    safe_stepsize = state.safe_stepsize
    step = tree_add_scalar_mul(
        state.params, safe_stepsize, state.descent_direction
    )
    (value_step, aux_step), grad_step = self._value_and_grad_fun_with_aux(
        step, *args, **kwargs
    )
    new_state = state._replace(
        params=step, value=value_step, grad=grad_step, aux=aux_step
    )
    return safe_stepsize, new_state

  def _keep_step(self, stepsize, state, _, __):
    return stepsize, state

  def _search_interval(self, init_stepsize, state, args, kwargs):
    """Line search procedure described in Algorithm 3.5 of [1]."""
    # init_stepsize only used for iter_num = 0

    iter_num = state.iter_num

    params_init = state.params
    grad_init = state.grad
    aux_init = state.aux

    fail_code = state.fail_code

    value_init = state.value_init
    slope_init = state.slope_init
    descent_direction = state.descent_direction

    prev_stepsize = state.prev_stepsize
    prev_value_step = state.prev_value_step
    prev_slope_step = state.prev_slope_step

    safe_stepsize = state.safe_stepsize

    # Choose new point, larger than previous one or set to initial guess
    # for first iteration.
    larger_stepsize = self.increase_factor * prev_stepsize
    new_stepsize_ = jnp.where(iter_num == 0, init_stepsize, larger_stepsize)
    new_stepsize = jnp.minimum(new_stepsize_, self.max_stepsize)

    max_stepsize_reached = new_stepsize >= self.max_stepsize
    fail_check1 = jnp.where(
        (fail_code == 0) & max_stepsize_reached, 2, fail_code
    )

    new_value_step, new_slope_step, new_step, new_grad_step, new_aux_step = (
        self._value_and_slope_on_line(
            params_init, new_stepsize, descent_direction, args, kwargs
        )
    )
    is_value_nan = jnp.isnan(new_value_step) | jnp.isinf(new_value_step)
    fail_check2 = jnp.where((fail_check1 == 0) & is_value_nan, 5, fail_check1)

    decrease_error_ = self._decrease_error(
        new_stepsize, new_value_step, new_slope_step, value_init, slope_init
    )
    decrease_error = jnp.maximum(decrease_error_, 0.0)
    curvature_error_ = self._curvature_error(new_slope_step, slope_init)
    curvature_error = jnp.maximum(curvature_error_, 0.0)
    new_error = jnp.maximum(decrease_error, curvature_error)

    # If the new point satisfies at least the decrease error we keep it
    # in case the curvature error cannot be satisfied.
    safe_decrease = decrease_error <= self.tol
    new_safe_stepsize = jnp.where(safe_decrease, new_stepsize, safe_stepsize)

    # If the new point not good, set high and low values according to
    # conditions described in Algorithm 3.5 of [1]
    set_high_to_new = (decrease_error > 0.0) | (
        (new_value_step >= prev_value_step) & (iter_num > 0)
    ) | is_value_nan
    set_low_to_new = (new_slope_step >= 0.0) & (~set_high_to_new)

    # By default we set high to new and correct if we should have set
    # low to new. If none should have set, the search for the interval
    # continues anyway.
    low_, value_low_, slope_low_, high_, value_high_, slope_high_ = (
        prev_stepsize,
        prev_value_step,
        prev_slope_step,
        new_stepsize,
        new_value_step,
        new_slope_step,
    )

    default = [low_, value_low_, slope_low_, high_, value_high_, slope_high_]
    candidate = [
        new_stepsize,
        new_value_step,
        new_slope_step,
        prev_stepsize,
        prev_value_step,
        prev_slope_step,
    ]
    [low, value_low, slope_low, high, value_high, slope_high] = _set_values(
        set_low_to_new, candidate, default
    )

    # If high or low have been set or the point is good, the interval has been
    # found. Otherwise we'll keep on augmenting the stepsize.
    interval_found = set_high_to_new | set_low_to_new | (new_error <= self.tol) | is_value_nan

    # If new_error <= self.tol, the line search is done. In that case, we set
    # directly the new parameters, gradient, value and aux to the ones found.
    default = [0.0, params_init, value_init, grad_init, aux_init]
    candidate = [
        new_stepsize,
        new_step,
        new_value_step,
        new_grad_step,
        new_aux_step,
    ]
    best_stepsize, next_params, next_value, next_grad, next_aux = _set_values(
        new_error <= self.tol, candidate, default
    )

    done = new_error <= self.tol
    max_iter_reached = (iter_num + 1 >= self.maxiter) & (~done)
    new_fail_code = jnp.where(
        (fail_check2 == 0) & max_iter_reached, 3, fail_check2
    )

    new_state = state._replace(
        iter_num=iter_num + 1,
        params=next_params,
        value=next_value,
        grad=next_grad,
        aux=next_aux,
        #
        # If error is nan, the linesearch would stop while one may try a smaller stepsize
        error=jnp.where(jnp.isnan(new_error), jnp.inf, new_error),
        done=done,
        fail_code=new_fail_code,
        failed=jnp.asarray(new_fail_code > 0),
        interval_found=interval_found,
        #
        prev_stepsize=new_stepsize,
        prev_value_step=new_value_step,
        prev_slope_step=new_slope_step,
        #
        low=low,
        value_low=value_low,
        slope_low=slope_low,
        high=high,
        value_high=value_high,
        slope_high=slope_high,
        cubic_ref=low,
        value_cubic_ref=value_low,
        #
        safe_stepsize=new_safe_stepsize,
        #
        num_fun_eval=state.num_fun_eval + 1,
        num_grad_eval=state.num_grad_eval + 1,
    )
    return base.LineSearchStep(stepsize=best_stepsize, state=new_state)

  def _zoom_into_interval(self, stepsize, state, args, kwargs):
    """Zoom procedure described in Algorithm 3.6 of [1]."""

    # The stepsize is not used, only low, high, etc... are used to
    # find a good point
    dtype = stepsize.dtype
    del stepsize

    iter_num = state.iter_num

    params_init = state.params
    grad_init = state.grad
    aux_init = state.aux

    value_init = state.value_init
    slope_init = state.slope_init
    descent_direction = state.descent_direction

    fail_code = state.fail_code

    low = state.low
    value_low = state.value_low
    slope_low = state.slope_low
    high = state.high
    value_high = state.value_high
    slope_high = state.slope_high
    cubic_ref = state.cubic_ref
    value_cubic_ref = state.value_cubic_ref

    safe_stepsize = state.safe_stepsize

    # Check if interval not too small otherwise fail
    delta = jnp.abs(high - low)
    left = jnp.minimum(high, low)
    right = jnp.maximum(high, low)
    cubic_chk = self.rel_tol_cubic * delta
    quad_chk = self.rel_tol_quad * delta
    threshold = jnp.where((jnp.finfo(delta).bits < 64), 1e-5, 1e-10)
    too_small_int = delta <= threshold
    fail_check1 = jnp.where((fail_code == 0) & too_small_int, 4, fail_code)

    # Find new point by interpolation
    middle_cubic = _cubicmin(
        low, value_low, slope_low, high, value_high, cubic_ref, value_cubic_ref
    )
    middle_cubic_valid = (middle_cubic > left + cubic_chk) & (
        middle_cubic < right - cubic_chk
    )
    use_cubic = middle_cubic_valid
    middle_quad = _quadmin(low, value_low, slope_low, high, value_high)
    middle_quad_valid = (middle_quad > left + quad_chk) & (
        middle_quad < right - quad_chk
    )
    use_quad = (~use_cubic) & middle_quad_valid
    middle_bisection = (low + high) / 2.0
    use_bisection = (~use_cubic) & (~use_quad)

    middle = jnp.where(use_cubic, middle_cubic, cubic_ref)
    middle = jnp.where(use_quad, middle_quad, middle)
    middle = jnp.where(use_bisection, middle_bisection, middle).astype(dtype)

    # Check if new point is good
    value_middle, slope_middle, step, grad_step, aux_step = (
        self._value_and_slope_on_line(
            params_init, middle, descent_direction, args, kwargs
        )
    )
    is_value_nan = jnp.isnan(value_middle) | jnp.isinf(value_middle)
    fail_check2 = jnp.where((fail_check1 == 0) & is_value_nan, 5, fail_check1)

    decrease_error_ = self._decrease_error(
        middle, value_middle, slope_middle, value_init, slope_init
    )
    decrease_error = jnp.maximum(decrease_error_, 0.0)
    curvature_error_ = self._curvature_error(slope_middle, slope_init)
    curvature_error = jnp.maximum(curvature_error_, 0.0)

    new_error = jnp.maximum(decrease_error, curvature_error)

    # If the new point satisfies at least the decrease error we keep it in case
    # the curvature error cannot be satisfied. We take the largest possible one
    safe_decrease = decrease_error <= self.tol
    new_safe_stepsize_ = jnp.where(safe_decrease, middle, safe_stepsize)
    new_safe_stepsize = jnp.maximum(new_safe_stepsize_, safe_stepsize)

    # If both armijo and curvature conditions are satisfied, we are done.
    done = new_error <= self.tol
    default = [0.0, params_init, value_init, grad_init, aux_init]
    candidate = [middle, step, value_middle, grad_step, aux_step]
    best_stepsize, next_params, next_value, next_grad, next_aux = _set_values(
        new_error <= self.tol, candidate, default
    )

    # Otherwise, we update high and low values
    set_high_to_middle = (decrease_error > 0.0) | (value_middle >= value_low)
    secant_interval = slope_middle * (high - low)
    set_high_to_low = (secant_interval >= 0.0) & (~set_high_to_middle)
    set_low_to_middle = ~set_high_to_middle

    # Set high to middle, or low, or keep as it is
    default = [high, value_high, slope_high]
    candidate = [middle, value_middle, slope_middle]
    [new_high_, new_value_high_, new_slope_high_] = _set_values(
        set_high_to_middle, candidate, default
    )
    default = [new_high_, new_value_high_, new_slope_high_]
    candidate = [low, value_low, slope_low]
    [new_high, new_value_high, new_slope_high] = _set_values(
        set_high_to_low, candidate, default
    )

    # Set low to middle or keep as it is
    default = [low, value_low, slope_low]
    candidate = [middle, value_middle, slope_middle]
    [new_low, new_value_low, new_slope_low] = _set_values(
        set_low_to_middle, candidate, default
    )

    # Update cubic reference point.
    # If high changed then it can be used as the new ref point.
    # Otherwise, low has been updated and not kept as high
    # so it can be used as the new ref point.
    [new_cubic_ref, new_value_cubic_ref] = _set_values(
        set_high_to_middle | set_high_to_low,
        [high, value_high],
        [low, value_low],
    )

    max_iter_reached = (iter_num + 1 >= self.maxiter) & (~done)
    new_fail_code = jnp.where(
        (fail_check2 == 0) & max_iter_reached, 3, fail_check2
    )

    new_state = state._replace(
        iter_num=iter_num + 1,
        params=next_params,
        value=next_value,
        grad=next_grad,
        aux=next_aux,
        #
        error=jnp.where(jnp.isnan(new_error), jnp.inf, new_error),
        done=done,
        fail_code=new_fail_code,
        failed=jnp.asarray(new_fail_code > 0),
        #
        low=new_low,
        value_low=new_value_low,
        slope_low=new_slope_low,
        high=new_high,
        value_high=new_value_high,
        slope_high=new_slope_high,
        cubic_ref=new_cubic_ref,
        value_cubic_ref=new_value_cubic_ref,
        #
        safe_stepsize=new_safe_stepsize,
        #
        num_fun_eval=state.num_fun_eval + 1,
        num_grad_eval=state.num_grad_eval + 1,
    )
    return base.LineSearchStep(stepsize=best_stepsize, state=new_state)

  def init_state(
      self,
      init_stepsize: float,
      params: Any,
      value: Optional[float] = None,
      grad: Optional[Any] = None,
      descent_direction: Optional[Any] = None,
      fun_args: list = [],
      fun_kwargs: dict = {},
  ) -> base.LineSearchStep:
    """Initialize the line search state by computing all relevant quantities and store it in the initial state.

    Args:
      init_stepsize: initial step size value (used in update, not in
        init_state).
      params: current parameters.
      value: current function value (recomputed if None).
      grad: current gradient (recomputed if None).
      descent_direction: descent direction (negative gradient if None).
      fun_args: additional positional arguments to be passed to ``fun``.
      fun_kwargs: additional keyword arguments to be passed to ``fun``.

    Returns:
      state
    """
    # FIXME: Signature issue in base.IterativeLineSearch: Keyword argument
    # before variable positional arguments.
    dtype = tree_single_dtype(params)
    num_fun_eval = jnp.asarray(0)
    num_grad_eval = jnp.asarray(0)
    del init_stepsize
    aux = None
    if value is None or grad is None:
      (value, aux), grad = self._value_and_grad_fun_with_aux(
          params, *fun_args, **fun_kwargs
      )
      num_fun_eval = num_fun_eval + 1
      num_grad_eval = num_grad_eval + 1

    # TODO(vroulet): ideally, we shall also provide aux as arguments to avoid
    # recomputing the function. It's especially problematic if the function
    # provided has an artificial aux = None coming from its instanciation via
    # base._make_funs_with_aux. This requires changing the signature of
    # base.IterativeLineSearch.
    if aux is None and self.has_aux:
      _, aux = self._fun_with_aux(params, *fun_args, **fun_kwargs)

    if descent_direction is None:
      descent_direction = tree_scalar_mul(-1.0, grad)

    slope = tree_vdot(grad, descent_direction)

    fail_code = jnp.where(slope > 0, 1, 0)

    return ZoomLineSearchState(
        iter_num=jnp.asarray(0),
        params=params,
        value=value,
        grad=grad,
        aux=aux,
        #
        value_init=value,
        slope_init=slope,
        descent_direction=descent_direction,
        #
        error=jnp.asarray(jnp.inf),
        done=jnp.asarray(False),
        fail_code=fail_code,
        failed=jnp.asarray(fail_code > 0),
        interval_found=jnp.asarray(False),
        #
        prev_stepsize=jnp.asarray(0.0).astype(dtype),
        prev_value_step=value,
        prev_slope_step=slope,
        #
        low=jnp.asarray(0.0).astype(dtype),
        value_low=value,
        slope_low=slope,
        high=jnp.asarray(0.0).astype(dtype),
        value_high=value,
        slope_high=slope,
        cubic_ref=jnp.asarray(0.0).astype(dtype),
        value_cubic_ref=value,
        #
        safe_stepsize=jnp.asarray(0.0).astype(dtype),
        num_fun_eval=num_fun_eval,
        num_grad_eval=num_grad_eval,
    )

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
    """Combines Algorithms 3.5 and 3.6 of [1].

    Final state contains next_params, next_value, next_grad, next_aux if the
    linesearch succeeded.

    Args:
      stepsize: current estimate of the step size.
      state: named tuple containing the line search state.
      params: current parameters (not used, recorded during init in state).
      value: current function value (not used, recorded during init in state).
      grad: current gradient (not used, recorded during init in state).
      descent_direction: descent direction (not used, recorded during init in
        state).
      fun_args: additional positional arguments to be passed to ``fun``.
      fun_kwargs: additional keyword arguments to be passed to ``fun``.

    Returns:
      (stepsize, state)
    """
    # FIXME: Signature issue in base.IterativeLineSearch: Keyword argument
    # before variable positional arguments.
    # Params, value, grad, descent direction recorded in state at initialization
    dtype = tree_single_dtype(params)
    init_stepsize = jnp.asarray(stepsize).astype(dtype)
    del params
    del value
    del grad
    del descent_direction
    
    jit, _ = self._get_loop_options()

    best_stepsize_, new_state_ = cond(
        state.interval_found, 
        self._zoom_into_interval, 
        self._search_interval, 
        init_stepsize,
        state,
        fun_args,
        fun_kwargs,
        jit=jit
    )

    best_stepsize, new_state = cond(
        (new_state_.failed) & (new_state_.iter_num == self.maxiter), 
        self._make_safe_step,
        self._keep_step,
        best_stepsize_,
        new_state_,
        fun_args,
        fun_kwargs,
        jit=jit
    )

    if self.verbose:
      _check_failure_status(new_state.fail_code)

    return base.LineSearchStep(stepsize=best_stepsize, state=new_state)

  def __post_init__(self):
    self._fun_with_aux, _, self._value_and_grad_fun_with_aux = (
        _make_funs_with_aux(
            self.fun, value_and_grad=self.value_and_grad, has_aux=self.has_aux
        )
    )