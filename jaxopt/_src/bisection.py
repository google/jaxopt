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

"""Bisection algorithm in JAX."""

from typing import Any
from typing import Callable
from typing import NamedTuple

from dataclasses import dataclass

import jax.numpy as jnp

from jaxopt._src import base
from jaxopt._src import implicit_diff as idf
from jaxopt._src import linear_solve
from jaxopt._src import loop


class BisectionState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  value: float
  error: float
  low: float
  high: float
  sign: int


@dataclass
class Bisection:
  """One-dimensional root finding using bisection.

  Attributes:
    optimality_fun: a function ``optimality_fun(x, *args, **kwargs)``
      where ``x`` is a 1d variable. The function should have opposite signs
      when evaluated at ``lower`` and at ``upper``.
    lower: the lower end of the bracketing interval.
    upper: the upper end of the bracketing interval.
    maxiter: maximum number of iterations.
    tol: tolerance.
    check_bracket: whether to check correctness of the bracketing interval.
      If True, the method ``run`` cannot be jitted.
    implicit_diff: if True, enable implicit differentiation.
    verbose: whether to print error on every iteration or not.
      Warning: verbose=True will automatically disable jit.

  """
  optimality_fun: Callable
  lower: float
  upper: float
  maxiter: int = 30
  tol: float = 1e-5
  check_bracket: bool = True
  implicit_diff: bool = True
  verbose: bool = False

  def init(self,
           init_params=None,
           *args,
           **kwargs) -> base.OptStep:
    """Initialize the ``(params, state)`` pair.

    Args:
      init_params: initial scalar value. If None (default), we use
        0.5 * (self.high + self.low) instead. This initialization is mainly for
        API consistency with the rest of JAXopt and will not affect the next
        bracketed interval.
      *args: additional positional arguments to be passed to ``optimality_fun``.
      **kwargs: additional keyword arguments to be passed to ``optimality_fun``.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    lower_value = self.optimality_fun(self.lower, *args, **kwargs)
    upper_value = self.optimality_fun(self.upper, *args, **kwargs)

    # sign = 1: the function is increasing
    # sign = -1: the function is decreasing
    # sign = 0: the root is not contained in [lower, upper]
    sign = jnp.where((lower_value < 0) & (upper_value >= 0),
                     1,
                     jnp.where((lower_value > 0) & (upper_value <= 0), -1, 0))

    if self.check_bracket:
      # Not jittable...
      if sign == 0:
        raise ValueError("The root is not contained in [lower, upper]. "
                         "`optimality_fun` evaluated at lower and upper should "
                         "have opposite signs.")

    state = BisectionState(iter_num=0,
                           value=jnp.inf,
                           error=jnp.inf,
                           low=self.lower,
                           high=self.upper,
                           sign=sign)

    if init_params is None:
      init_params = 0.5 * (state.high + state.low)

    return base.OptStep(params=init_params, state=state)

  def update(self,
             params: Any,
             state: NamedTuple,
             *args,
             **kwargs) -> base.OptStep:
    """Performs one iteration of the bisection solver.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    value = self.optimality_fun(params, *args, **kwargs)
    too_large = state.sign * value > 0

    # When `value` is too large, `params` becomes the next `high`,
    # and `low` remains the same. Otherwise, it is the opposite.
    high = jnp.where(too_large, params, state.high)
    low = jnp.where(too_large, state.low, params)

    state = BisectionState(iter_num=state.iter_num + 1,
                           value=value,
                           error=jnp.sqrt(value ** 2),
                           low=low,
                           high=high,
                           sign=state.sign)

    # We return `midpoint` as the next guess.
    # Users can also inspect state.low and state.high.
    midpoint = 0.5 * (low + high)
    return base.OptStep(params=midpoint, state=state)

  def run(self,
          init_params=None,
          *args,
          **kwargs) -> base.OptStep:
    """Runs the bisection algorithm.

    Args:
      init_params: initial scalar value. If None (default), we use
        0.5 * (self.high + self.low) instead. This initialization is mainly for
        API consistency with the rest of JAXopt and will not affect the next
        bracketed interval.
      *args: additional positional arguments to be passed to ``optimality_fun``.
      **kwargs: additional keyword arguments to be passed to ``optimality_fun``.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    def cond_fun(pair):
      _, state = pair
      if self.verbose:
        print(state.error)
      return state.error > self.tol

    def body_fun(pair):
      params, state = pair
      return self.update(params, state, *args, **kwargs)

    return loop.while_loop(cond_fun=cond_fun, body_fun=body_fun,
                           init_val=self.init(init_params, *args, **kwargs),
                           maxiter=self.maxiter, jit=self._jit,
                           unroll=self._unroll)

  def l2_optimality_error(self, params, *args, **kwargs):
    """Computes the L2 optimality error."""
    return jnp.sqrt(self.optimality_fun(params, *args, **kwargs) ** 2)

  def __post_init__(self):
    # Make sure integers are converted to floats.
    self.lower = jnp.array(self.lower, float)
    self.upper = jnp.array(self.upper, float)

    # We always jit unless verbose mode is enabled.
    self._jit = not self.verbose
    # We unroll when implicit diff is disabled or when jit is disabled.
    self._unroll = not self.implicit_diff or not self._jit

    if self.implicit_diff:
      decorator = idf.custom_root(self.optimality_fun,
                                  has_aux=True,
                                  solve=linear_solve.solve_lu)
      # pylint: disable=g-missing-from-attributes
      self.run = decorator(self.run)
