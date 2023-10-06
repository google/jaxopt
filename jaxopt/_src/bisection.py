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
from typing import Optional
from typing import Union

from dataclasses import dataclass

import jax.numpy as jnp

from jaxopt._src import base
from jaxopt._src import implicit_diff as idf
from jaxopt._src import loop


class BisectionState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  value: float
  error: float
  low: float
  high: float
  sign: int
  aux: Optional[Any] = None

  num_fun_eval: int = 0


@dataclass(eq=False)
class Bisection(base.IterativeSolver):
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
    implicit_diff_solve: the linear system solver to use.
    verbose: whether to print information on every iteration or not.
    jit: whether to JIT-compile the bisection loop (default: True).
    unroll: whether to unroll the bisection loop (default: "auto").

  """
  optimality_fun: Callable
  lower: float
  upper: float
  maxiter: int = 30
  tol: float = 1e-5
  check_bracket: bool = True
  verbose: Union[bool, int] = False
  implicit_diff_solve: Optional[Callable] = None
  has_aux: bool = False
  jit: bool = True
  unroll: base.AutoOrBoolean = "auto"

  def init_state(self,
                 init_params: Optional[Any] = None,
                 *args,
                 **kwargs) -> BisectionState:
    """Initialize the solver state.

    Args:
      init_params: ignored, we use 0.5 * (state.high + state.low) instead.
      *args: additional positional arguments to be passed to ``optimality_fun``.
      **kwargs: additional keyword arguments to be passed to ``optimality_fun``.
    Returns:
      state
    """
    del init_params

    lower = jnp.asarray(self.lower, float)
    upper = jnp.asarray(self.upper, float)

    lower_value = self.optimality_fun(lower, *args, **kwargs)
    upper_value = self.optimality_fun(upper, *args, **kwargs)

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

    return BisectionState(iter_num=jnp.asarray(0),
                          value=jnp.asarray(jnp.inf),
                          error=jnp.asarray(jnp.inf),
                          low=lower, high=upper,
                          sign=jnp.asarray(sign),
                          num_fun_eval=jnp.array(2, base.NUM_EVAL_DTYPE))

  def update(self,
             params,
             state: NamedTuple,
             *args,
             **kwargs) -> base.OptStep:
    """Performs one iteration of the bisection solver.

    Args:
      params: ignored, we use 0.5 * (state.high + state.low) instead.
      state: named tuple containing the solver state.
    Returns:
      (params, state)
    """
    # Ignored: we use the midpoint between state.low and state.high instead.
    del params

    params = 0.5 * (state.high + state.low)
    value, aux = self._fun_with_aux(params, *args, **kwargs)
    error = jnp.sqrt(value ** 2)

    # Compute `low` and `high` for the next iteration.
    # When `value` is too large, `params` becomes the next `high`,
    # and `low` remains the same. Otherwise, it is the opposite.
    too_large = state.sign * value > 0
    high = jnp.where(too_large, params, state.high)
    low = jnp.where(too_large, state.low, params)

    state = BisectionState(iter_num=state.iter_num + 1,
                           value=value,
                           error=error,
                           low=low,
                           high=high,
                           sign=state.sign,
                           aux=aux,
                           num_fun_eval=state.num_fun_eval + 1)

    if self.verbose:
      self.log_info(
          state,
          error_name="Absolute Value Output", 
          additional_info={
              "High Point": high, 
              "Low Point": low
          }
      )
    return base.OptStep(params=params, state=state)

  def run(self,
          init_params: Optional[Any] = None,
          *args,
          **kwargs) -> base.OptStep:
    # We override run in order to set init_params=None by default.
    return super().run(init_params, *args, **kwargs)

  def __post_init__(self):
    super().__post_init__()

    if self.has_aux:
      self._fun_with_aux = self.optimality_fun
    else:
      self._fun_with_aux = lambda *a, **kw: (self.optimality_fun(*a, **kw),
                                             None)
