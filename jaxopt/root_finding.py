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

"""Root finding algorithms in JAX."""

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional

from dataclasses import dataclass

import jax.numpy as jnp

from jaxopt import base
from jaxopt import implicit_diff3 as idf
from jaxopt import linear_solve
from jaxopt import loop


class BisectionState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  value: float
  error: float
  low: float
  high: float


@dataclass
class Bisection:
  """One-dimensional root finding using bisection.

  Attributes:
    optimality_fun: a function ``optimality_fun(x, *args, **kwargs)``
      where ``x`` is a 1d variable. The function should be increasing w.r.t.
      ``x`` on ``[lower, upper]`` if ``increasing=True`` and decreasing
      otherwise.
    lower: the lower end of the bracketing interval.
    upper: the upper end of the bracketing interval.
    increasing: whether ``optimality_fun(x, *args, **kwargs)`` is an
      increasing function of ``x``. Default: True.
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
  increasing: bool = True
  maxiter: int = 30
  tol: float = 1e-5
  check_bracket: bool = True
  implicit_diff: bool = True
  verbose: bool = False

  def _check_bracket(self, *args, **kwargs):
    if self._fun(self.lower, *args, **kwargs) > 0:
      raise ValueError("optimality_fun(lower, *args, **kwargs) should be < 0 "
                       "if increasing=True and > 0 if increasing=False.")

    if self._fun(self.upper, *args, **kwargs) < 0:
      raise ValueError("optimality_fun(upper, *args, **kwargs) should be > 0 "
                       "if increasing=True and < 0 if increasing=False.")

  def init(self,
           init_params,
           *args,
           **kwargs) -> base.OptStep:
    """Initialize the ``(params, state)`` pair.

    Args:
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    if self.check_bracket:
      self._check_bracket(*args, **kwargs)

    state = BisectionState(iter_num=0,
                           value=jnp.inf,
                           error=jnp.inf,
                           low=self.lower,
                           high=self.upper)

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
    value = self._fun(params, *args, **kwargs)
    too_large = value > 0
    # When `value` is too large, `params` becomes the next `high`,
    # and `low` remains the same. Otherwise, it is the opposite.
    high = jnp.where(too_large, params, state.high)
    low = jnp.where(too_large, state.low, params)

    state = BisectionState(iter_num=state.iter_num + 1,
                           value=value,
                           error=jnp.sqrt(value ** 2),
                           low=low,
                           high=high)

    # We return `midpoint` as the next guess.
    # Users can also inspect state.low and state.high.
    midpoint = 0.5 * (low + high)
    return base.OptStep(params=midpoint, state=state)

  def run(self,
          init_params,
          *args,
          **kwargs) -> base.OptStep:
    """Runs the bisection algorithm.

    Args:
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
    # We prepare the function below so that we don't have to worry about
    # the sign during the algorithm.
    if self.increasing:
      self._fun = self.optimality_fun
    else:
      self._fun = lambda x, *a, **kw: -self.optimality_fun(x, *a, **kw)

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
