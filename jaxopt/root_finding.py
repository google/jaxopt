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

from typing import Callable

import jax.numpy as jnp

from jaxopt import implicit_diff
from jaxopt import linear_solve
from jaxopt import loop


def make_bisect_fun(fun: Callable,
                    lower: float,
                    upper: float,
                    increasing: bool = True,
                    maxiter: int = 30,
                    tol: float = 1e-5):
  """Makes a bisection solver.

  Args:
    fun: an equation function, ``fun(x, params)`` where ``x`` is 1d.
      The function should be increasing w.r.t. ``x`` on ``[lower, upper]``
        if ``increasing=True`` and decreasing otherwise.
    lower: the lower end of the bracketing interval.
    upper: the upper end of the bracketing interval.
    increasing: whether ``fun(x, params)`` is an increasing function of ``x``.
    maxiter: maximum number of iterations.
    tol: tolerance.
  Returns:
    A solver function ``bisect(params)``.
  """
  fun_ = fun if increasing else lambda x, p: -fun(x, p)

  def bisect(params):
    def cond_fun(args):
      _, _, error = args
      return error > tol

    def body_fun(args):
      low, high, _ = args
      midpoint = 0.5 * (high + low)
      value = fun_(midpoint, params)
      error = jnp.abs(value)
      too_large = value < 0
      # When `value` is too large, `midpoint` becomes the next `high`,
      # and `low` remains the same. Otherwise, it is the opposite.
      high = jnp.where(too_large, midpoint, high)
      low = jnp.where(too_large, low, midpoint)
      return low, high, error

    args = (lower, upper, jnp.inf)
    low, high, _ = loop.while_loop(cond_fun=cond_fun, body_fun=body_fun,
                                   init_val=args, maxiter=maxiter,
                                   unroll=False, jit=True)
    return 0.5 * (high + low)

  # Add implicit differentiation to the bisection solver.
  return implicit_diff.custom_root(fun, solve=linear_solve.solve_lu)(bisect)
