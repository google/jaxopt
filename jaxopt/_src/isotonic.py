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

"""Isotonic Regression."""

import warnings
import numpy as onp
import jax
import jax.numpy as jnp


# pylint: disable=g-import-not-at-top
try:
  from numba import njit

  NUMBA_AVAILABLE = True
except ImportError:
  NUMBA_AVAILABLE = False
  # If Numba is not available, we define a dummy 'njit' function.

  def njit(func):
    return func


@njit
def _isotonic_l2_pav_numba(y):
  n = y.shape[0]
  target = onp.arange(n)
  c = onp.ones(n)
  sums = onp.zeros(n)
  sol = onp.zeros(n)

  # target describes a list of blocks.  At any time, if [i..j] (inclusive) is
  # an active block, then target[i] := j and target[j] := i.

  for i in range(n):
    sol[i] = y[i]
    sums[i] = y[i]

  i = 0
  while i < n:
    k = target[i] + 1
    if k == n:
      break
    if sol[i] > sol[k]:
      i = k
      continue
    sum_y = sums[i]
    sum_c = c[i]
    while True:
      # We are within an increasing subsequence.
      prev_y = sol[k]
      sum_y += sums[k]
      sum_c += c[k]
      k = target[k] + 1
      if k == n or prev_y > sol[k]:
        # Non-singleton increasing subsequence is finished,
        # update first entry.
        sol[i] = sum_y / sum_c
        sums[i] = sum_y
        c[i] = sum_c
        target[i] = k - 1
        target[k - 1] = i
        if i > 0:
          # Backtrack if we can.  This makes the algorithm
          # single-pass and ensures O(n) complexity.
          i = target[i - 1]
        # Otherwise, restart from the same point.
        break

  # Reconstruct the solution.
  i = 0
  while i < n:
    k = target[i] + 1
    sol[i + 1 : k] = sol[i]
    i = k
  return sol.astype(y.dtype)


@jax.custom_jvp
def _isotonic_l2_pav(y):
  if not NUMBA_AVAILABLE:
    warnings.warn(
        "Numba could not be imported. Code will run much more slowly."
        " To install, run 'pip install numba'."
    )
  # Define the expected shape & dtype of output.
  shape_dtype = jax.ShapeDtypeStruct(shape=y.shape, dtype=y.dtype)
  sol = jax.pure_callback(_isotonic_l2_pav_numba, shape_dtype, y)
  return sol


def isotonic_l2_pav(y, y_min=-jnp.inf, y_max=jnp.inf, increasing=True):
  r"""Solves an isotonic regression problem using PAV.

  Args:
    y: input to isotonic regression, a 1d-array.

    y_min : Lower bound on the lowest predicted value.
    y_max : Upper bound on the highest predicted value

    increasing : Order of the constraints:
        If True, it solves :math:`\mathop{\mathrm{arg\,min}}_{v_1 \leq ... \leq v_n} \|v - y\|^2`.
        If False, it solves :math:`\mathop{\mathrm{arg\,min}}_{v_1 \geq ... \geq v_n} \|v - y\|^2`.

  Returns:
    The solution, an array of the same size as y.
  """
  sign = -1 if increasing else 1
  sol = _isotonic_l2_pav(y * sign) * sign
  sol = jnp.clip(sol, y_min, y_max)
  return sol


def _jvp_isotonic_l2_jax_pav(solution, vector, eps=1e-8):
  x = solution
  mask = jnp.pad(jnp.absolute(jnp.diff(x)) <= eps, (1, 0))
  ar = jnp.arange(x.size)
  inds_start = jnp.where(mask == 0, ar, +jnp.inf).sort()
  one_hot_start = jax.nn.one_hot(inds_start, len(vector))
  A = jnp.cumsum(one_hot_start, axis=-1)
  A = jnp.append(jnp.diff(A[::-1], axis=0)[::-1], A[-1].reshape(1, -1), axis=0)
  B = A.copy()
  return (((B.T * (B @ vector)).T) / (A.sum(1, keepdims=True) + 1e-8)).sum(0)


@_isotonic_l2_pav.defjvp
def _isotonic_l2_pav_jvp(primals, tangents):
  """Jacobian-vector product of isotonic_l2_pav.

  See Section 5 of
  Fast Differentiable Sorting and Ranking
  Mathieu Blondel, Olivier Teboul, Quentin Berthet, Josip Djolonga
  ICML 2020 arXiv:2002.08871
  """
  (y, ) = primals
  (vector, ) = tangents
  primal_out = _isotonic_l2_pav(y)
  tangent_out = _jvp_isotonic_l2_jax_pav(primal_out, vector)
  return primal_out, tangent_out
