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

"""Line searches utilities."""

from jax import numpy as jnp
from jaxopt._src.backtracking_linesearch import BacktrackingLineSearch
from jaxopt._src.hager_zhang_linesearch import HagerZhangLineSearch
from jaxopt._src.zoom_linesearch import ZoomLineSearch


def _setup_linesearch(
    linesearch,
    fun,
    value_and_grad,
    has_aux,
    maxlsiter,
    max_stepsize,
    jit,
    unroll,
    verbose,
    condition,  # for backtracking only
    decrease_factor,  # for backtracking only
    increase_factor,  # for zoom only
):
  """Instantiate linesearch."""
  
  available_linesearches = ["backtracking", "zoom", "hager-zhang"]
  if linesearch == "backtracking":
    linesearch_solver = BacktrackingLineSearch(
        fun=fun,
        value_and_grad=value_and_grad,
        has_aux=has_aux,
        maxiter=maxlsiter,
        decrease_factor=decrease_factor,
        max_stepsize=max_stepsize,
        condition=condition,
        jit=jit,
        unroll=unroll,
        verbose=verbose,
    )
  elif linesearch == "zoom":
    linesearch_solver = ZoomLineSearch(
        fun=fun,
        value_and_grad=value_and_grad,
        has_aux=has_aux,
        maxiter=maxlsiter,
        max_stepsize=max_stepsize,
        increase_factor=increase_factor,
        jit=jit,
        unroll=unroll,
        verbose=verbose,
    )
  elif linesearch == "hager-zhang":
    linesearch_solver = HagerZhangLineSearch(
        fun=fun,
        value_and_grad=value_and_grad,
        has_aux=has_aux,
        maxiter=maxlsiter,
        max_stepsize=max_stepsize,
        jit=jit,
        unroll=unroll,
        verbose=verbose,
    )
  else:
    raise ValueError(
        f"Linesearch {linesearch} not available/tested. "
        f"Available linesearches: {available_linesearches}"
    )
  return linesearch_solver


def _reset_stepsize(
    linesearch, max_stepsize, min_stepsize, increase_factor, stepsize
):
  """Set stepsize at the start of the linesearch from previous guess."""
  available_linesearches = ["backtracking", "zoom", "hager-zhang"]
  if linesearch == "hager-zhang":
    # FIXME: HZL should be able to use the previous stepsize (see the paper)
    # For now, the current implementation is simply initialized at the maximum
    # stepsize.
    init_stepsize = max_stepsize
  elif linesearch == "zoom":
    init_stepsize = stepsize
  elif linesearch == "backtracking":
    init_stepsize = jnp.where(
        stepsize <= min_stepsize,
        # If stepsize became too small, we restart it.
        max_stepsize,
        # Else, we increase a bit the previous one.
        stepsize * increase_factor,
    )
  else:
    raise ValueError(
        f"Linesearch {linesearch} not available/tested. "
        f"Available linesearches: {available_linesearches}"
    )
  return init_stepsize
