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

"""Fixed Point Iteration in JAX."""

from typing import Any
from typing import Callable
from typing import NamedTuple

from dataclasses import dataclass

import jax.numpy as jnp
from jax.tree_util import tree_leaves, tree_structure

from jaxopt._src import base
from jaxopt._src import linear_solve
from jaxopt._src.tree_util import tree_l2_norm, tree_sub


class FixedPointState(NamedTuple):
  """Named tuple containing state information.
  
  Attributes:
    iter_num: iteration number
    value: pytree of current estimate of fixed point
    error: residuals of current estimate
  """
  iter_num: int
  value: Any
  error: float



@dataclass
class FixedPointIteration(base.IterativeSolver):
  """Fixed point resolution by iterating.

  fixed_point_fun should fulfil Banach fixed-point theorem assumptions.
  Otherwise convergence is not guaranteed.

  Attributes:
    fixed_point_fun: a function ``fixed_point_fun(x, *args, **kwargs)``
      returning a pytree with the same structure and type as x
      each leaf must be an array (not a scalar)
    maxiter: maximum number of iterations.
    tol: tolerance (stopping criterion)
    has_aux: wether fixed_point_fun returns additional data. (default: False)
      if True, the fixed is computed only with respect to first element of the sequence
      returned. Other elements are carried during computation.
    verbose: whether to print error on every iteration or not.
      Warning: verbose=True will automatically disable jit.
    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.
    jit: whether to JIT-compile the optimization loop (default: "auto").
    unroll: whether to unroll the optimization loop (default: "auto")
  """
  fixed_point_fun: Callable
  maxiter: int = 100
  tol: float = 1e-5
  has_aux: bool = False
  verbose: bool = False
  implicit_diff: bool = True
  implicit_diff_solve: Callable = linear_solve.solve_normal_cg
  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"

  def _params(self, fpf_return):
    return fpf_return[0] if self.has_aux else fpf_return

  def init(self,
           init_params,
           *args,
           **kwargs) -> base.OptStep:
    """Initialize the ``(params, state)`` pair.
    Args:
      init_params: initial guess of the fixed point, pytree
      *args: additional positional arguments to be passed to ``optimality_fun``.
      **kwargs: additional keyword arguments to be passed to ``optimality_fun``.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    fpf_return = self.fixed_point_fun(init_params, *args, **kwargs)
    params = self._params(fpf_return)
    state = FixedPointState(iter_num=0,
                            value=fpf_return,
                            error=jnp.inf)
    return base.OptStep(params=params, state=state)

  def update(self,
             params: Any,
             state: NamedTuple,
             *args,
             **kwargs) -> base.OptStep:
    """Performs one iteration of fixed point iterations.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      *args: additional positional arguments to be passed to ``fixed_point_fun``.
      **kwargs: additional keyword arguments to be passed to ``fixed_point_fun``.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    fpf_return = self.fixed_point_fun(params, *args, **kwargs)
    next_params = self._params(fpf_return)
    error = tree_l2_norm(tree_sub(next_params, params))
    next_state = FixedPointState(iter_num=state.iter_num+1,
                                 value=fpf_return,
                                 error=error)
    return base.OptStep(params=next_params, state=next_state)

  def optimality_fun(self, params, *args, **kwargs):
    """Optimality function mapping compatible with ``@custom_root``."""
    fpf_return = self.fixed_point_fun(params, *args, **kwargs)
    f_params = self._params(fpf_return)
    return tree_sub(f_params, params)
