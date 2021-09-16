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

"""Anderson Acceleration for finding a fixed point in JAX."""

from typing import Any
from typing import Callable
from typing import NamedTuple

from dataclasses import dataclass

import jax.numpy as jnp
from jax.tree_util import tree_leaves, tree_structure

from jaxopt._src import base
from jaxopt._src import linear_solve
from jaxopt._src.tree_util import tree_collapse, tree_uncollapse, tree_l2_norm, tree_sub


class AndersonState(NamedTuple):
  """Named tuple containing state information.
  
  Attributes:
    iter_num: iteration number
    value: pytree of current estimate of fixed point
    error: residuals of current estimate
    v_history: history of previous iterates
    f_history: fixed_point_fun(v_history)
  """
  iter_num: int
  value: Any
  error: float
  v_history: Any
  f_history: Any


@dataclass
class AndersonAcceleration(base.IterativeSolver):
  """Multi-dimensional fixed point solver using Anderson.

  Read [1] to see the hypothesis ``fixed_point_fun`` must fulfil
  to ensure convergence. In particular, if Banach fixed point theorem
  hypothesis hold, Anderson acceleration will converge.
  
  [1] Pollock, Sara, and Leo Rebholz.
  "Anderson acceleration for contractive and noncontractive operators."
  arXiv preprint arXiv:1909.04638 (2019).

  Attributes:
    fixed_point_fun: a function ``fixed_point_fun(x, *args, **kwargs)``
      returning a pytree with the same structure and type as x
      each leaf must be an array (not a scalar)
    history_size: size of history. Affect memory cost.
    beta: momentum in Anderson updates. Default = 1.
    maxiter: maximum number of iterations.
    tol: tolerance (stoping criterion).
    ridge: ridge regularization in solver.
      Consider increasing this value if the solver returns ``NaN``.
    has_aux: wether fixed_point_fun returns additional data. (default: False)
      if True, the fixed-point is computed only with respect to first element of the sequence
      returned. ``AndersonState.value`` will contain the full sequence returned.
    verbose: whether to print error on every iteration or not.
      Warning: verbose=True will automatically disable jit.
    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.
    jit: whether to JIT-compile the optimization loop (default: "auto").
    unroll: whether to unroll the optimization loop (default: "auto")
  """
  fixed_point_fun: Callable = None
  history_size: int = 5
  beta: float = 1.
  maxiter: int = 100
  tol: float = 1e-5
  ridge: float = 1e-5
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
      *args: additional positional arguments to be passed to ``fixed_point_fun``.
      **kwargs: additional keyword arguments to be passed to ``fixed_point_fun``.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    params = init_params
    params_history = [tree_collapse(params)]
    for _ in range(self.history_size):
      fpf_return = self.fixed_point_fun(params, *args, **kwargs)
      params = self._params(fpf_return)
      params_history.append(tree_collapse(params))

    v_h = jnp.stack(params_history[:-1], axis=1)
    f_h = jnp.stack(params_history[1:], axis=1)
    error = jnp.linalg.norm(params_history[-1] - params_history[-2])

    state = AndersonState(iter_num=0,
                          value=fpf_return,
                          error=error,
                          v_history=v_h,
                          f_history=f_h)
    return base.OptStep(params=params, state=state)

  def _minimize_residuals(self, m, G):
    c = jnp.zeros((m+1)).at[0].set(1)
    GTG = G.T @ G
    GTG = GTG + self.ridge * jnp.eye(m) # avoid ill-posed systems
    H = jnp.block([[jnp.zeros((1, 1)), jnp.ones((1, m))],
                   [ jnp.ones((m, 1)),       GTG       ]])
    alpha = jnp.linalg.solve(H, c)
    return alpha

  def update(self,
             params: Any,
             state: NamedTuple,
             *args,
             **kwargs) -> base.OptStep:
    """Performs one iteration of the Anderson acceleration.

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
    m = self.history_size
    v_h = state.v_history
    f_h = state.f_history
    pos = jnp.mod(state.iter_num, self.history_size)
    
    G = f_h - v_h  # residuals
    alpha = self._minimize_residuals(m, G)
    alpha = alpha[1:]  # drop dummy variable (constraint satisfaction)
    
    # get next iterate from linear combination
    old = jnp.dot(v_h, alpha)
    new = jnp.dot(f_h, alpha)

    # get next iterate from convex combination
    aa_params_flat = (1-self.beta) * old + self.beta * new
    aa_params = tree_uncollapse(params, aa_params_flat)

    fpf_return = self.fixed_point_fun(aa_params, *args, **kwargs)
    faa_params = self._params(fpf_return)

    error = tree_l2_norm(tree_sub(faa_params, aa_params))

    faa_params_flat = tree_collapse(faa_params)
    v_history = state.v_history.at[:,pos].set(aa_params_flat)
    f_history = state.f_history.at[:,pos].set(faa_params_flat)

    next_state = AndersonState(iter_num=state.iter_num+1,
                               value=fpf_return,
                               error=error,  
                               v_history=v_history,
                               f_history=f_history)

    return base.OptStep(params=faa_params, state=next_state)

  def optimality_fun(self, params, *args, **kwargs):
    """Optimality function mapping compatible with ``@custom_root``."""
    fpf_return = self.fixed_point_fun(params, *args, **kwargs)
    f_params = self._params(fpf_return)
    return tree_sub(f_params, params)

  def __post_init__(self):
    if self.history_size < 2:
      raise ValueError("You must set m >= 2. Otherwise you should use ``PicardIterations``.")
