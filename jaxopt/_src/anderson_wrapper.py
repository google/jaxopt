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

"""Wrapper to accelerate iterative solver with Anderson."""

from typing import Any
from typing import Callable
from typing import Optional
from typing import NamedTuple
from typing import Union

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt._src.tree_util import tree_l2_norm, tree_sub, tree_map
from jaxopt._src.anderson import AndersonAcceleration
from jaxopt._src.anderson import anderson_step, update_history


class AndersonWrapperState(NamedTuple):
  """Named tuple containing state information.

  Attributes:
    iter_num: iteration number
    solver_state: state of the solver wrapped
    error: residuals of current estimate
    params_history: history of previous anderson iterates
    residuals_history: residuals of previous iterates
      fixed_point_fun(params_history) - params_history
    residual_gram: Gram matrix: G.T @ G with G the matrix of residuals
      each column of G is a flattened pytree of residuals_history
  """
  iter_num: int
  solver_state: Any
  error: float
  params_history: Any
  residuals_history: Any
  residual_gram: jnp.ndarray


@dataclass(eq=False)
class AndersonWrapper(base.IterativeSolver):
  """Wrapper for accelerating JAXopt solvers.

  Note that the internal solver state can be accessed via the ``aux`` attribute
  of AndersonState.

  Attributes:
    solver: solver object to accelerate. Must exhibit init() and update() methods.
    history_size: size of history. Affect memory cost. (default: 5).
    mixing_frequency: frequency of Anderson updates. (default: ``history_size``).
      Only one every ``mixing_frequency`` updates uses Anderson, while the other
      updates use regular fixed point iterations.
    beta: momentum in Anderson updates. (default: 1).
    ridge: ridge regularization in solver.
      Consider increasing this value if the solver returns ``NaN``.
    verbose: whether to print information on every iteration or not.
    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.
    jit: whether to JIT-compile the optimization loop (default: True).
    unroll: whether to unroll the optimization loop (default: "auto")
  """
  solver: base.IterativeSolver
  history_size: int = 5
  mixing_frequency: int = None
  beta: float = 1.
  ridge: float = 1e-5
  verbose: Union[bool, int] = False
  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None
  jit: bool = True
  unroll: base.AutoOrBoolean = "auto"

  def init_state(self, init_params, *args, **kwargs) -> AndersonWrapperState:
    solver_state = self.solver.init_state(init_params, *args, **kwargs)
    m = self.history_size
    params_history = tree_map(lambda x: jnp.tile(x, [m]+[1]*x.ndim),
                              init_params)
    residuals_history = tree_map(jnp.zeros_like, params_history)
    residual_gram = jnp.zeros((m,m))
    return AndersonWrapperState(iter_num=jnp.asarray(0),
                                solver_state=solver_state,
                                error=solver_state.error,
                                params_history=params_history,
                                residuals_history=residuals_history,
                                residual_gram=residual_gram)

  def update(self, params, state, *args, **kwargs) -> base.OptStep:
    """Perform one step of Anderson acceleration over the internal solver update.

    The reset_state attribute is used to update the internal solver state after
    the Anderson step.

    Args:
      params: parameters optimized by solver.
        Only its pytree structure matters (content unused).
      state: AndersonWrapperState
        Crucially, state.params_history and state.residuals_history are the
        sequences used to generate next iterate.  Note: state.solver_state is
        the internal solver state.
      args,kwargs: additional parameters passed to ``update`` method of internal
        solver Note: sometimes those are hyper-parameters of the solver, but if
        the solver is a Jaxopt solver they will be forwarded to the underlying
        function being optimized
    """
    iter_num = state.iter_num
    anderson_freq = jnp.equal(jnp.mod(iter_num, self.mixing_frequency), 0)
    is_not_init = jnp.greater_equal(iter_num, self.history_size)

    def perform_anderson_step(t):
      _, state = t
      extrapolated = anderson_step(state.params_history,
                                   state.residuals_history,
                                   state.residual_gram,
                                   self.ridge, self.beta)
      solver_state = self.solver.init_state(extrapolated, *args, **kwargs)
      return extrapolated, solver_state

    def use_param(t):
      params, state = t
      return params, state.solver_state

    extrapolated, solver_state = jax.lax.cond(
      jnp.logical_and(anderson_freq, is_not_init),
      perform_anderson_step,  # extrapolation
      use_param,  # re-use previous iterate instead
      operand=(params, state)
    )

    params_history = state.params_history
    residuals_history = state.residuals_history
    residual_gram = state.residual_gram
    pos = jnp.mod(state.iter_num, self.history_size)

    next_params, solver_state = self.solver.update(extrapolated, solver_state,
                                                   *args, **kwargs)

    residual = tree_sub(next_params, extrapolated)
    ret = update_history(pos, params_history, residuals_history,
                         residual_gram, extrapolated, residual)
    params_history, residuals_history, residual_gram, error = ret

    next_state = AndersonWrapperState(iter_num=state.iter_num+1,
                                      solver_state=solver_state,
                                      error=solver_state.error,
                                      params_history=params_history,
                                      residuals_history=residuals_history,
                                      residual_gram=residual_gram)
    
    if self.verbose:
      self.log_info(next_state, error_name="Inner Solver Error")
    return base.OptStep(params=next_params, state=next_state)

  def optimality_fun(self, params, *args, **kwargs):
    """Optimality function mapping compatible with ``@custom_root``."""
    return self.solver.optimality_fun(params, *args, **kwargs)

  def __post_init__(self):
    super().__post_init__()

    self.maxiter = self.solver.maxiter
    self.tol = self.solver.tol

    if self.mixing_frequency is None:
      self.mixing_frequency = self.history_size

    self.reference_signature = self.solver.reference_signature
