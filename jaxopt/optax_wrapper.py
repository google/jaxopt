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

"""Optax wrapper for JAXopt."""

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxopt import implicit_diff2 as idf
from jaxopt import linear_solve
from jaxopt import loop
from jaxopt import tree_util


class OptaxState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  error: float
  internal_state: NamedTuple


@dataclass
class OptaxSolver:
  """Optax solver.

  Attributes:
    fun: a function of the form ``fun(params, hyperparams, data)``, where
      ``params`` are parameters of the model,
      ``hyperparams`` are hyper-parameters of the model, and
      ``data`` are any extra arguments such as data, rng, etc.
    opt: the optimizer to use, an optax.GradientTransformation, which is just a
      NamedTuple with ``init`` and ``update`` functions.
    pre_update_fun: a function to execute before Optax's update.
      The function signature must be
      ``params, state = pre_update_fun(params, state, hyperparams, data)``.
    maxiter: maximum number of solver iterations.
    tol: tolerance to use.
    verbose: whether to print error on every iteration or not. verbose=True will
      automatically disable jit.
    implicit_diff: if True, enable implicit differentiation using cg,
      if Callable, do implicit differentiation using callable as linear solver,
      if False, use autodiff through the solver implementation (note:
        this will unroll syntactic loops).
    has_aux: whether function fun outputs one (False) or more values (True).
      When True it will be assumed by default that ``fun(...)[0]``
      is the objective.
  """
  fun: Callable
  opt: NamedTuple
  pre_update_fun: Optional[Callable] = None
  maxiter: int = 500
  tol: float = 1e-3
  verbose: int = 0
  implicit_diff: Union[bool, Callable] = True
  has_aux: bool = False

  def init(self, init_params: Any) -> Tuple[Any, NamedTuple]:
    """Initialize the ``(params, state)`` pair.

    Args:
      init_params: pytree containing the initial parameters.
    Returns:
      (params, state)
    """
    opt_state = self.opt.init(init_params)
    state = OptaxState(iter_num=0, error=jnp.inf, internal_state=opt_state)
    return init_params, state

  def _apply_updates(self, params, updates):
    update_fun = lambda p, u: jnp.asarray(p + u).astype(jnp.asarray(p).dtype)
    return jax.tree_multimap(update_fun, params, updates)

  def update(self,
             params: Any,
             state: NamedTuple,
             hyperparams: Any,
             data: Any) -> Tuple[Any, NamedTuple]:
    """Performs one iteration of the optax solver.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      hyperparams: pytree containing hyper-parameters.
      data: pytree containing data.
    Returns:
      (params, state)
    """
    if self.pre_update_fun:
      params, state = self.pre_update_fun(params, state, hyperparams, data)

    grad = self.grad_fun(params, hyperparams, data)
    delta, opt_state = self.opt.update(grad, state.internal_state, params)
    params = self._apply_updates(params, delta)
    error = self.l2_optimality_error(params, hyperparams, data)
    new_state = OptaxState(iter_num=state.iter_num + 1,
                           error=error,
                           internal_state=opt_state)
    return params, new_state

  def run(self,
          hyperparams: Any,
          data: Any,
          init_params: Any) -> Tuple[Any, NamedTuple]:
    """Runs the optax solver on a fixed dataset.

    Args:
      hyperparams: pytree containing hyper-parameters.
      data: pytree containing the entire data.
      init_params: pytree containing the initial parameters.
    Returns:
      (params, info)
    """
    def cond_fun(pair):
      _, state = pair
      if self.verbose:
        print(state.error)
      return state.error > self.tol

    def body_fun(pair):
      params, state = pair
      return self.update(params, state, hyperparams, data)

    return loop.while_loop(cond_fun=cond_fun, body_fun=body_fun,
                           init_val=self.init(init_params),
                           maxiter=self.maxiter, jit=self.jit,
                           unroll=self.unroll)

  def run_iterator(self,
                   hyperparams: Any,
                   iterator: Any,
                   init_params: Any) -> Tuple[Any, NamedTuple]:
    """Runs the optax solver on a dataset iterator.

    Args:
      hyperparams: pytree containing hyper-parameters.
      iterator: iterator generating data batches.
      init_params: pytree containing the initial parameters.
    Returns:
      (params, info)
    """
    params, state = self.init(init_params)


    for _ in range(self.maxiter):
      try:
        data = next(iterator)
      except StopIteration:
        break

      params, state = self.update(params, state, hyperparams, data)

    return params, state

  def optimality_fun(self, sol, hyperparams, data):
    """Optimality function mapping compatible with ``@custom_root``."""
    return self.grad_fun(sol, hyperparams, data)

  def l2_optimality_error(self, params, hyperparams, data):
    """Computes the L2 optimality error."""
    optimality = self.optimality_fun(params, hyperparams, data)
    return tree_util.tree_l2_norm(optimality)

  def __post_init__(self):
    if self.has_aux:
      self.fun = jax.jit(lambda x, par: self.fun(x, par)[0])
    else:
      self.fun = jax.jit(self.fun)
    self.grad_fun = jax.jit(jax.grad(self.fun))
    # We always jit unless verbose mode is enabled.
    self.jit = not self.verbose
    # We unroll when implicit diff is disabled or when jit is disabled.
    self.unroll = not self.implicit_diff or not self.jit

    if self.implicit_diff:
      if isinstance(self.implicit_diff, Callable):
        solve = implicit_diff
      else:
        solve = linear_solve.solve_normal_cg

      decorator = idf.custom_root(self.optimality_fun,
                                  has_aux=True,
                                  solve=solve)
      self.run = decorator(self.run)

