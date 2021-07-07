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
from typing import Union

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxopt import base
from jaxopt import implicit_diff3 as idf
from jaxopt import linear_solve
from jaxopt import loop
from jaxopt import tree_util


class OptaxState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  value: float
  error: float
  internal_state: NamedTuple
  aux: Any


@dataclass
class OptaxSolver:
  """Optax solver.

  Attributes:
    fun: a function of the form ``fun(params, *args, **kwargs)``, where
      ``params`` are parameters of the model,
      ``*args`` and ``**kwargs`` are additional arguments.
    opt: the optimizer to use, an optax.GradientTransformation, which is just a
      NamedTuple with ``init`` and ``update`` functions.
    pre_update_fun: a function to execute before Optax's update.
      The function signature must be
      ``params, state = pre_update_fun(params, state, *args, **kwargs)``.
    maxiter: maximum number of solver iterations.
    tol: tolerance to use.
    verbose: whether to print error on every iteration or not. verbose=True will
      automatically disable jit.
    implicit_diff: if True, enable implicit differentiation using cg,
      if Callable, do implicit differentiation using callable as linear solver,
      if False, use autodiff through the solver implementation (note:
        this will unroll syntactic loops).
    has_aux: whether ``fun`` outputs one (False) or more values (True).
      When True it will be assumed by default that ``fun(...)[0]``
      is the objective value. The auxiliary outputs are stored in
      ``state.aux``.
  """
  fun: Callable
  opt: NamedTuple
  pre_update_fun: Optional[Callable] = None
  maxiter: int = 500
  tol: float = 1e-3
  verbose: int = 0
  implicit_diff: Union[bool, Callable] = False
  has_aux: bool = False

  def init(self,
           init_params: Any) -> base.OptStep:
    """Initialize the ``(params, state)`` pair.

    Args:
      init_params: pytree containing the initial parameters.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    opt_state = self.opt.init(init_params)
    state = OptaxState(iter_num=0,
                       value=jnp.inf,
                       error=jnp.inf,
                       aux=None,
                       internal_state=opt_state)
    return base.OptStep(params=init_params, state=state)

  def _apply_updates(self, params, updates):
    update_fun = lambda p, u: jnp.asarray(p + u).astype(jnp.asarray(p).dtype)
    return jax.tree_multimap(update_fun, params, updates)

  def update(self,
             params: Any,
             state: NamedTuple,
             *args,
             **kwargs) -> base.OptStep:
    """Performs one iteration of the optax solver.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    if self.pre_update_fun:
      params, state = self.pre_update_fun(params, state, *args, **kwargs)

    if self.has_aux:
      (value, aux), grad = self._value_and_grad_fun(params, *args, **kwargs)
    else:
      value, grad = self._value_and_grad_fun(params, *args, **kwargs)
      aux = None

    delta, opt_state = self.opt.update(grad, state.internal_state, params)
    params = self._apply_updates(params, delta)
    error = self.l2_optimality_error(params, *args, **kwargs)
    new_state = OptaxState(iter_num=state.iter_num + 1,
                           error=error,
                           value=value,
                           aux=aux,
                           internal_state=opt_state)
    return base.OptStep(params=params, state=new_state)

  def run(self,
          init_params: Any,
          *args,
          **kwargs) -> base.OptStep:
    """Runs the optax solver on a fixed dataset.

    Args:
      init_params: pytree containing the initial parameters.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Return type:
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
                           init_val=self.init(init_params),
                           maxiter=self.maxiter, jit=self._jit,
                           unroll=self._unroll)

  def run_iterator(self,
                   init_params: Any,
                   iterator,
                   *args,
                   **kwargs) -> base.OptStep:
    """Runs the optax solver on a dataset iterator.

    Args:
      init_params: pytree containing the initial parameters.
      iterator: iterator generating data batches.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    params, state = self.init(init_params)


    for _ in range(self.maxiter):
      try:
        data = next(iterator)
      except StopIteration:
        break

      params, state = self.update(params, state, *args, **kwargs, data=data)

    return base.OptStep(params=params, state=state)

  def optimality_fun(self, params, *args, **kwargs):
    """Optimality function mapping compatible with ``@custom_root``."""
    if self.has_aux:
      return self._grad_fun(params, *args, **kwargs)[0]
    else:
      return self._grad_fun(params, *args, **kwargs)

  def l2_optimality_error(self, params, *args, **kwargs):
    """Computes the L2 optimality error."""
    optimality = self.optimality_fun(params, *args, **kwargs)
    return tree_util.tree_l2_norm(optimality)

  def __post_init__(self):
    # Pre-compile useful functions.
    self._value_and_grad_fun = jax.value_and_grad(self.fun,
                                                  has_aux=self.has_aux)
    self._grad_fun = jax.grad(self.fun, has_aux=self.has_aux)

    # We always jit unless verbose mode is enabled.
    self._jit = not self.verbose
    # We unroll when implicit diff is disabled or when jit is disabled.
    self._unroll = not self.implicit_diff or not self._jit

    # Set up implicit diff.
    if self.implicit_diff:
      if isinstance(self.implicit_diff, Callable):
        solve = self.implicit_diff
      else:
        solve = linear_solve.solve_normal_cg

      decorator = idf.custom_root(self.optimality_fun,
                                  has_aux=True,
                                  solve=solve)
      # pylint: disable=g-missing-from-attributes
      self.run = decorator(self.run)

