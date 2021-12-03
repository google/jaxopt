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

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt._src import tree_util


class OptaxState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  value: float
  error: float
  internal_state: NamedTuple
  aux: Any


@dataclass(eq=False)
class OptaxSolver(base.StochasticSolver):
  """Optax solver.

  Attributes:
    fun: a function of the form ``fun(params, *args, **kwargs)``, where
      ``params`` are parameters of the model,
      ``*args`` and ``**kwargs`` are additional arguments.
    opt: the optimizer to use, an optax.GradientTransformation, which is just a
      NamedTuple with ``init`` and ``update`` functions.
    pre_update: a function to execute before Optax's update.
      The function signature must be
      ``params, state = pre_update(params, state, *args, **kwargs)``.
    maxiter: maximum number of solver iterations.
    tol: tolerance to use.
    verbose: whether to print error on every iteration or not. verbose=True will
      automatically disable jit.
    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.
    has_aux: whether ``fun`` outputs one (False) or more values (True).
      When True it will be assumed by default that ``fun(...)[0]``
      is the objective value. The auxiliary outputs are stored in
      ``state.aux``.
    jit: whether to JIT-compile the optimization loop (default: "auto").
    unroll: whether to unroll the optimization loop (default: "auto").
  """
  fun: Callable
  opt: NamedTuple
  pre_update: Optional[Callable] = None
  maxiter: int = 500
  tol: float = 1e-3
  verbose: int = 0
  implicit_diff: bool = False
  implicit_diff_solve: Optional[Callable] = None
  has_aux: bool = False
  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"

  def init_state(self,
                 init_params: Any,
                 *args,
                 **kwargs) -> OptaxState:
    """Initialize the solver state.

    Args:
      init_params: pytree containing the initial parameters.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      state
    """
    del args, kwargs  # Not used.
    opt_state = self.opt.init(init_params)
    return OptaxState(iter_num=jnp.asarray(0),
                      value=jnp.asarray(jnp.inf),
                      error=jnp.asarray(jnp.inf),
                      aux=None,
                      internal_state=opt_state)

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
    Returns:
      (params, state)
    """
    if self.pre_update:
      params, state = self.pre_update(params, state, *args, **kwargs)

    (value, aux), grad = self._value_and_grad_fun(params, *args, **kwargs)

    delta, opt_state = self.opt.update(grad, state.internal_state, params)
    params = self._apply_updates(params, delta)

    # Computes optimality error before update to re-use grad evaluation.
    new_state = OptaxState(iter_num=state.iter_num + 1,
                           error=tree_util.tree_l2_norm(grad),
                           value=value,
                           aux=aux,
                           internal_state=opt_state)
    return base.OptStep(params=params, state=new_state)

  def optimality_fun(self, params, *args, **kwargs):
    """Optimality function mapping compatible with ``@custom_root``."""
    return self._grad_fun(params, *args, **kwargs)[0]

  def __post_init__(self):
    # Handle has_aux once for all.
    if self.has_aux:
      fun_with_aux = self.fun
    else:
      fun_with_aux = lambda p, *a, **kw: (self.fun(p, *a, **kw), None)

    # Pre-compile useful functions.
    self._value_and_grad_fun = jax.value_and_grad(fun_with_aux, has_aux=True)
    self._grad_fun = jax.grad(fun_with_aux, has_aux=True)

    self.reference_signature = self.fun
