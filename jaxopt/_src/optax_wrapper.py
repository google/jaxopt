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
    value_and_grad: whether ``fun`` just returns the value (False) or both
      the value and gradient (True).
    has_aux: whether ``fun`` outputs auxiliary data or not.
      If ``has_aux`` is False, ``fun`` is expected to be
        scalar-valued.
      If ``has_aux`` is True, then we have one of the following
        two cases.
      If ``value_and_grad`` is False, the output should be
      ``value, aux = fun(...)``.
      If ``value_and_grad == True``, the output should be
      ``(value, aux), grad = fun(...)``.
      At each iteration of the algorithm, the auxiliary outputs are stored
        in ``state.aux``.

    pre_update: a function to execute before Optax's update.
      The function signature must be
      ``params, state = pre_update(params, state, *args, **kwargs)``.

    maxiter: maximum number of solver iterations.
    tol: tolerance to use.
    verbose: whether to print information on every iteration or not.

    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.

    jit: whether to JIT-compile the optimization loop (default: True).
    unroll: whether to unroll the optimization loop (default: "auto").
  """
  fun: Callable
  opt: NamedTuple
  value_and_grad: bool = False
  pre_update: Optional[Callable] = None
  maxiter: int = 500
  tol: float = 1e-3
  verbose: Union[bool, int] = False
  implicit_diff: bool = False
  implicit_diff_solve: Optional[Callable] = None
  has_aux: bool = False
  jit: bool = True
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
    opt_state = self.opt.init(init_params)

    value, aux = self._fun(init_params, *args, **kwargs)

    params_dtype = tree_util.tree_single_dtype(init_params)

    return OptaxState(iter_num=jnp.asarray(0),
                      value=jnp.asarray(jnp.inf, value.dtype),
                      error=jnp.asarray(jnp.inf, dtype=params_dtype),
                      aux=aux,
                      internal_state=opt_state)

  def _apply_updates(self, params, updates):
    update_fun = lambda p, u: jnp.asarray(p + u).astype(jnp.asarray(p).dtype)
    return jax.tree_map(update_fun, params, updates)

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
    dtype = tree_util.tree_single_dtype(params)
    error = tree_util.tree_l2_norm(grad)
    new_state = OptaxState(iter_num=state.iter_num + 1,
                           error=jnp.asarray(error, dtype=dtype),
                           value=jnp.asarray(value),
                           aux=aux,
                           internal_state=opt_state)
    
    if self.verbose:
      self.log_info(
          new_state,
          error_name="Gradient Norm",
          additional_info={"Objective Value": value}
      )
    return base.OptStep(params=params, state=new_state)

  def optimality_fun(self, params, *args, **kwargs):
    """Optimality function mapping compatible with ``@custom_root``."""
    return self._grad_fun(params, *args, **kwargs)[0]

  def __post_init__(self):
    super().__post_init__()

    self._fun, self._grad_fun, self._value_and_grad_fun = \
      base._make_funs_with_aux(fun=self.fun,
                               value_and_grad=self.value_and_grad,
                               has_aux=self.has_aux)

    self.reference_signature = self.fun
