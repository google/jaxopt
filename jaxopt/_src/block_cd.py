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

"""Implementation of block coordinate descent in JAX."""

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Union

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt._src import implicit_diff as idf
from jaxopt._src import loop
from jaxopt._src import objective
from jaxopt._src import tree_util


class BlockCDState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  error: float
  predictions: jnp.ndarray
  subfun_g: jnp.ndarray


@dataclass
class BlockCoordinateDescent(base.IterativeSolver):
  """Block coordinate solver.

  This solver minimizes::

    objective(params, hyperparams_prox, *args, **kwargs) =
      fun(params, *args, **kwargs) + non_smooth(params, hyperparams_prox)

  Attributes:
    fun: a smooth function of the form ``fun(params, *args, **kwargs)``.
      It should be a ``objective.CompositeLinearFunction`` object.
    block_prox: block-wise proximity operator associated with ``non_smooth``,
      a function of the form ``block_prox(x[j], hyperparams_prox, scaling=1.0)``.
      See ``jaxopt.prox`` for examples.
    maxiter: maximum number of proximal gradient descent iterations.
    tol: tolerance to use.
    verbose: whether to print error on every iteration or not.
      Warning: verbose=True will automatically disable jit.
    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.
    jit: whether to JIT-compile the optimization loop (default: "auto").
    unroll: whether to unroll the optimization loop (default: "auto").
  """
  fun: objective.CompositeLinearFunction
  block_prox: Callable
  maxiter: int = 500
  tol: float = 1e-4
  verbose: int = 0
  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None
  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"

  def init(self,
           init_params: Any,
           hyperparams_prox: Any,
           *args,
           **kwargs) -> base.OptStep:
    """Initialize the parameters and state.

    Args:
      init_params: pytree containing the initial parameters.
      hyperparams_prox: pytree containing hyperparameters of block_prox.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      (params, state)
    """
    del hyperparams_prox  # Not used.
    linop = self.fun.make_linop(*args, **kwargs)
    predictions = linop.matvec(init_params)
    subfun_g = self._grad_subfun(predictions, *args, **kwargs)
    state = BlockCDState(iter_num=0,
                         predictions=predictions,
                         subfun_g=subfun_g,
                         error=jnp.inf)
    return base.OptStep(params=init_params, state=state)

  def update(self,
             params: Any,
             state: NamedTuple,
             hyperparams_prox: Any,
             *args,
             **kwargs) -> base.OptStep:
    """Performs one epoch of block CD.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      hyperparams_prox: pytree containing hyperparameters of block_prox.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      (params, state)
    """
    linop = self.fun.make_linop(*args, **kwargs)
    stepsizes = 1.0 / self.fun.columnwise_lipschitz_const(*args, **kwargs)

    # todo: ability to permute block order.

    def body_fun(i, tup):
      x, subfun_g, predictions, sqerror_sum = tup
      x_i_old = x[i]
      g_i = linop.rmatvec_element(subfun_g, i)
      b = self.fun.b(*args, **kwargs)
      if b is not None:
        g_i += b[i]
      x_i_new = self.block_prox(x[i] - stepsizes[i] * g_i,
                                hyperparams_prox,
                                stepsizes[i])
      diff_i = x_i_new - x_i_old
      # A cheap-to-compute lower-bound of self.l2_optimality_error.
      sqerror_sum += jnp.sum(diff_i ** 2)
      x = x.at[i].set(x_i_new)
      predictions = linop.update_matvec(predictions, diff_i, i)
      subfun_g = self._grad_subfun(predictions, *args, **kwargs)
      return x, subfun_g, predictions, sqerror_sum

    init = (params, state.subfun_g, state.predictions, 0)
    params, subfun_g, predictions, sqerror_sum = jax.lax.fori_loop(
        lower=0, upper=params.shape[0], body_fun=body_fun, init_val=init)
    state = BlockCDState(iter_num=state.iter_num + 1,
                         predictions=predictions,
                         subfun_g=subfun_g,
                         error=jnp.sqrt(sqerror_sum))
    return base.OptStep(params=params, state=state)

  def _fixed_point_fun(self, params, hyperparams_prox, *args, **kwargs):
    grad_step = params - self._grad_fun(params, *args, **kwargs)
    return self._prox(grad_step, hyperparams_prox)

  def optimality_fun(self,
                     params: Any,
                     hyperparams_prox: Any,
                     *args,
                     **kwargs) -> Any:
    """Proximal-gradient fixed point residual.

    This function is compatible with ``@custom_root``.

    The fixed point function is defined as::

      fixed_point_fun(params, hyperparams_prox, *args, **kwargs) =
        prox(params - grad(fun)(params, *args, **kwargs), hyperparams_prox)

    where::

      prox = jax.vmap(block_prox, in_axes=(0, None))

    The residual is defined as::

      optimality_fun(params, hyperparams_prox, *args, **kwargs) =
        fixed_point_fun(params, hyperparams_prox, *args, **kwargs) - params

    Args:
      params: pytree containing the parameters.
      hyperparams_prox: pytree containing hyperparameters of block_prox.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      residual: pytree with same structure as ``params``.
    """
    fp = self._fixed_point_fun(params, hyperparams_prox, *args, **kwargs)
    return  fp - params

  def __post_init__(self):
    if not isinstance(self.fun, objective.CompositeLinearFunction):
      raise AttributeError("fun should be an instance of "
                           "objective.CompositeLinearFunction.")

    # Pre-compile useful functions.
    self._grad_fun = jax.grad(self.fun)
    self._grad_subfun = jax.grad(self.fun.subfun)
    self._prox = jax.vmap(self.block_prox, in_axes=(0, None))
