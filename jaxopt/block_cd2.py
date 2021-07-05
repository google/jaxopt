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

from jaxopt import base
from jaxopt import implicit_diff2 as idf
from jaxopt import linear_solve
from jaxopt import loop
from jaxopt import tree_util


class BlockCDState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  error: float
  predictions: jnp.ndarray
  subfun_g: jnp.ndarray


@dataclass
class BlockCoordinateDescent:
  """Block coordinate solver.

  This solver minimizes::

    objective(params, hyperparams, data) =
      fun(params, hyperparams_fun, data) + non_smooth(params, hyperparams_prox)

  where ``(hyperparams_fun, hyperparams_prox) = hyperparams``

  Attributes:
    fun: a smooth function of the form ``fun(x, hyperparams_fun, data)``.
      It should be a ``base.CompositeLinearFunction`` object.
    block_prox: block-wise proximity operator associated with ``non_smooth``,
      a function of the form ``block_prox(x[j], params_prox, scaling=1.0)``.
      See ``jaxopt.prox`` for examples.
    maxiter: maximum number of proximal gradient descent iterations.
    tol: tolerance to use.
    verbose: whether to print error on every iteration or not.
      Warning: verbose=True will automatically disable jit.
    implicit_diff: if True, enable implicit differentiation using cg,
      if Callable, do implicit differentiation using callable as linear solver,
      if False, use autodiff through the solver implementation (note:
        this will unroll syntactic loops).
  """
  fun: Callable
  block_prox: Callable
  maxiter: int = 500
  tol: float = 1e-4
  verbose: int = 0
  implicit_diff: Union[bool, Callable] = True

  def init(self,
           init_params: Any,
           hyperparams: Optional[jnp.ndarray] = None,
           data: Optional[jnp.ndarray] = None) -> base.OptStep:
    """Initialize the ``(params, state)`` pair.

    Args:
      init_params: pytree containing the initial parameters.
      hyperparams: tuple ``(hyperparams_fun, hyperparams_prox)`` containing the
        hyper-parameters (i.e., differentiable arguments) of ``fun`` and
        ``prox``.
      data: pytree containing data, i.e., differentiable arguments to be passed
        to ``fun``.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    if hyperparams is None:
      hyperparams = None, None

    linop = self.fun.make_linop(data)
    predictions = linop.matvec(init_params)
    hyperparams_fun, _ = hyperparams
    subfun_g = self._grad_subfun(predictions, hyperparams_fun, data)
    state = BlockCDState(iter_num=0,
                         predictions=predictions,
                         subfun_g=subfun_g,
                         error=jnp.inf)
    return base.OptStep(params=init_params, state=state)

  def _update_composite(self, params, state, hyperparams, data):
    """Performs one epoch of block CD in the composite case."""

    hyperparams_fun, hyperparams_prox = hyperparams
    linop = self.fun.make_linop(data)
    stepsizes = 1.0 / self.fun.columnwise_lipschitz_const(hyperparams_fun, data)



    def body_fun(i, tup):
      x, subfun_g, predictions, sqerror_sum = tup
      x_i_old = x[i]
      g_i = linop.rmatvec_element(subfun_g, i)
      b = self.fun.b(data)
      if b is not None:
        g_i += b[i]
      x_i_new = self.block_prox(x[i] - stepsizes[i] * g_i,
                                hyperparams_prox,
                                stepsizes[i])
      diff_i = x_i_new - x_i_old
      # A cheap-to-compute lower-bound of self.l2_optimality_error.
      sqerror_sum += jnp.sum(diff_i ** 2)
      x = jax.ops.index_update(x, i, x_i_new)
      predictions = linop.update_matvec(predictions, diff_i, i)
      subfun_g = self._grad_subfun(predictions, hyperparams_fun, data)
      return x, subfun_g, predictions, sqerror_sum

    init = (params, state.subfun_g, state.predictions, 0)
    params, subfun_g, predictions, sqerror_sum = jax.lax.fori_loop(
        lower=0, upper=params.shape[0], body_fun=body_fun, init_val=init)
    state = BlockCDState(iter_num=state.iter_num + 1,
                         predictions=predictions,
                         subfun_g=subfun_g,
                         error=jnp.sqrt(sqerror_sum))
    return base.OptStep(params=params, state=state)

  def update(self,
             params: Any,
             state: NamedTuple,
             hyperparams: Optional[Any] = None,
             data: Optional[Any] = None) -> base.OptStep:
    """Performs one epoch of block CD.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      hyperparams: tuple ``(hyperparams_fun, hyperparams_prox)`` containing the
        hyper-parameters (i.e., differentiable arguments) of ``fun`` and
        ``prox``.
      data: pytree containing data, i.e., differentiable arguments to be passed
        to ``fun``.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    if hyperparams is None:
      hyperparams = None, None


    return self._update_composite(params, state, hyperparams, data)

  def run(self,
          init_params: Any,
          hyperparams: Optional[Any] = None,
          data: Optional[Any] = None) -> base.OptStep:
    """Runs block CD until convergence or max number of iterations.

    Args:
      init_params: pytree containing the initial parameters.
      hyperparams: tuple ``(hyperparams_fun, hyperparams_prox)`` containing the
        hyper-parameters (i.e., differentiable arguments) of ``fun`` and
        ``prox``.
      data: pytree containing data, i.e., differentiable arguments to be passed
        to ``fun``.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    def cond_fun(opt_step):
      _, state = opt_step
      if self.verbose:
        print(state.iter_num, state.error)
      return state.error > self.tol

    def body_fun(opt_step):
      params, state = opt_step
      return self.update(params, state, hyperparams, data)

    return loop.while_loop(cond_fun=cond_fun, body_fun=body_fun,
                           init_val=self.init(init_params, hyperparams, data),
                           maxiter=self.maxiter, jit=self._jit,
                           unroll=self._unroll)

  def _fixed_point_fun(self, params, hyperparams, data):
    hyperparams_fun, hyperparams_prox = hyperparams
    grad_step = params - self._grad_fun(params, hyperparams_fun, data)
    return self._prox(grad_step, hyperparams_prox)

  def optimality_fun(self,
                     params: Any,
                     hyperparams: Any,
                     data: Any) -> Any:
    """Proximal-gradient fixed point residual.

    This function is compatible with ``@custom_root``.

    The fixed point function is defined as::

      fixed_point_fun(params, hyperparams, data) =
        prox(params - grad(fun)(params, hyperparams_fun), hyperparams_prox)

    where::

      hyperparams = (params_fun, params_prox)
      prox = jax.vmap(block_prox, in_axes=(0, None))

    The residual is defined as::

      optimality_fun(params, hyperparams, data) =
        fixed_point_fun(params, hyperparams, data) - params

    Args:
      params: pytree containing the parameters.
      hyperparams: tuple ``(hyperparams_fun, hyperparams_prox)`` containing the
        hyper-parameters (i.e., differentiable arguments) of ``fun`` and
        ``prox``.
      data: pytree containing data, i.e., differentiable arguments to be passed
        to ``fun``.
    Returns:
      residual: pytree with same structure as ``params``.
    """
    return self._fixed_point_fun(params, hyperparams, data) - params

  def l2_optimality_error(self,
                          params: Any,
                          hyperparams: Any,
                          data: Any) -> float:
    """L2 norm of the proximal-gradient fixed point residual.

    Args:
      params: pytree containing the parameters.
      hyperparams: tuple ``(hyperparams_fun, hyperparams_prox)`` containing the
        hyper-parameters (i.e., differentiable arguments) of ``fun`` and
        ``prox``.
      data: pytree containing data, i.e., differentiable arguments to be passed
        to ``fun``.
    Returns:
      l2_norm
    """
    optimality = self.optimality_fun(params, hyperparams, data)
    return tree_util.tree_l2_norm(optimality)

  def __post_init__(self):
    # Pre-compile useful functions.
    self._grad_fun = jax.grad(self.fun)
    self._grad_subfun = jax.grad(self.fun.subfun)
    self._prox = jax.vmap(self.block_prox, in_axes=(0, None))

    # We always jit unless verbose mode is enabled.
    self._jit = not self.verbose
    # We unroll when implicit diff is disabled or when jit is disabled.
    self._unroll = not self.implicit_diff or not self._jit

    # Set up implicit differentiation.
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
