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
from typing import Union

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt._src import implicit_diff as idf
from jaxopt._src import linear_solve
from jaxopt._src import loop
from jaxopt._src import tree_util


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

    objective(params, hyperparams_prox, *args, **kwargs) =
      fun(params, *args, **kwargs) + non_smooth(params, hyperparams_prox)

  Attributes:
    fun: a smooth function of the form ``fun(params, *args, **kwargs)``.
      It should be a ``objectives.CompositeLinearFunction`` object.
    block_prox: block-wise proximity operator associated with ``non_smooth``,
      a function of the form ``block_prox(x[j], hyperparams_prox, scaling=1.0)``.
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
  implicit_diff: Union[bool, Callable] = False

  def init(self,
           init_params: Any,
           *args,
           **kwargs) -> base.OptStep:
    """Initialize the ``(params, state)`` pair.

    Args:
      init_params: pytree containing the initial parameters.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
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
    Return type:
      base.OptStep
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
      x = jax.ops.index_update(x, i, x_i_new)
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

  def run(self,
          init_params: Any,
          hyperparams_prox,
          *args,
          **kwargs) -> base.OptStep:
    """Runs block CD until convergence or max number of iterations.

    Args:
      init_params: pytree containing the initial parameters.
      hyperparams_prox: pytree containing hyperparameters of block_prox.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
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
      return self.update(params, state, hyperparams_prox, *args, **kwargs)

    return loop.while_loop(cond_fun=cond_fun, body_fun=body_fun,
                           init_val=self.init(init_params, *args, **kwargs),
                           maxiter=self.maxiter, jit=self._jit,
                           unroll=self._unroll)

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

  def l2_optimality_error(self,
                          params: Any,
                          *args,
                          **kwargs) -> float:
    """L2 norm of the proximal-gradient fixed point residual.

    Args:
      params: pytree containing the parameters.
      hyperparams_prox: pytree containing hyperparameters of block_prox.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      l2_norm
    """
    optimality = self.optimality_fun(params, hyperparams_prox, *args, **kwargs)
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
