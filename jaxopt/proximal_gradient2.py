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

"""Implementation of proximal gradient descent in JAX."""

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxopt import base
from jaxopt import implicit_diff2 as idf
from jaxopt import linear_solve
from jaxopt import loop
from jaxopt.prox import prox_none
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_l2_norm
from jaxopt.tree_util import tree_sub
from jaxopt.tree_util import tree_vdot


class ProxGradState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  stepsize: float
  error: float


class AccelProxGradState(NamedTuple):
  """Named tuple containing state information for the accelerated case."""
  iter_num: int
  y: Any
  t: float
  stepsize: float
  error: float


@dataclass
class ProximalGradient:
  """Proximal gradient solver.

  Attributes:
    fun: a smooth function of the form ``fun(x, hyperparams_fun)``.
    prox: proximity operator associated with the function g.
    stepsize: a stepsize to use (if <= 0, use backtracking line search).
    maxiter: maximum number of proximal gradient descent iterations.
    maxls: maximum number of iterations to use in the line search.
    tol: tolerance to use.
    acceleration: whether to use acceleration (also known as FISTA) or not.
    stepfactor: factor by which to reduce the stepsize during line search.
    verbose: whether to print error on every iteration or not.
      Warning: verbose=True will automatically disable jit.
    implicit_diff: if True, enable implicit differentiation using cg,
      if Callable, do implicit differentiation using callable as linear solver,
      if False, use autodiff through the solver implementation (note:
        this will unroll syntactic loops).
    has_aux: whether function fun outputs one (False) or more values (True).
      When True it will be assumed by default that fun(...)[0] is the objective.

  References:
    Beck, Amir, and Marc Teboulle. "A fast iterative shrinkage-thresholding
    algorithm for linear inverse problems." SIAM imaging sciences (2009)

    Nesterov, Yu. "Gradient methods for minimizing composite functions."
    Mathematical Programming (2013).
  """
  fun: Callable
  prox: Callable = prox_none
  stepsize: float = 0.0
  maxiter: int = 500
  maxls: int = 15
  tol: float = 1e-3
  acceleration: bool = True
  stepfactor: float = 0.5
  verbose: int = 0
  implicit_diff: Union[bool, Callable] = True
  has_aux: bool = False

  def init(self, init_params: Any) -> Tuple[Any, NamedTuple]:
    """Initialize the ``(params, state)`` pair.

    Args:
      init_params: pytree containing the initial parameters.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    if self.acceleration:
      state = AccelProxGradState(iter_num=0.0,
                                 y=init_params,
                                 t=1.0,
                                 stepsize=1.0,
                                 error=jnp.inf)
    else:
      state = ProxGradState(iter_num=0.0,
                            stepsize=1.0,
                            error=jnp.inf)

    return base.OptStep(params=init_params, state=state)

  def _error(self, x, x_fun_grad, hyperparams_prox):
    next_x = self._prox_grad(x, x_fun_grad, 1.0, hyperparams_prox)
    diff_x = tree_sub(next_x, x)
    return tree_l2_norm(diff_x)

  def _prox_grad(self, x, x_fun_grad, stepsize, hyperparams_prox):
    update = tree_add_scalar_mul(x, -stepsize, x_fun_grad)
    return self.prox(update, hyperparams_prox, stepsize)

  def _ls(self, x, x_fun_val, x_fun_grad, stepsize, hyperparams, data):
    hyperparams_fun, hyperparams_prox = hyperparams

    # epsilon of current dtype for robust checking of
    # sufficient decrease condition
    eps = jnp.finfo(x_fun_val.dtype).eps

    def cond_fun(args):
      next_x, stepsize = args
      diff_x = tree_sub(next_x, x)
      sqdist = tree_l2_norm(diff_x, squared=True)
      # The expression below checks the sufficient decrease condition
      # f(next_x) < f(x) + dot(grad_f(x), diff_x) + (0.5/stepsize) ||diff_x||^2
      # where the terms have been reordered for numerical stability.
      fun_decrease = stepsize * (self.fun(next_x, hyperparams_fun, data)
                                 - x_fun_val)
      condition = stepsize * tree_vdot(diff_x, x_fun_grad) + 0.5 * sqdist
      return fun_decrease > condition + eps

    def body_fun(args):
      stepsize = args[1]
      next_stepsize = stepsize * self.stepfactor
      next_x = self._prox_grad(x, x_fun_grad, next_stepsize, hyperparams_prox)
      return next_x, next_stepsize

    init_x = self._prox_grad(x, x_fun_grad, stepsize, hyperparams_prox)
    init_val = (init_x, stepsize)

    return loop.while_loop(cond_fun=cond_fun, body_fun=body_fun,
                           init_val=init_val, maxiter=self.maxls,
                           unroll=self.unroll, jit=True)

  def _iter(self, x, x_fun_val, x_fun_grad, stepsize, hyperparams, data):
    if self.stepsize <= 0:
      # With line search.
      next_x, next_stepsize = self._ls(x, x_fun_val, x_fun_grad, stepsize,
                                       hyperparams, data)

      # If step size becomes too small, we restart it to 1.0.
      # Otherwise, we attempt to increase it.
      next_stepsize = jnp.where(next_stepsize <= 1e-6, 1.0,
                                next_stepsize / self.stepfactor)

      return next_x, next_stepsize
    else:
      # Without line search.
      next_x = self._prox_grad(x, x_fun_grad, self.stepsize, hyperparams[1])
      return next_x, self.stepsize

  def _update(self, x, state, hyperparams, data):
    iter_num, stepsize, _ = state
    hyperparams_fun, hyperparams_prox = hyperparams
    x_fun_val, x_fun_grad = self.value_and_grad_fun(x, hyperparams_fun, data)
    next_x, next_stepsize = self._iter(x, x_fun_val, x_fun_grad, stepsize,
                                       hyperparams, data)
    error = self._error(x, x_fun_grad, hyperparams_prox)
    next_state = ProxGradState(iter_num=iter_num + 1, stepsize=next_stepsize,
                               error=error)
    return base.OptStep(params=next_x, state=next_state)

  def _update_accel(self, x, state, hyperparams, data):
    iter_num, y, t, stepsize, _ = state
    hyperparams_fun, hyperparams_prox = hyperparams
    y_fun_val, y_fun_grad = self.value_and_grad_fun(y, hyperparams_fun, data)
    next_x, next_stepsize = self._iter(y, y_fun_val, y_fun_grad, stepsize,
                                       hyperparams, data)
    next_t = 0.5 * (1 + jnp.sqrt(1 + 4 * t ** 2))
    diff_x = tree_sub(next_x, x)
    next_y = tree_add_scalar_mul(next_x, (t - 1) / next_t, diff_x)
    next_x_fun_grad = self.grad_fun(next_x, hyperparams_fun, data)
    next_error = self._error(next_x, next_x_fun_grad, hyperparams_prox)
    next_state = AccelProxGradState(iter_num=iter_num + 1, y=next_y, t=next_t,
                                    stepsize=next_stepsize, error=next_error)
    return base.OptStep(params=next_x, state=next_state)

  def update(self,
             params: Any,
             state: NamedTuple,
             hyperparams: Optional[Any] = None,
             data: Optional[Any] = None) -> Tuple[Any, NamedTuple]:
    """Performs one iteration of proximal gradient.

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

    if self.acceleration:
      return self._update_accel(params, state, hyperparams, data)
    else:
      return self._update(params, state, hyperparams, data)

  def run(self,
          hyperparams: Optional[Any] = None,
          data: Optional[Any] = None,
          init_params: Any = None) -> Tuple[Any, NamedTuple]:
    """Runs proximal gradient until convergence or max number of iterations.

    Args:
      hyperparams: tuple ``(hyperparams_fun, hyperparams_prox)`` containing the
        hyper-parameters (i.e., differentiable arguments) of ``fun`` and
        ``prox``.
      data: pytree containing data, i.e., differentiable arguments to be passed
        to ``fun``.
      init_params: pytree containing the initial parameters.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    def cond_fun(pair):
      _, state = pair
      if self.verbose:
        print(state.iter_num, state.error)
      return state.error > self.tol

    def body_fun(pair):
      params, state = pair
      return self.update(params, state, hyperparams, data)

    if init_params is None:
      raise ValueError("`init_params` cannot be None.")

    return loop.while_loop(cond_fun=cond_fun, body_fun=body_fun,
                           init_val=self.init(init_params),
                           maxiter=self.maxiter, jit=self.jit,
                           unroll=self.unroll)

  def _fixed_point_fun(self, sol, hyperparams, data):
    hyperparams_fun, hyperparams_prox = hyperparams
    step = tree_sub(sol, self.grad_fun(sol, hyperparams_fun, data))
    return self.prox(step, hyperparams_prox, 1.0)

  def optimality_fun(self, sol, hyperparams, data):
    """Optimality function mapping compatible with ``@custom_root``."""
    return tree_sub(self._fixed_point_fun(sol, hyperparams, data), sol)

  def __post_init__(self):
    if self.has_aux:
      self.fun = jax.jit(lambda x, par: self.fun(x, par)[0])
    else:
      self.fun = jax.jit(self.fun)
    self.value_and_grad_fun = jax.jit(jax.value_and_grad(self.fun))
    self.grad_fun = jax.jit(jax.grad(self.fun))
    # We always jit unless verbose mode is enabled.
    self.jit = not self.verbose
    # We unroll when implicit diff is disabled or when jit is disabled.
    self.unroll = not self.implicit_diff or not self.jit

    if self.implicit_diff:
      if isinstance(self.implicit_diff, Callable):
        solve = self.implicit_diff
      else:
        solve = linear_solve.solve_normal_cg

      decorator = idf.custom_root(self.optimality_fun,
                                  has_aux=True,
                                  solve=solve)
      self.run = decorator(self.run)
