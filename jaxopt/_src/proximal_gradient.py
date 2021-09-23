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

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt._src import linear_solve
from jaxopt._src import loop
from jaxopt._src.prox import prox_none
from jaxopt._src.tree_util import tree_add_scalar_mul
from jaxopt._src.tree_util import tree_l2_norm
from jaxopt._src.tree_util import tree_sub
from jaxopt._src.tree_util import tree_vdot


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
class ProximalGradient(base.IterativeSolver):
  """Proximal gradient solver.

  This solver minimizes::

    objective(params, hyperparams_prox, *args, **kwargs) =
      fun(params, *args, **kwargs) + non_smooth(params, hyperparams_prox)

  Attributes:
    fun: a smooth function of the form ``fun(x, *args, **kwargs)``.
    prox: proximity operator associated with the function ``non_smooth``.
      It should be of the form ``prox(params, hyperparams_prox, scale=1.0)``.
      See ``jaxopt.prox`` for examples.
    stepsize: a stepsize to use (if <= 0, use backtracking line search).
    maxiter: maximum number of proximal gradient descent iterations.
    maxls: maximum number of iterations to use in the line search.
    tol: tolerance to use.
    acceleration: whether to use acceleration (also known as FISTA) or not.
    stepfactor: factor by which to reduce the stepsize during line search.
    verbose: whether to print error on every iteration or not.
      Warning: verbose=True will automatically disable jit.
    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.
    has_aux: whether function fun outputs one (False) or more values (True).
      When True it will be assumed by default that fun(...)[0] is the objective.
    jit: whether to JIT-compile the optimization loop (default: "auto").
    unroll: whether to unroll the optimization loop (default: "auto").

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
  implicit_diff: bool = True
  implicit_diff_solve: Callable = linear_solve.solve_normal_cg
  has_aux: bool = False
  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"

  def init(self,
           init_params: Any,
           hyperparams_prox: Any,
           *args,
           **kwargs) -> base.OptStep:
    """Initialize the ``(params, state)`` pair.

    Args:
      init_params: pytree containing the initial parameters.
      hyperparams_prox: pytree containing hyperparameters of prox.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    del hyperparams_prox, args, kwargs  # Not used.

    if self.acceleration:
      state = AccelProxGradState(iter_num=0,
                                 y=init_params,
                                 t=1.0,
                                 stepsize=1.0,
                                 error=jnp.inf)
    else:
      state = ProxGradState(iter_num=0,
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

  def _ls(self,
          x,
          x_fun_val,
          x_fun_grad,
          stepsize,
          hyperparams_prox,
          args,
          kwargs):
    # epsilon of current dtype for robust checking of
    # sufficient decrease condition
    eps = jnp.finfo(x_fun_val.dtype).eps

    def cond_fun(pair):
      next_x, stepsize = pair
      diff_x = tree_sub(next_x, x)
      sqdist = tree_l2_norm(diff_x, squared=True)
      # The expression below checks the sufficient decrease condition
      # f(next_x) < f(x) + dot(grad_f(x), diff_x) + (0.5/stepsize) ||diff_x||^2
      # where the terms have been reordered for numerical stability.
      fun_decrease = stepsize * (self._fun(next_x, *args, **kwargs) - x_fun_val)
      condition = stepsize * tree_vdot(diff_x, x_fun_grad) + 0.5 * sqdist
      return fun_decrease > condition + eps

    def body_fun(pair):
      stepsize = pair[1]
      next_stepsize = stepsize * self.stepfactor
      next_x = self._prox_grad(x, x_fun_grad, next_stepsize, hyperparams_prox)
      return next_x, next_stepsize

    init_x = self._prox_grad(x, x_fun_grad, stepsize, hyperparams_prox)
    init_val = (init_x, stepsize)

    # We unroll when implicit diff is disabled or when verbose mode is enabled.
    unroll = not self.implicit_diff or self.verbose

    return loop.while_loop(cond_fun=cond_fun, body_fun=body_fun,
                           init_val=init_val, maxiter=self.maxls,
                           unroll=unroll, jit=True)

  def _iter(self,
            x,
            x_fun_val,
            x_fun_grad,
            stepsize,
            hyperparams_prox,
            args,
            kwargs):
    if self.stepsize <= 0:
      # With line search.
      next_x, next_stepsize = self._ls(x, x_fun_val, x_fun_grad, stepsize,
                                       hyperparams_prox, args, kwargs)

      # If step size becomes too small, we restart it to 1.0.
      # Otherwise, we attempt to increase it.
      next_stepsize = jnp.where(next_stepsize <= 1e-6, 1.0,
                                next_stepsize / self.stepfactor)

      return next_x, next_stepsize
    else:
      # Without line search.
      next_x = self._prox_grad(x, x_fun_grad, self.stepsize, hyperparams_prox)
      return next_x, self.stepsize

  def _update(self, x, state, hyperparams_prox, args, kwargs):
    iter_num, stepsize, _ = state
    x_fun_val, x_fun_grad = self._value_and_grad_fun(x, *args, **kwargs)
    next_x, next_stepsize = self._iter(x, x_fun_val, x_fun_grad, stepsize,
                                       hyperparams_prox, args, kwargs)
    error = self._error(x, x_fun_grad, hyperparams_prox)
    next_state = ProxGradState(iter_num=iter_num + 1,
                               stepsize=next_stepsize,
                               error=error)
    return base.OptStep(params=next_x, state=next_state)

  def _update_accel(self, x, state, hyperparams_prox, args, kwargs):
    iter_num, y, t, stepsize, _ = state
    y_fun_val, y_fun_grad = self._value_and_grad_fun(y, *args, **kwargs)
    next_x, next_stepsize = self._iter(y, y_fun_val, y_fun_grad, stepsize,
                                       hyperparams_prox, args, kwargs)
    next_t = 0.5 * (1 + jnp.sqrt(1 + 4 * t ** 2))
    diff_x = tree_sub(next_x, x)
    next_y = tree_add_scalar_mul(next_x, (t - 1) / next_t, diff_x)
    next_x_fun_grad = self._grad_fun(next_x, *args, **kwargs)
    next_error = self._error(next_x, next_x_fun_grad, hyperparams_prox)
    next_state = AccelProxGradState(iter_num=iter_num + 1, y=next_y, t=next_t,
                                    stepsize=next_stepsize, error=next_error)
    return base.OptStep(params=next_x, state=next_state)

  def update(self,
             params: Any,
             state: NamedTuple,
             hyperparams_prox: Any,
             *args,
             **kwargs) -> base.OptStep:
    """Performs one iteration of proximal gradient.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      hyperparams_prox: pytree containing hyperparameters of prox.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    f = self._update_accel if self.acceleration else self._update
    return f(params, state, hyperparams_prox, args, kwargs)

  def _fixed_point_fun(self, sol, hyperparams_prox, args, kwargs):
    step = tree_sub(sol, self._grad_fun(sol, *args, **kwargs))
    return self.prox(step, hyperparams_prox, 1.0)

  def optimality_fun(self, sol, hyperparams_prox, *args, **kwargs):
    """Optimality function mapping compatible with ``@custom_root``."""
    fp = self._fixed_point_fun(sol, hyperparams_prox, args, kwargs)
    return tree_sub(fp, sol)

  def __post_init__(self):
    if self.has_aux:
      self._fun = lambda x, par: self.fun(x, par)[0]
    else:
      self._fun = self.fun

    # Pre-compile useful functions.
    self._value_and_grad_fun = jax.value_and_grad(self._fun)
    self._grad_fun = jax.grad(self._fun)
