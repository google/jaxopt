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

from functools import partial
import inspect

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Union

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt._src import loop
from jaxopt._src.prox import prox_none
from jaxopt._src.tree_util import tree_add_scalar_mul
from jaxopt._src.tree_util import tree_l2_norm
from jaxopt._src.tree_util import tree_sub
from jaxopt._src.tree_util import tree_vdot
from jaxopt._src.tree_util import tree_single_dtype


def fista_line_search(
  fun,
  prox_grad,
  jit,
  unroll,
  maxls,
  x,
  x_fun_val,
  x_fun_grad,
  stepsize,
  decrease_factor,
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
    fun_decrease = stepsize * (fun(next_x, *args, **kwargs) - x_fun_val)
    condition = stepsize * tree_vdot(diff_x, x_fun_grad) + 0.5 * sqdist
    return fun_decrease > condition + eps

  def body_fun(pair):
    stepsize = pair[1]
    next_stepsize = stepsize * decrease_factor
    next_x = prox_grad(x, x_fun_grad, next_stepsize, hyperparams_prox)
    return next_x, next_stepsize

  init_x = prox_grad(x, x_fun_grad, stepsize, hyperparams_prox)
  init_val = (init_x, stepsize)

  return loop.while_loop(cond_fun=cond_fun, body_fun=body_fun,
                         init_val=init_val, maxiter=maxls,
                         unroll=unroll, jit=jit)


class ProxGradState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  stepsize: float
  error: float
  aux: Optional[Any] = None
  velocity: Optional[Any] = None
  t: float = 1.0


@dataclass(eq=False)
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

    stepsize: a stepsize to use (if <= 0, use backtracking line search),
      or a callable specifying the **positive** stepsize to use at each iteration.
    maxiter: maximum number of proximal gradient descent iterations.
    maxls: maximum number of iterations to use in the line search.
    tol: tolerance to use.
    acceleration: whether to use acceleration (also known as FISTA) or not.
    decrease_factor: factor by which to reduce the stepsize during line search.
    verbose: whether to print information on every iteration or not.

    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.

    jit: whether to JIT-compile the optimization loop (default: True).
    unroll: whether to unroll the optimization loop (default: "auto").

  References:
    Beck, Amir, and Marc Teboulle. "A fast iterative shrinkage-thresholding
    algorithm for linear inverse problems." SIAM imaging sciences (2009)

    Nesterov, Yu. "Gradient methods for minimizing composite functions."
    Mathematical Programming (2013).
  """
  fun: Callable
  prox: Callable = prox_none
  value_and_grad: bool = False
  has_aux: bool = False

  stepsize: Union[float, Callable] = 0.0
  maxiter: int = 500
  maxls: int = 15
  tol: float = 1e-3
  acceleration: bool = True
  decrease_factor: float = 0.5
  verbose: Union[bool, int] = False

  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None

  jit: bool = True
  unroll: base.AutoOrBoolean = "auto"

  def init_state(self,
                 init_params: Any,
                 hyperparams_prox: Any,
                 *args,
                 **kwargs) -> ProxGradState:
    """Initialize the solver state.

    Args:
      init_params: pytree containing the initial parameters.
      hyperparams_prox: pytree containing hyperparameters of prox.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      state
    """
    del hyperparams_prox  # Not used.

    if self.has_aux:
      _, aux = self.fun(init_params, *args, **kwargs)
    else:
      aux = None

    dtype = tree_single_dtype(init_params)

    if self.acceleration:
      state = ProxGradState(iter_num=jnp.asarray(0),
                            velocity=init_params,
                            t=jnp.asarray(1.0),
                            stepsize=jnp.asarray(1.0, dtype=dtype),
                            error=jnp.asarray(jnp.inf, dtype=dtype),
                            aux=aux)
    else:
      state = ProxGradState(iter_num=jnp.asarray(0),
                            t=jnp.asarray(1.0),
                            stepsize=jnp.asarray(1.0, dtype=dtype),
                            error=jnp.asarray(jnp.inf, dtype=dtype),
                            aux=aux)

    return state

  def _error(self, diff_x, stepsize):
    diff_norm = tree_l2_norm(diff_x)
    return diff_norm / stepsize

  def _prox_grad(self, x, x_fun_grad, stepsize, hyperparams_prox):
    update = tree_add_scalar_mul(x, -stepsize, x_fun_grad)
    return self.prox(update, hyperparams_prox, stepsize)

  def _iter(self,
            iter_num,
            x,
            x_fun_val,
            x_fun_grad,
            stepsize,
            hyperparams_prox,
            args,
            kwargs):

    if not isinstance(self.stepsize, Callable) and self.stepsize <= 0:
      # with line search
      next_x, next_stepsize = self._fista_line_search(self.maxls, x, x_fun_val,
                                                      x_fun_grad, stepsize,
                                                      self.decrease_factor,
                                                      hyperparams_prox, args,
                                                      kwargs)
      # If step size becomes too small, we restart it to 1.0.
      # Otherwise, we attempt to increase it.
      next_stepsize = jnp.where(next_stepsize <= 1e-6, 1.0,
                                next_stepsize / self.decrease_factor)
      return next_x, next_stepsize
    else:
      # without line search
      if isinstance(self.stepsize, Callable):
        next_stepsize = self.stepsize(iter_num)
      else:
        next_stepsize = self.stepsize
      next_x = self._prox_grad(x, x_fun_grad, next_stepsize, hyperparams_prox)
      return next_x, next_stepsize

  def _update(self, x, state, hyperparams_prox, args, kwargs):
    dtype = tree_single_dtype(x)
    iter_num = state.iter_num
    stepsize = state.stepsize
    (x_fun_val, aux), x_fun_grad = self._value_and_grad_with_aux(x, *args,
                                                                 **kwargs)
    next_x, next_stepsize = self._iter(iter_num, x, x_fun_val, x_fun_grad,
                                       stepsize, hyperparams_prox, args, kwargs)
    error = self._error(tree_sub(next_x, x), next_stepsize)
    next_state = ProxGradState(iter_num=iter_num + 1,
                               stepsize=jnp.asarray(next_stepsize, dtype=dtype),
                               error=jnp.asarray(error, dtype=dtype),
                               aux=aux)
    return base.OptStep(params=next_x, state=next_state)

  def _update_accel(self, x, state, hyperparams_prox, args, kwargs):
    dtype = tree_single_dtype(x)
    iter_num = state.iter_num
    y = state.velocity
    t = state.t
    stepsize = state.stepsize
    (y_fun_val, aux), y_fun_grad = self._value_and_grad_with_aux(y, *args,
                                                                 **kwargs)
    next_x, next_stepsize = self._iter(iter_num, y, y_fun_val, y_fun_grad,
                                       stepsize, hyperparams_prox, args, kwargs)
    next_t = 0.5 * (1 + jnp.sqrt(1 + 4 * t ** 2))
    diff_x = tree_sub(next_x, x)
    next_y = tree_add_scalar_mul(next_x, (t - 1) / next_t, diff_x)
    next_error = self._error(diff_x, next_stepsize)
    next_state = ProxGradState(iter_num=iter_num + 1, velocity=next_y, t=next_t,
                               stepsize=jnp.asarray(next_stepsize, dtype=dtype),
                               error=jnp.asarray(next_error, dtype=dtype),
                               aux=aux)

    if self.verbose:
      self.log_info(
          next_state,
          error_name="Distance btw Iterates",
          additional_info={
            "Stepsize": next_stepsize
          }
      )
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

  def _value_and_grad_fun(self, params, *args, **kwargs):
    (value, aux), grad = self._value_and_grad_with_aux(params, *args, **kwargs)
    return value, grad

  def _grad_fun(self, params, *args, **kwargs):
    return self._value_and_grad_fun(params, *args, **kwargs)[1]

  def _grad_with_aux(self, params, *args, **kwargs):
    (value, aux), grad = self._value_and_grad_with_aux(params, *args, **kwargs)
    return grad, aux

  def __post_init__(self):
    super().__post_init__()

    fun_with_aux, _, self._value_and_grad_with_aux = \
      base._make_funs_with_aux(fun=self.fun,
                               value_and_grad=self.value_and_grad,
                               has_aux=self.has_aux)
    fun_without_aux = lambda *a, **kw: fun_with_aux(*a, **kw)[0]

    # Sets up reference signature.
    fun = getattr(self.fun, "subfun", self.fun)
    signature = inspect.signature(fun)
    parameters = list(signature.parameters.values())
    new_param = inspect.Parameter(name="hyperparams_prox",
                                  kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
    parameters.insert(1, new_param)
    self.reference_signature = inspect.Signature(parameters)

    unroll = self._get_unroll_option()

    fista_ls_with_fun= partial(fista_line_search, fun_without_aux,
                               self._prox_grad, self.jit, unroll)

    if self.jit:
      jitted_fista_ls_with_fun = jax.jit(fista_ls_with_fun, static_argnums=(0,))
      self._fista_line_search = jitted_fista_ls_with_fun
    else:
      self._fista_line_search = fista_ls_with_fun
