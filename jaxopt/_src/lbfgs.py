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

"""Limited-memory BFGS"""

from functools import partial

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Union

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt._src.backtracking_linesearch import BacktrackingLineSearch
from jaxopt._src.hager_zhang_linesearch import HagerZhangLineSearch
from jaxopt._src.zoom_linesearch import zoom_linesearch
from jaxopt.tree_util import tree_map
from jaxopt.tree_util import tree_vdot
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_sub
from jaxopt.tree_util import tree_sum
from jaxopt.tree_util import tree_l2_norm
from jaxopt._src.tree_util import tree_single_dtype


def inv_hessian_product_leaf(v: jnp.ndarray,
                             s_history: jnp.ndarray,
                             y_history: jnp.ndarray,
                             rho_history: jnp.ndarray,
                             gamma: float = 1.0,
                             start: int = 0):

  history_size = len(s_history)

  indices = (start + jnp.arange(history_size)) % history_size

  def body_right(r, i):
    alpha = rho_history[i] * jnp.vdot(s_history[i], r)
    r = r - alpha * y_history[i]
    return r, alpha

  r, alpha = jax.lax.scan(body_right, v, indices, reverse=True)

  r = r * gamma

  def body_left(r, args):
    i, alpha = args
    beta = rho_history[i] * jnp.vdot(y_history[i], r)
    r = r + s_history[i] * (alpha - beta)
    return r, beta

  r, beta = jax.lax.scan(body_left, r, (indices, alpha))

  return r


def inv_hessian_product(pytree: Any,
                        s_history: Any,
                        y_history: Any,
                        rho_history: jnp.ndarray,
                        gamma: float = 1.0,
                        start: int = 0):
  """Product between an approximate Hessian inverse and a pytree.

  Histories are pytrees of the same structure as `pytree`.
  Leaves are arrays of shape `(history_size, ...)`, where
  `...` means the same shape as `pytree`'s leaves.

  The notation follows the reference below.

  Args:
    pytree: pytree to multiply with.
    s_history: pytree with the same structure as `pytree`.
      Leaves contain parameter residuals, i.e., `s[k] = x[k+1] - x[k]`.
    y_history: pytree with the same structure as `pytree`.
      Leaves contain gradient residuals, i.e., `y[k] = g[k+1] - g[k]`.
    rho_history: array containing `rho[k] = 1. / vdot(s[k], y[k])`.
    gamma: scalar to use for the initial inverse Hessian approximation,
      i.e., `gamma * I`.
    start: starting index in the circular buffer.

  Reference:
    Jorge Nocedal and Stephen Wright.
    Numerical Optimization, second edition.
    Algorithm 7.4 (page 178).
  """
  fun = partial(inv_hessian_product_leaf,
                rho_history=rho_history,
                gamma=gamma,
                start=start)
  return tree_map(fun, pytree, s_history, y_history)


def compute_gamma(s_history: Any, y_history: Any, last: int):
  # Let gamma = vdot(y_history[last], s_history[last]) / sqnorm(y_history[last]).
  # The initial inverse Hessian approximation can be set to gamma * I.
  # See Numerical Optimization, second edition, equation (7.20).
  # Note that unlike BFGS, the initialization can change on every iteration.

  fun = lambda s_history, y_history: tree_vdot(y_history[last], s_history[last])
  num = tree_sum(tree_map(fun, s_history, y_history))

  fun = lambda y_history: tree_vdot(y_history[last], y_history[last])
  denom = tree_sum(tree_map(fun, y_history))

  return jnp.where(denom > 0, num / denom, 1.0)


def init_history(pytree, history_size):
  fun = lambda leaf: jnp.zeros((history_size,) + leaf.shape, dtype=leaf.dtype)
  return tree_map(fun, pytree)


def update_history(history_pytree, new_pytree, last):
  fun = lambda history_array, new_value: history_array.at[last].set(new_value)
  return tree_map(fun, history_pytree, new_pytree)


class LbfgsState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  value: float
  grad: Any
  stepsize: float
  error: float
  s_history: Any
  y_history: Any
  rho_history: jnp.ndarray
  gamma: jnp.ndarray
  aux: Optional[Any] = None
  failed_linesearch: bool = False


@dataclass(eq=False)
class LBFGS(base.IterativeSolver):
  """LBFGS solver.

  Attributes:
    fun: a smooth function of the form ``fun(x, *args, **kwargs)``.
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

    maxiter: maximum number of proximal gradient descent iterations.
    tol: tolerance of the stopping criterion.

    stepsize: a stepsize to use (if <= 0, use backtracking line search),
      or a callable specifying the **positive** stepsize to use at each iteration.
    linesearch: the type of line search to use: "backtracking" for backtracking
      line search, "zoom" for zoom line search or "hager-zhang" for Hager-Zhang
      line search.
    stop_if_linesearch_fails: whether to stop iterations if the line search fails.
      When True, this matches the behavior of core JAX.
    maxls: maximum number of iterations to use in the line search.
    decrease_factor: factor by which to decrease the stepsize during line search
      (default: 0.8).
    increase_factor: factor by which to increase the stepsize during line search
      (default: 1.5).
    max_stepsize: upper bound on stepsize.
    min_stepsize: lower bound on stepsize.

    history_size: size of the memory to use.
    use_gamma: whether to initialize the inverse Hessian approximation with
      gamma * I, where gamma is chosen following equation (7.20) of 'Numerical
      Optimization' (reference below). If use_gamma is set to False, the
      identity is used as initialization.

    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.

    jit: whether to JIT-compile the optimization loop (default: "auto").
    unroll: whether to unroll the optimization loop (default: "auto").

    verbose: whether to print error on every iteration or not.
      Warning: verbose=True will automatically disable jit.

  Reference:
    Jorge Nocedal and Stephen Wright.
    Numerical Optimization, second edition.
    Algorithm 7.5 (page 179).
  """

  fun: Callable
  value_and_grad: bool = False
  has_aux: bool = False

  maxiter: int = 500
  tol: float = 1e-3

  stepsize: Union[float, Callable] = 0.0
  linesearch: str = "zoom"
  stop_if_linesearch_fails: bool = False
  condition: str = "strong-wolfe"
  maxls: int = 15
  decrease_factor: float = 0.8
  increase_factor: float = 1.5
  max_stepsize: float = 1.0
  # FIXME: should depend on whether float32 or float64 is used.
  min_stepsize: float = 1e-6

  history_size: int = 10
  use_gamma: bool = True

  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None

  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"

  verbose: bool = False

  def _cond_fun(self, inputs):
    _, state = inputs[0]
    if self.verbose:
      print("error:", state.error)
    # We continue the optimization loop while the error tolerance is not met and,
    # either failed linesearch is disallowed or linesearch hasn't failed.
    return (state.error > self.tol) & jnp.logical_or(not self.stop_if_linesearch_fails, ~state.failed_linesearch)

  def init_state(self,
                 init_params: Any,
                 *args,
                 **kwargs) -> LbfgsState:
    """Initialize the solver state.

    Args:
      init_params: pytree containing the initial parameters.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      state
    """
    if isinstance(init_params, base.OptStep):
      # `init_params` can either be a pytree or an OptStep object
      state_kwargs = dict(
        s_history=init_params.state.s_history,
        y_history=init_params.state.y_history,
        rho_history=init_params.state.rho_history,
        gamma=init_params.state.gamma,
        iter_num=init_params.state.iter_num,
        stepsize=init_params.state.stepsize,
      )
      init_params = init_params.params
      dtype = tree_single_dtype(init_params)
    else:
      dtype = tree_single_dtype(init_params)
      state_kwargs = dict(
        s_history=init_history(init_params, self.history_size),
        y_history=init_history(init_params, self.history_size),
        rho_history=jnp.zeros(self.history_size, dtype=dtype),
        gamma=jnp.asarray(1.0, dtype=dtype),
        iter_num=jnp.asarray(0),
        stepsize=jnp.asarray(self.max_stepsize, dtype=dtype),
      )
    (value, aux), grad = self._value_and_grad_with_aux(init_params, *args, **kwargs)
    return LbfgsState(value=value,
                      grad=grad,
                      error=jnp.asarray(jnp.inf),
                      **state_kwargs,
                      aux=aux,
                      failed_linesearch=jnp.asarray(False))

  def update(self,
             params: Any,
             state: LbfgsState,
             *args,
             **kwargs) -> base.OptStep:
    """Performs one iteration of LBFGS.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      (params, state)
    """
    if isinstance(params, base.OptStep):
      params = params.params

    start = state.iter_num % self.history_size
    value, grad = (state.value, state.grad)

    product = inv_hessian_product(pytree=grad, s_history=state.s_history,
                                  y_history=state.y_history,
                                  rho_history=state.rho_history, gamma=state.gamma,
                                  start=start)
    descent_direction = tree_scalar_mul(-1.0, product)

    use_linesearch = not isinstance(self.stepsize, Callable) and self.stepsize <= 0
    if use_linesearch:
      # with line search

      if self.linesearch == "backtracking":
        ls = BacktrackingLineSearch(fun=self._value_and_grad_with_aux,
                                    value_and_grad=True,
                                    maxiter=self.maxls,
                                    decrease_factor=self.decrease_factor,
                                    max_stepsize=self.max_stepsize,
                                    condition=self.condition,
                                    jit=self.jit,
                                    unroll=self.unroll,
                                    has_aux=True)
        init_stepsize = jnp.where(state.stepsize <= self.min_stepsize,
                                  # If stepsize became too small, we restart it.
                                  self.max_stepsize,
                                  # Else, we increase a bit the previous one.
                                  state.stepsize * self.increase_factor)
        new_stepsize, ls_state = ls.run(init_stepsize,
                                        params, value, grad,
                                        descent_direction,
                                        *args, **kwargs)
        new_value = ls_state.value
        new_aux = ls_state.aux
        new_params = ls_state.params
        new_grad = ls_state.grad

      elif self.linesearch == "zoom":
        ls_state = zoom_linesearch(f=self._value_and_grad_with_aux,
                                   xk=params, pk=descent_direction,
                                   old_fval=value, gfk=grad, maxiter=self.maxls,
                                   value_and_grad=True, has_aux=True, aux=state.aux,
                                   args=args, kwargs=kwargs)
        new_value = ls_state.f_k
        new_aux = ls_state.aux
        new_stepsize = ls_state.a_k
        new_grad = ls_state.g_k
        # FIXME: zoom_linesearch currently doesn't return new_params
        # so we have to recompute it.
        t = new_stepsize.astype(tree_single_dtype(params))
        new_params = tree_add_scalar_mul(params, t, descent_direction)
        # FIXME: (zaccharieramzi) sometimes the linesearch fails
        # and therefore its value g_k does not correspond
        # to the gradient at the new parameters.
        # with the following conditional loop we have a hot fix that just
        # recomputes the value, gradient and auxiliary value
        # at the new parameters. It would be better to understand
        # what the g_k passed by zoom_linesearch is in this case
        # and why it is wrong.
        (new_value, new_aux), new_grad = jax.lax.cond(
          ls_state.failed,
          lambda: self._value_and_grad_with_aux(new_params, *args, **kwargs),
          lambda: ((new_value, new_aux), new_grad),
        )
      elif self.linesearch == "hager-zhang":
        # By default Hager-Zhang uses the Wolfe Conditions & Approximate Wolfe
        # Conditions.
        ls = HagerZhangLineSearch(fun=self._value_and_grad_fun,
                                  value_and_grad=True,
                                  maxiter=self.maxls,
                                  max_stepsize=self.max_stepsize,
                                  jit=self.jit,
                                  unroll=self.unroll)
        # Note that HZL doesn't use the previous step size.
        new_stepsize, ls_state = ls.run(self.max_stepsize,
                                        params, value, grad,
                                        descent_direction,
                                        *args, **kwargs)
        new_params = ls_state.params
        new_value = ls_state.value
        new_grad = ls_state.grad
        new_aux = ls_state.aux
      else:
        raise ValueError("Invalid name in 'linesearch' option.")

    else:
      # without line search
      if isinstance(self.stepsize, Callable):
        new_stepsize = self.stepsize(state.iter_num)
      else:
        new_stepsize = self.stepsize

      new_params = tree_add_scalar_mul(params, new_stepsize, descent_direction)
      (new_value, new_aux), new_grad = self._value_and_grad_with_aux(new_params, *args, **kwargs)
    s = tree_sub(new_params, params)
    y = tree_sub(new_grad, grad)
    vdot_sy = tree_vdot(s, y)
    rho = jnp.where(vdot_sy == 0, 0, 1. / vdot_sy)

    last = (start + self.history_size) % self.history_size
    s_history = update_history(state.s_history, s, last)
    y_history = update_history(state.y_history, y, last)
    rho_history = update_history(state.rho_history, rho, last)

    if self.use_gamma:
      gamma = compute_gamma(s_history, y_history, last)
    else:
      gamma = jnp.array(1.0)

    if use_linesearch and self.linesearch == "zoom":
      failed_linesearch = ls_state.failed
    else:  # backtracking linesearch doesn't support failed state yet
      failed_linesearch = jnp.asarray(False)

    new_state = LbfgsState(iter_num=state.iter_num + 1,
                           value=new_value,
                           grad=new_grad,
                           stepsize=jnp.asarray(new_stepsize),
                           error=tree_l2_norm(new_grad),
                           s_history=s_history,
                           y_history=y_history,
                           rho_history=rho_history,
                           gamma=gamma,
                           aux=new_aux,
                           failed_linesearch=failed_linesearch)

    return base.OptStep(params=new_params, state=new_state)

  def optimality_fun(self, params, *args, **kwargs):
    """Optimality function mapping compatible with ``@custom_root``."""
    return self._value_and_grad_fun(params, *args, **kwargs)[1]

  def _value_and_grad_fun(self,
                          params,
                          *args,
                          **kwargs):
    if isinstance(params, base.OptStep):
      params = params.params
    (value, aux), grad = self._value_and_grad_with_aux(params, *args, **kwargs)
    return value, grad

  def __post_init__(self):
    _, _, self._value_and_grad_with_aux = \
      base._make_funs_with_aux(fun=self.fun,
                               value_and_grad=self.value_and_grad,
                               has_aux=self.has_aux)

    self.reference_signature = self.fun
