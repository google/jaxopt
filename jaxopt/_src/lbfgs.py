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
from jaxopt.tree_util import tree_map
from jaxopt.tree_util import tree_vdot
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_sub
from jaxopt.tree_util import tree_sum
from jaxopt.tree_util import tree_l2_norm


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
  fun = lambda leaf: jnp.zeros((history_size,) + leaf.shape)
  return tree_map(fun, pytree)


def update_history(history_pytree, new_pytree, last):
  fun = lambda history_array, new_value: history_array.at[last].set(new_value)
  return tree_map(fun, history_pytree, new_pytree)


class LbfgsState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  value: float
  stepsize: float
  error: float
  s_history: Any
  y_history: Any
  rho_history: jnp.ndarray
  aux: Optional[Any] = None


@dataclass(eq=False)
class LBFGS(base.IterativeSolver):
  """LBFGS solver.

  Attributes:
    fun: a smooth function of the form ``fun(x, *args, **kwargs)``.
    has_aux: whether function fun outputs one (False) or more values (True).
      When True it will be assumed by default that fun(...)[0] is the objective.

    maxiter: maximum number of proximal gradient descent iterations.
    tol: tolerance of the stopping criterion.

    maxls: maximum number of iterations to use in the line search.
    decrease_factor: factor by which to decrease the stepsize during line search
      (default: 0.8).
    increase_factor: factor by which to increase the stepsize during line search
      (default: 1.5).

    history_size: size of the memory to use.
    use_gamma: whether to initialize the inverse Hessian approximation with
      gamma * I, see 'Numerical Optimization', equation (7.20).

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
  has_aux: bool = False

  maxiter: int = 500
  tol: float = 1e-3

  condition: str = "strong-wolfe"
  maxls: int = 15
  decrease_factor: float = 0.8
  increase_factor: float = 1.5

  history_size: int = 10
  use_gamma: bool = True

  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None

  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"

  verbose: bool = False

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
    return LbfgsState(iter_num=jnp.asarray(0),
                      value=jnp.asarray(jnp.inf),
                      stepsize=jnp.asarray(1.0),
                      error=jnp.asarray(jnp.inf),
                      s_history=init_history(init_params, self.history_size),
                      y_history=init_history(init_params, self.history_size),
                      rho_history=jnp.zeros(self.history_size))

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
    (value, aux), grad = self._value_and_grad_with_aux(params, *args, **kwargs)

    start = state.iter_num % self.history_size
    last = (start + self.history_size) % self.history_size

    if self.use_gamma:
      gamma = compute_gamma(state.s_history, state.y_history, last)
    else:
      gamma = 1.0

    product = inv_hessian_product(pytree=grad, s_history=state.s_history,
                                  y_history=state.y_history,
                                  rho_history=state.rho_history, gamma=gamma,
                                  start=start)
    descent_direction = tree_scalar_mul(-1, product)

    ls = BacktrackingLineSearch(fun=self._value_and_grad_fun,
                                value_and_grad=True,
                                maxiter=self.maxls,
                                decrease_factor=self.decrease_factor,
                                condition=self.condition,
                                jit=self.jit,
                                unroll=self.unroll)
    init_stepsize = state.stepsize * self.increase_factor
    new_stepsize, ls_state = ls.run(init_stepsize=init_stepsize,
                                    params=params, value=value, grad=grad,
                                    descent_direction=descent_direction,
                                    *args, **kwargs)
    new_value = ls_state.value
    new_params = ls_state.params
    new_grad = ls_state.grad

    s = tree_sub(new_params, params)
    y = tree_sub(new_grad, grad)
    vdot_sy = tree_vdot(s, y)
    rho = jnp.where(vdot_sy == 0, 0, 1. / vdot_sy)

    s_history = update_history(state.s_history, s, last)
    y_history = update_history(state.y_history, y, last)
    rho_history = update_history(state.rho_history, rho, last)

    new_state = LbfgsState(iter_num=state.iter_num + 1,
                           value=new_value,
                           stepsize=jnp.asarray(new_stepsize),
                           error=tree_l2_norm(new_grad),
                           s_history=s_history,
                           y_history=y_history,
                           rho_history=rho_history,
                           # FIXME: we should return new_aux here but
                           # BacktrackingLineSearch currently doesn't support
                           # an aux.
                           aux=aux)

    return base.OptStep(params=new_params, state=new_state)

  def optimality_fun(self, params, *args, **kwargs):
    """Optimality function mapping compatible with ``@custom_root``."""
    return self._grad_fun(params, *args, **kwargs)

  def _value_and_grad_fun(self, params, *args, **kwargs):
    (value, aux), grad = self._value_and_grad_with_aux(params, *args, **kwargs)
    return value, grad

  def _grad_fun(self, params, *args, **kwargs):
    return self._value_and_grad_fun(params, *args, **kwargs)[1]

  def __post_init__(self):
    if self.has_aux:
      self._fun = lambda *a, **kw: self.fun(*a, **kw)[0]
      fun_with_aux = self.fun
    else:
      self._fun = self.fun
      fun_with_aux = lambda *a, **kw: (self.fun(*a, **kw), None)

    self._value_and_grad_with_aux = jax.value_and_grad(fun_with_aux,
                                                       has_aux=True)

    self.reference_signature = self.fun
