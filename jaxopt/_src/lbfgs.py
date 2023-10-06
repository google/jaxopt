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

"""Limited-memory BFGS."""

import warnings

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from jaxopt._src import base
from jaxopt._src.linesearch_util import _init_stepsize
from jaxopt._src.linesearch_util import _setup_linesearch
from jaxopt._src.tree_util import tree_single_dtype
from jaxopt._src.tree_util import get_real_dtype
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_conj
from jaxopt.tree_util import tree_l2_norm
from jaxopt.tree_util import tree_map
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_sub
from jaxopt.tree_util import tree_sum
from jaxopt.tree_util import tree_vdot_real


def select_ith_tree(tree_history, i):
  """Select tree corresponding to ith history entry."""
  return tree_map(lambda a: a[i, ...], tree_history)

def inv_hessian_product(pytree: Any,
                        s_history: Any,
                        y_history: Any,
                        rho_history: jnp.ndarray,
                        gamma: float = 1.0,
                        start: int = 0):
  """Product between an approximate Hessian inverse and a pytree.

  Histories are pytrees of the same structure as `pytree` except
  that the leaves are arrays of shape `(history_size, ...)`, where
  `...` means the same shape as `pytree`'s leaves.

  The notation follows the reference below.

  Args:
    pytree: pytree to multiply with.
    s_history: pytree whose leaves contain parameter residuals,
      i.e., `s[k] = x[k+1] - x[k]`
    y_history: pytree whose leaves contain gradient residuals,
      i.e., `y[k] = g[k+1] - g[k]`.
    rho_history: array containing `rho[k] = 1. / vdot(s[k], y[k])`.
    gamma: scalar to use for the initial inverse Hessian approximation,
      i.e., `gamma * I`.
    start: starting index in the circular buffer.

  Returns:
    Product between approximate Hessian inverse and the pytree

  Reference:
    Jorge Nocedal and Stephen Wright.
    Numerical Optimization, second edition.
    Algorithm 7.4 (page 178).
  """

  history_size = rho_history.shape[0]

  indices = (start + jnp.arange(history_size)) % history_size

  def body_right(r, i):
    si, yi = select_ith_tree(s_history, i), select_ith_tree(y_history, i)
    alpha = rho_history[i] * tree_vdot_real(si, r)
    r = tree_add_scalar_mul(r, -alpha, yi)
    return r, alpha
  r, alpha = jax.lax.scan(body_right, pytree, indices, reverse=True)

  r = tree_scalar_mul(gamma, r)

  def body_left(r, args):
    i, alpha = args
    si, yi = select_ith_tree(s_history, i), select_ith_tree(y_history, i)
    beta = rho_history[i] * tree_vdot_real(yi, r)
    r = tree_add_scalar_mul(r, alpha - beta, si)
    return r, beta
  r, _ = jax.lax.scan(body_left, r, (indices, alpha))

  return r


def compute_gamma(s_history: Any, y_history: Any, last: int):
  """Compute scalar gamma defining the initialization of the approximate Hessian."""
  # Let gamma = vdot(y_history[last], s_history[last]) / sqnorm(y_history[last]).
  # The initial inverse Hessian approximation can be set to gamma * I.
  # See Numerical Optimization, second edition, equation (7.20).
  # Note that unlike BFGS, the initialization can change on every iteration.

  fun = lambda s_history, y_history: tree_vdot_real(y_history[last], s_history[last])
  num = tree_sum(tree_map(fun, s_history, y_history))

  fun = lambda y_history: tree_vdot_real(y_history[last], y_history[last])
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
  num_fun_eval: int = 0
  num_grad_eval: int = 0
  num_linesearch_iter: int = 0


@dataclass(eq=False)
class LBFGS(base.IterativeSolver):
  """LBFGS solver.

  Supports complex variables, see second reference.

  Attributes:
    fun: a smooth function of the form ``fun(x, *args, **kwargs)``.
    value_and_grad: whether ``fun`` just returns the value (False) or both the
      value and gradient (True).
    has_aux: whether ``fun`` outputs auxiliary data or not. If ``has_aux`` is
      False, ``fun`` is expected to be scalar-valued. If ``has_aux`` is True,
      then we have one of the following two cases. If ``value_and_grad`` is
      False, the output should be ``value, aux = fun(...)``. If ``value_and_grad
      == True``, the output should be ``(value, aux), grad = fun(...)``. At each
      iteration of the algorithm, the auxiliary outputs are stored in
      ``state.aux``.
    maxiter: maximum number of proximal gradient descent iterations.
    tol: tolerance of the stopping criterion.
    stepsize: a stepsize to use (if <= 0, use backtracking line search), or a
      callable specifying the **positive** stepsize to use at each iteration.
    linesearch: the type of line search to use: "backtracking" for backtracking
      line search, "zoom" for zoom line search or "hager-zhang" for Hager-Zhang
      line search.
    linesearch_init: strategy for line-search initialization. By default, it
      will use "increase", which will increase the step-size by a factor of
      `increase_factor` at each iteration if the step-size is larger than
      `min_stepsize`, and set it to `max_stepsize` otherwise. Other choices are
      "max", that initializes the step-size to `max_stepsize` at every
      iteration, and "current", that uses the step-size from the previous
      iteration.
    stop_if_linesearch_fails: whether to stop iterations if the line search
      fails. When True, this matches the behavior of core JAX.
    condition: Deprecated. Condition used to select the stepsize when using
      backtracking linesearch.
    maxls: maximum number of iterations to use in the line search.
    decrease_factor: Deprecated. Factor by which to decrease the stepsize during
      line search when using backtracking linesearch (default: 0.8).
    increase_factor: factor by which to increase the stepsize during line search
      (default: 1.5).
    max_stepsize: upper bound on stepsize.
    min_stepsize: lower bound on stepsize guess at start of each linesearch run.
    history_size: size of the memory to use.
    use_gamma: whether to initialize the inverse Hessian approximation with
      gamma * I, where gamma is chosen following equation (7.20) of 'Numerical
      Optimization' (reference below). If use_gamma is set to False, the
      identity is used as initialization.
    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.
    jit: whether to JIT-compile the optimization loop (default: True).
    unroll: whether to unroll the optimization loop (default: "auto").
    verbose: if set to True or 1 prints the information at each step of 
      the solver, if set to 2, print also the information of the linesearch.

  References:
    Jorge Nocedal and Stephen Wright.
    Numerical Optimization, second edition.
    Algorithm 7.5 (page 179).

    Laurent Sorber, Marc van Barel, and Lieven de Lathauwer.
    Unconstrained Optimization of Real Functions in Complex Variables.
    SIAM J. Optim., Vol. 22, No. 3, pp. 879-898
  """

  fun: Callable
  value_and_grad: bool = False
  has_aux: bool = False

  maxiter: int = 500
  # FIXME: should depend on whether float32 or float64 is used.
  # Tests should pass in float64 without modifying tol
  tol: float = 1e-3

  stepsize: Union[float, Callable] = 0.0
  linesearch: str = "zoom"
  linesearch_init: str = "increase"
  stop_if_linesearch_fails: bool = False
  condition: Any = None  # deprecated in v0.8
  maxls: int = 30
  decrease_factor: Any = None  # deprecated in v0.8
  increase_factor: float = 1.5
  max_stepsize: float = 1.0
  # FIXME: should depend on whether float32 or float64 is used.
  min_stepsize: float = 1e-6

  history_size: int = 10
  use_gamma: bool = True

  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None

  jit: bool = True
  unroll: base.AutoOrBoolean = "auto"

  verbose: Union[bool, int] = False

  def _cond_fun(self, inputs):
    _, state = inputs[0]
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
      realdtype = get_real_dtype(tree_single_dtype(init_params))
    else:
      realdtype = get_real_dtype(tree_single_dtype(init_params))
      state_kwargs = dict(
          s_history=init_history(init_params, self.history_size),
          y_history=init_history(init_params, self.history_size),
          rho_history=jnp.zeros(self.history_size, dtype=realdtype),
          gamma=jnp.asarray(1.0, dtype=realdtype),
          iter_num=jnp.asarray(0),
          stepsize=jnp.asarray(self.max_stepsize, dtype=realdtype),
      )
    (value, aux), grad = self._value_and_grad_with_aux(init_params, *args, **kwargs)
    return LbfgsState(value=value,
                      grad=grad,
                      error=jnp.asarray(jnp.inf, dtype=realdtype),
                      **state_kwargs,
                      aux=aux,
                      failed_linesearch=jnp.asarray(False),
                      num_grad_eval=jnp.array(1, base.NUM_EVAL_DTYPE),
                      num_fun_eval=jnp.array(1, base.NUM_EVAL_DTYPE),
                      num_linesearch_iter=jnp.array(0, base.NUM_EVAL_DTYPE))

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

    realdtype = state.rho_history.dtype  # avoid recomputation, take realdtype from initialized state

    start = state.iter_num % self.history_size
    value, grad = (state.value, state.grad)
    descent_direction = tree_scalar_mul(-1.0, tree_conj(grad))
    s_history = state.s_history
    y_history = state.y_history
    rho_history = state.rho_history

    if self.history_size:
      descent_direction = inv_hessian_product(
          pytree=descent_direction,
          s_history=s_history,
          y_history=y_history,
          rho_history=rho_history,
          gamma=state.gamma,
          start=start,
      )

    use_linesearch = (
        not isinstance(self.stepsize, Callable) and self.stepsize <= 0
    )
    if use_linesearch:
      init_stepsize = _init_stepsize(
          self.linesearch_init,
          self.max_stepsize,
          self.min_stepsize,
          self.increase_factor,
          state.stepsize,
      )
      new_stepsize, ls_state = self.run_ls(
          init_stepsize,
          params,
          value=value,
          grad=grad,
          descent_direction=descent_direction,
          fun_args=args,
          fun_kwargs=kwargs)
      new_params = ls_state.params
      new_value = ls_state.value
      new_grad = ls_state.grad
      new_aux = ls_state.aux
      failed_linesearch = ls_state.failed
      new_num_linesearch_iter = state.num_linesearch_iter + ls_state.iter_num
      new_num_grad_eval = state.num_grad_eval + ls_state.num_grad_eval
      new_num_fun_eval = state.num_fun_eval + ls_state.num_fun_eval

    else:
      if isinstance(self.stepsize, Callable):
        new_stepsize = self.stepsize(state.iter_num)
      else:
        new_stepsize = self.stepsize

      new_params = tree_add_scalar_mul(params, new_stepsize, descent_direction)
      (new_value, new_aux), new_grad = self._value_and_grad_with_aux(
          new_params, *args, **kwargs
      )
      new_num_grad_eval = state.num_grad_eval + 1
      new_num_fun_eval = state.num_fun_eval + 1
      new_num_linesearch_iter = state.num_linesearch_iter
      failed_linesearch = jnp.asarray(False)
    s = tree_sub(new_params, params)
    y = tree_conj(tree_sub(new_grad, grad))
    vdot_sy = tree_vdot_real(s, y)
    rho = jnp.where(vdot_sy == 0, 0, 1. / vdot_sy)

    if self.history_size:
      s_history = update_history(s_history, s, start)
      y_history = update_history(y_history, y, start)
      rho_history = update_history(rho_history, rho, start)

    if self.history_size and self.use_gamma:
      gamma = compute_gamma(s_history, y_history, start)
    else:
      gamma = jnp.array(1.0, dtype=realdtype)

    error = tree_l2_norm(new_grad)
    new_state = LbfgsState(iter_num=state.iter_num + 1,
                           value=new_value,
                           grad=new_grad,
                           stepsize=jnp.asarray(new_stepsize, dtype=realdtype),
                           error=jnp.asarray(error, dtype=realdtype),
                           s_history=s_history,
                           y_history=y_history,
                           rho_history=rho_history,
                           gamma=gamma,
                           aux=new_aux,
                           failed_linesearch=failed_linesearch,
                           num_grad_eval=new_num_grad_eval,
                           num_fun_eval=new_num_fun_eval,
                           num_linesearch_iter=new_num_linesearch_iter)

    if self.verbose:
      self.log_info(
          new_state,
          error_name="Gradient Norm",
          additional_info={
              "Objective Value": new_value,
              "Stepsize": new_stepsize,
              "Number Linesearch Iterations": 
              new_state.num_linesearch_iter - state.num_linesearch_iter
          }
      )
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
    (value, _), grad = self._value_and_grad_with_aux(params, *args, **kwargs)
    return value, grad

  def __post_init__(self):
    super().__post_init__()

    _fun_with_aux, _, self._value_and_grad_with_aux = \
      base._make_funs_with_aux(fun=self.fun,
                               value_and_grad=self.value_and_grad,
                               has_aux=self.has_aux)

    self.reference_signature = self.fun

    unroll = self._get_unroll_option()

    self.linesearch_solver = _setup_linesearch(
        linesearch=self.linesearch,
        fun=_fun_with_aux,
        value_and_grad=self._value_and_grad_with_aux,
        has_aux=True,
        maxlsiter=self.maxls,
        max_stepsize=self.max_stepsize,
        jit=self.jit,
        unroll=unroll,
        verbose=self.verbose,
    )
    self.run_ls = self.linesearch_solver.run

    # FIXME: to remove in future releases
    if self.condition is not None:
      warnings.warn("Argument condition is deprecated", DeprecationWarning)
    if self.decrease_factor is not None:
      warnings.warn(
          "Argument decrease_factor is deprecated", DeprecationWarning
      )
