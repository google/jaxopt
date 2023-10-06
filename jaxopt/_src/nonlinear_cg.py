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

"""Nonlinear conjugate gradient algorithm."""

import warnings
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from jaxopt._src import base
from jaxopt._src.linesearch_util import _init_stepsize
from jaxopt._src.linesearch_util import _setup_linesearch
from jaxopt._src.tree_util import tree_single_dtype, get_real_dtype
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_div
from jaxopt.tree_util import tree_l2_norm
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_sub
from jaxopt.tree_util import tree_vdot_real
from jaxopt.tree_util import tree_conj


class NonlinearCGState(NamedTuple):
  """Named tuple containing state information."""

  iter_num: int
  stepsize: float
  error: float
  value: float
  grad: Any
  descent_direction: Any
  aux: Optional[Any] = None

  num_fun_eval: int = 0
  num_grad_eval: int = 0
  num_linesearch_iter: int = 0


@dataclass(eq=False)
class NonlinearCG(base.IterativeSolver):
  """Nonlinear conjugate gradient solver.

  Supports complex variables, see second reference.

  Attributes:
    fun: a smooth function of the form ``fun(x, *args, **kwargs)``.
    value_and_grad: whether ``fun`` just returns the value (False) or both
      the value and gradient (True).
    has_aux: whether ``fun`` outputs auxiliary data or not.
      If ``value_and_grad == False``, the output should be
      ``value, aux = fun(...)``.
      If ``value_and_grad == True``, the output should be
      ``(value, aux), grad = fun(...)``.
      The auxiliary outputs are stored in ``state.aux``.

    maxiter: maximum number of proximal gradient descent iterations.
    tol: tolerance of the stopping criterion.

    method: which variant to calculate the beta parameter in Nonlinear CG.
      "polak-ribiere", "fletcher-reeves", "hestenes-stiefel"
      (default: "polak-ribiere")

    linesearch: the type of line search to use: "backtracking" for backtracking
      line search, "zoom" for zoom line search or "hager-zhang" for Hager-Zhang
      line search.
    linesearch_init: strategy for line-search initialization. By default, it
      will use "increase", which will increased the step-size by a factor of
      `increase_factor` at each iteration if the step-size is larger than
      `min_stepsize`, and set it to `max_stepsize` otherwise. Other choices are
      "max", that initializes the step-size to `max_stepsize` at every
      iteration, and "current", that uses the step-size from the previous
      iteration.
    condition: Deprecated. Condition used to select the stepsize when using
      backtracking linesearch.
    maxls: maximum number of iterations to use in the line search.
    decrease_factor: Deprecated. Factor by which to decrease the stepsize during
      line search when using backtracking linesearch (default: 0.8).
    increase_factor: factor by which to increase the stepsize during line search
      (default: 1.2).
    max_stepsize: upper bound on stepsize.
    min_stepsize: lower bound on stepsize guess at start of each linesearch run.
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
    Algorithm 5.4 (page 121).

    Laurent Sorber, Marc van Barel, and Lieven de Lathauwer.
    Unconstrained Optimization of Real Functions in Complex Variables.
    SIAM J. Optim., Vol. 22, No. 3, pp. 879-898
  """

  fun: Callable
  value_and_grad: bool = False
  has_aux: bool = False

  maxiter: int = 100
  tol: float = 1e-3

  method: str = "polak-ribiere"  # same as SciPy
  linesearch: str = "zoom"
  linesearch_init: str = "increase"
  condition: Any = None  # deprecated in v0.8
  maxls: int = 30
  decrease_factor: Any = None  # deprecated in v0.8
  increase_factor: float = 1.2
  max_stepsize: float = 1.0
  # FIXME: should depend on whether float32 or float64 is used.
  min_stepsize: float = 1e-6

  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None

  jit: bool = True
  unroll: base.AutoOrBoolean = "auto"

  verbose: Union[bool, int] = False

  def init_state(self,
                 init_params: Any,
                 *args,
                 **kwargs) -> NonlinearCGState:
    """Initialize the solver state.

    Args:
      init_params: pytree containing the initial parameters.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.

    Returns:
      state
    """
    (value, aux), grad = self._value_and_grad_with_aux(init_params,
                                                       *args,
                                                       **kwargs)

    dtype = tree_single_dtype(init_params)
    realdtype = get_real_dtype(dtype)

    return NonlinearCGState(iter_num=jnp.asarray(0),
                            stepsize=jnp.asarray(
        self.max_stepsize, dtype=realdtype),
        error=jnp.asarray(jnp.inf, dtype=realdtype),
        value=value,
        grad=grad,
        descent_direction=tree_scalar_mul(
        -1.0, tree_conj(grad)),
        aux=aux,
        num_fun_eval=jnp.asarray(1, base.NUM_EVAL_DTYPE),
        num_grad_eval=jnp.asarray(1, base.NUM_EVAL_DTYPE),
        num_linesearch_iter=jnp.array(
        0, base.NUM_EVAL_DTYPE)
    )

  def update(self,
             params: Any,
             state: NonlinearCGState,
             *args,
             **kwargs) -> base.OptStep:
    """Performs one iteration of Fletcher-Reeves Algorithm.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.

    Returns:
      (params, state)
    """

    eps = 1e-6
    value = state.value
    grad = state.grad
    descent_direction = state.descent_direction

    # Kept choice of no descent direction for backtracking line-search
    # FIXME: should discuss why it was the case
    if self.linesearch == "backtracking":
      ls_descent_direction = None
    else:
      ls_descent_direction = descent_direction

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
        value,
        grad,
        ls_descent_direction,
        args,
        kwargs,
    )
    new_params = ls_state.params
    new_value = ls_state.value
    new_grad = ls_state.grad
    new_aux = ls_state.aux
    new_num_fun_eval = state.num_fun_eval + ls_state.num_fun_eval
    new_num_grad_eval = state.num_grad_eval + ls_state.num_grad_eval
    new_num_linesearch_iter = state.num_linesearch_iter + ls_state.iter_num

    if self.method == "polak-ribiere":
      # See Numerical Optimization, second edition, equation (5.44).
      gTg = tree_vdot_real(grad, grad)
      gTg = jnp.where(gTg >= eps, gTg, eps)
      new_beta = tree_vdot_real(
        tree_conj(tree_sub(new_grad, grad)), tree_conj(new_grad)) / gTg
      new_beta = jax.nn.relu(new_beta)
    elif self.method == "fletcher-reeves":
      # See Numerical Optimization, second edition, equation (5.41a).
      gTg = tree_vdot_real(grad, grad)
      gTg = jnp.where(gTg >= eps, gTg, eps)
      new_beta = tree_vdot_real(new_grad, new_grad) / gTg
      new_beta = jax.nn.relu(new_beta)
    elif self.method == "hestenes-stiefel":
      # See Numerical Optimization, second edition, equation (5.45).
      grad_diff = tree_sub(new_grad, grad)
      dTg = tree_vdot_real(tree_conj(grad_diff), descent_direction)
      dTg = jnp.where(dTg >= eps, dTg, eps)
      new_beta = tree_vdot_real(
        tree_conj(grad_diff), tree_conj(new_grad)) / dTg
      new_beta = jax.nn.relu(new_beta)
    else:
      raise ValueError("method argument should be either 'polak-ribiere', "
                       "'fletcher-reeves', or 'hestenes-stiefel'.")

    new_descent_direction = tree_add_scalar_mul(tree_scalar_mul(-1, tree_conj(new_grad)),
                                                new_beta,
                                                descent_direction)
    error = tree_l2_norm(grad)
    realdtype = state.error.dtype
    new_state = NonlinearCGState(iter_num=state.iter_num + 1,
                                 stepsize=jnp.asarray(
                                   new_stepsize, dtype=realdtype),
                                 error=jnp.asarray(error, dtype=realdtype),
                                 value=new_value,
                                 grad=new_grad,
                                 descent_direction=new_descent_direction,
                                 aux=new_aux,
                                 num_fun_eval=new_num_fun_eval,
                                 num_grad_eval=new_num_grad_eval,
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
    return self._grad_fun(params, *args, **kwargs)

  def _value_and_grad_fun(self, params, *args, **kwargs):
    (value, _), grad = self._value_and_grad_with_aux(params, *args, **kwargs)
    return value, grad

  def _grad_fun(self, params, *args, **kwargs):
    return self._value_and_grad_fun(params, *args, **kwargs)[1]

  def __post_init__(self):
    super().__post_init__()

    _fun_with_aux, _, self._value_and_grad_with_aux = base._make_funs_with_aux(
        fun=self.fun, value_and_grad=self.value_and_grad, has_aux=self.has_aux
    )

    self.reference_signature = self.fun

    unroll = self._get_unroll_option()

    linesearch_solver = _setup_linesearch(
        linesearch=self.linesearch,
        fun=_fun_with_aux,
        value_and_grad=self._value_and_grad_with_aux,
        has_aux=True,
        maxlsiter=self.maxls,
        max_stepsize=self.max_stepsize,
        jit=self.jit,
        unroll=unroll,
        verbose=int(self.verbose)-1
    )

    self.run_ls = linesearch_solver.run

    if self.condition is not None:
      warnings.warn("Argument condition is deprecated", DeprecationWarning)
    if self.decrease_factor is not None:
      warnings.warn(
          "Argument decrease_factor is deprecated", DeprecationWarning
      )
