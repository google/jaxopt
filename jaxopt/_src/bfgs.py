# Copyright 2022 Google LLC
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

"""Implementation of the BFGS solver."""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Union
import warnings

import jax
import jax.numpy as jnp
from jaxopt._src import base
from jaxopt._src.linesearch_util import _init_stepsize
from jaxopt._src.linesearch_util import _setup_linesearch
from jaxopt._src.scipy_wrappers import make_onp_to_jnp
from jaxopt._src.scipy_wrappers import pytree_topology_from_example
from jaxopt._src.tree_util import tree_single_dtype
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_l2_norm
from jaxopt.tree_util import tree_sub


_dot = partial(jnp.dot, precision=jax.lax.Precision.HIGHEST)
_einsum = partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)


# Note that BFGS is not meant to be used with high-dimensional problems.
# We support pytrees via flattening.
def pytree_to_flat_array(pytree, dtype):
  """Utility to flatten a pytree."""
  flattened = [jnp.asarray(leaf, dtype).reshape(-1)
               for leaf in jax.tree_util.tree_leaves(pytree)]
  return jnp.concatenate(flattened)


class BfgsState(NamedTuple):
  """Named tuple containing state information."""

  iter_num: int
  value: float
  grad: Any
  stepsize: float
  error: float
  H: jnp.ndarray
  aux: Optional[Any] = None

  num_fun_eval: int = 0
  num_grad_eval: int = 0
  num_linesearch_iter: int = 0


@dataclass(eq=False)
class BFGS(base.IterativeSolver):
  """BFGS solver.

  BFGS is not meant to be used with high-dimensional problems (use LBFGS in this
  case).

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
    condition: Deprecated. Condition used to select the stepsize when using
      backtracking linesearch.
    maxls: maximum number of iterations to use in the line search.
    decrease_factor: Deprecated. factor by which to decrease the stepsize during
      backtracking line search (default: 0.8).
    increase_factor: factor by which to increase the stepsize during line search
      (default: 1.5).
    max_stepsize: upper bound on stepsize.
    min_stepsize: lower bound on stepsize guess at start of the linesearch run.
    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.
    jit: whether to JIT-compile the optimization loop (default: True).
    unroll: whether to unroll the optimization loop (default: "auto").
    verbose: if set to True or 1 prints the information at each step of 
      the solver, if set to 2, print also the information of the linesearch.

  Reference:
    Jorge Nocedal and Stephen Wright.
    Numerical Optimization, second edition.
    Algorithm 6.1 (page 140).
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
  condition: Any = None  # deprecated in v0.8
  maxls: int = 30
  decrease_factor: Any = None  # deprecated in v0.8
  increase_factor: float = 1.5
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
                 **kwargs) -> BfgsState:
    """Initialize the solver state.

    Args:
      init_params: pytree containing the initial parameters.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.

    Returns:
      state
    """
    (value, aux), grad = self._value_and_grad_with_aux(init_params, *args, **kwargs)

    dtype = tree_single_dtype(init_params)
    flat_init_params = pytree_to_flat_array(init_params, dtype)

    return BfgsState(iter_num=jnp.asarray(0),
                     value=value,
                     grad=grad,
                     stepsize=jnp.asarray(self.max_stepsize, dtype=dtype),
                     error=jnp.asarray(jnp.inf, dtype=dtype),
                     H=jnp.eye(len(flat_init_params), dtype=dtype),
                     aux=aux,
                     num_fun_eval=jnp.array(1, base.NUM_EVAL_DTYPE),
                     num_grad_eval=jnp.array(1, base.NUM_EVAL_DTYPE),
                     num_linesearch_iter=jnp.asarray(0, base.NUM_EVAL_DTYPE)
                     )

  def update(self,
             params: Any,
             state: BfgsState,
             *args,
             **kwargs) -> base.OptStep:
    """Performs one iteration of BFGS.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.

    Returns:
      (params, state)
    """
    value, grad = state.value, state.grad
    pytree_topology = pytree_topology_from_example(params)
    flat_array_to_pytree = make_onp_to_jnp(pytree_topology)
    dtype = tree_single_dtype(params)
    flat_grad = pytree_to_flat_array(grad, dtype)

    descent_direction = flat_array_to_pytree(-_dot(state.H, flat_grad))

    use_linesearch = not isinstance(self.stepsize, Callable) and self.stepsize <= 0

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
          value,
          grad,
          descent_direction,
          fun_args=args,
          fun_kwargs=kwargs,
      )
      new_params = ls_state.params
      new_value = ls_state.value
      new_grad = ls_state.grad
      new_aux = ls_state.aux
      new_num_linesearch_iter = state.num_linesearch_iter + ls_state.iter_num
      new_num_grad_eval = state.num_grad_eval + ls_state.num_grad_eval
      new_num_fun_eval = state.num_fun_eval + ls_state.num_fun_eval
    else:
      if isinstance(self.stepsize, Callable):
        new_stepsize = self.stepsize(state.iter_num)
      else:
        new_stepsize = self.stepsize

      new_params = tree_add_scalar_mul(params, new_stepsize, descent_direction)
      (new_value, new_aux), new_grad = self._value_and_grad_with_aux(new_params, *args, **kwargs)
      new_num_grad_eval = state.num_grad_eval + 1
      new_num_fun_eval = state.num_fun_eval + 1
      new_num_linesearch_iter = state.num_linesearch_iter

    s = tree_sub(new_params, params)
    y = tree_sub(new_grad, grad)
    flat_s = pytree_to_flat_array(s, dtype)
    flat_y = pytree_to_flat_array(y, dtype)
    rho = jnp.reciprocal(_dot(flat_y, flat_s))

    sy = jnp.outer(flat_s, flat_y)
    ss = jnp.outer(flat_s, flat_s)
    w = jnp.eye(len(flat_grad), dtype=rho.dtype) - rho * sy
    new_H = _einsum('ij,jk,lk', w, state.H, w) + rho * ss
    new_H = jnp.where(jnp.isfinite(rho), new_H, state.H)

    error = tree_l2_norm(new_grad)
    new_state = BfgsState(iter_num=state.iter_num + 1,
                          value=new_value,
                          grad=new_grad,
                          stepsize=jnp.asarray(new_stepsize),
                          error=jnp.asarray(error, dtype=dtype),
                          H=new_H,
                          aux=new_aux,
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

  def _value_and_grad_fun(self, params, *args, **kwargs):
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
        verbose=int(self.verbose)-1
    )
    self.run_ls = self.linesearch_solver.run

    # FIXME: to remove in future releases
    if self.condition is not None:
      warnings.warn("Argument condition is deprecated", DeprecationWarning)
    if self.decrease_factor is not None:
      warnings.warn(
          "Argument decrease_factor is deprecated", DeprecationWarning
      )
