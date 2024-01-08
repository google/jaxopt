# Copyright 2023 Google LLC
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

"""Limited-memory Broyden method"""

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
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_sub
from jaxopt.tree_util import tree_l2_norm
from jaxopt._src.tree_util import tree_single_dtype


def matvec(d_history, c_history, x, indices):
  # CD^T x
  # here j is the history dimension and i is the space dimension
  dtx = jnp.einsum('j..., ... -> j', d_history[indices], x)
  cdtx = jnp.einsum('j..., j -> ...', c_history[indices], dtx)
  return cdtx

def inv_jacobian_product_leaf(v: jnp.ndarray,
                              d_history: jnp.ndarray,
                              c_history: jnp.ndarray,
                              gamma: float = 1.0,
                              start: int= 0):
  # the computation of the jacobian inverse for
  # the Broyden method
  # this is done using the notations of DEQ
  history_size = len(d_history)

  indices = (start + jnp.arange(history_size)) % history_size

  v = gamma * v + matvec(d_history, c_history, v, indices)
  return v


def inv_jacobian_product(pytree: Any,
                         d_history: Any,
                         c_history: Any,
                         gamma: float = 1.0,
                         start: int = 0):
  """Product between an approximate jacobian inverse and a pytree.

  Histories are pytrees of the same structure as `pytree`.
  Leaves are arrays of shape `(history_size, ...)`, where
  `...` means the same shape as `pytree`'s leaves.

  The notation follows the scipy one.

  Args:
    pytree: pytree to multiply with.
    d_history: pytree with the same structure as `pytree`.
      Leaves contain v variables, i.e., `(x[k] - x[k-1])^T B / ((g[k] - g[k-1])^T (x[k] - x[k-1])^T B)`.
    c_history: pytree with the same structure as `pytree`.
      Leaves contain u variables, i.e., `(x[k] - x[k-1]) - B(g[k] - g[k-1])`.
    gamma: scalar to use for the initial inverse jacobian approximation,
      i.e., `gamma * I`.
    start: starting index in the circular buffer.
  """
  fun = partial(inv_jacobian_product_leaf,
                gamma=gamma,
                start=start)
  return tree_map(fun, pytree, d_history, c_history)

def inv_jacobian_rproduct(pytree: Any,
                          d_history: Any,
                          c_history: Any,
                          gamma: float = 1.0,
                          start: int = 0):
  return inv_jacobian_product(pytree, c_history, d_history, jnp.conjugate(gamma), start)


def init_history(pytree, history_size):
  fun = lambda leaf: jnp.zeros((history_size,) + leaf.shape, dtype=leaf.dtype)
  return tree_map(fun, pytree)


def update_history(history_pytree, new_pytree, last):
  fun = lambda history_array, new_value: history_array.at[last].set(new_value)
  return tree_map(fun, history_pytree, new_pytree)


class BroydenState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  value: float
  stepsize: float
  error: float
  d_history: Any
  c_history: Any
  gamma: jnp.ndarray
  aux: Optional[Any] = None
  failed_linesearch: bool = False

  num_fun_eval: int = 0
  num_linesearch_iter: int = 0


@dataclass(eq=False)
class Broyden(base.IterativeSolver):
  """Limited-memory Broyden solver.

  This method is a quasi-Newton approach to root finding.
  While similar to L-BFGS in spirit, it is not applied
  in the same situations: indeed, because the function
  whose root we are looking for is not necessarily a
  gradient, its Jacobian (i.e. its Hessian in the
  optimization case) is not necessarily symmetric.
  As a consequence, we cannot include symmetry in the
  secant conditions defining the updates of the Broyden
  matrices, and therefore the resulting Jacobian
  approximation is not symmetric, while it is for L-BFGS.
  Another consequence is that each Broyden update is of
  rank-1 while it is rank-2 for L-BFGS.

  Attributes:
    fun: a function of the form ``fun(x, *args, **kwargs)``.
    has_aux: whether ``fun`` outputs auxiliary data or not.
      If ``has_aux`` is False, ``fun`` is expected to be
        scalar-valued.
      If ``has_aux`` is True, then we have one of the following
        two cases.
      At each iteration of the algorithm, the auxiliary outputs are stored
        in ``state.aux``.

    maxiter: maximum number of Broyden iterations.
    tol: tolerance of the stopping criterion.

    stepsize: a stepsize to use (if <= 0, use backtracking line search),
      or a callable specifying the **positive** stepsize to use at each iteration.
    linesearch: the type of line search to use: for now only "backtracking" for backtracking
      line search is available.
    stop_if_linesearch_fails: whether to stop iterations if the line search fails.
      When True, this matches the behavior of core JAX.
    maxls: maximum number of iterations to use in the line search.
    decrease_factor: factor by which to decrease the stepsize during line search
      (default: 0.8).
    increase_factor: factor by which to increase the stepsize during line search
      (default: 1.5).
    max_stepsize: upper bound on stepsize.
    min_stepsize: lower bound on stepsize guess at start of the linesearch run.

    history_size: size of the memory to use.
    gamma: the initialization of the inverse Jacobian is going to be gamma * I.

    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.

    jit: whether to JIT-compile the optimization loop (default: True).
    unroll: whether to unroll the optimization loop (default: "auto").

    verbose: if set to True or 1 prints the information at each step of 
      the solver, if set to 2, print also the information of the linesearch.

  Reference:
    Charles G. Broyden.
    A Class of Methods for Solving Nonlinear Simultaneous Equations.
    Equation (4.5) (page 581).
  """

  fun: Callable
  has_aux: bool = False

  maxiter: int = 500
  tol: float = 1e-3

  stepsize: Union[float, Callable] = 0.0
  linesearch: str = "backtracking"
  stop_if_linesearch_fails: bool = False
  condition: str = "wolfe"
  maxls: int = 15
  decrease_factor: float = 0.8
  increase_factor: float = 1.5
  max_stepsize: float = 1.0
  # FIXME: should depend on whether float32 or float64 is used.
  min_stepsize: float = 1e-6

  history_size: int = None
  gamma: float = 1.0

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
                 **kwargs) -> BroydenState:
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
        d_history=init_params.state.d_history,
        c_history=init_params.state.c_history,
        gamma=init_params.state.gamma,
        iter_num=init_params.state.iter_num,
        stepsize=init_params.state.stepsize,
      )
      init_params = init_params.params
      dtype = tree_single_dtype(init_params)
    else:
      dtype = tree_single_dtype(init_params)
      state_kwargs = dict(
        d_history=init_history(init_params, self.history_size),
        c_history=init_history(init_params, self.history_size),
        gamma=jnp.asarray(self.gamma, dtype=dtype),
        iter_num=jnp.asarray(0),
        stepsize=jnp.asarray(self.max_stepsize, dtype=dtype),
      )
    value, aux = self._value_with_aux(init_params, *args, **kwargs)
    return BroydenState(value=value,
                        error=jnp.asarray(jnp.inf),
                        **state_kwargs,
                        aux=aux,
                        failed_linesearch=jnp.asarray(False),
                        num_fun_eval=jnp.array(1, base.NUM_EVAL_DTYPE),
                        num_linesearch_iter=jnp.array(0, base.NUM_EVAL_DTYPE)
                        )

  def update(self,
             params: Any,
             state: BroydenState,
             *args,
             **kwargs) -> base.OptStep:
    """Performs one iteration of Broyden.

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
    value = state.value

    jac_prod_kwargs = dict(
      d_history=state.d_history,
      c_history=state.c_history,
      gamma=state.gamma,
      start=start,
    )

    jac_prod = partial(
      inv_jacobian_product,
      **jac_prod_kwargs,
    )

    jac_rprod = partial(
      inv_jacobian_rproduct,
      **jac_prod_kwargs,
    )

    product = jac_prod(pytree=value)
    descent_direction = tree_scalar_mul(-1.0, product)

    use_linesearch = not isinstance(self.stepsize, Callable) and self.stepsize <= 0
    if use_linesearch:
      if self.linesearch == "backtracking":
        # we need to build the function used for the line search
        # which is going to be the squared norm of the original function
        # as in scipy https://github.com/scipy/scipy/blob/main/scipy/optimize/_nonlin.py#L278
        # we then need to check if the gradient can be obtained with jax
        # and if not we can build it in the same fashion as scipy
        # https://github.com/scipy/scipy/blob/main/scipy/optimize/_nonlin.py#L285
        def ls_fun_with_aux(params, *args, **kwargs):
          f, aux = self._value_with_aux(params, *args, **kwargs)
          norm_squared = tree_l2_norm(f, squared=True)
          return norm_squared, (f, aux)
        # here we need a check if the function is not smooth
        ls_fun_with_aux_and_grad = jax.value_and_grad(ls_fun_with_aux, has_aux=True)
        ls = BacktrackingLineSearch(fun=ls_fun_with_aux_and_grad,
                                    value_and_grad=True,
                                    maxiter=self.maxls,
                                    decrease_factor=self.decrease_factor,
                                    max_stepsize=self.max_stepsize,
                                    condition=self.condition,
                                    jit=self.jit,
                                    unroll=self.unroll,
                                    has_aux=True,
                                    verbose=int(self.verbose)-1,
                                    tol=1e-2)
        init_stepsize = jnp.where(state.stepsize <= self.min_stepsize,
                                  # If stepsize became too small, we restart it.
                                  self.max_stepsize,
                                  # Else, we increase a bit the previous one.
                                  state.stepsize * self.increase_factor)
        new_stepsize, ls_state = ls.run(init_stepsize,
                                        params, value, None,
                                        descent_direction,
                                        fun_args=args, fun_kwargs=kwargs)
        new_value, new_aux = ls_state.aux
        new_params = ls_state.params
        new_num_linesearch_iter = state.num_linesearch_iter + ls_state.iter_num
        new_num_fun_eval = state.num_fun_eval + ls_state.num_fun_eval
        failed_linesearch = ls_state.failed
      else:
        raise ValueError("Invalid name in 'linesearch' option.")
    else:
      # without line search
      if isinstance(self.stepsize, Callable):
        new_stepsize = self.stepsize(state.iter_num)
      else:
        new_stepsize = self.stepsize
      failed_linesearch = False

      new_params = tree_add_scalar_mul(params, new_stepsize, descent_direction)
      new_value, new_aux = self._value_with_aux(new_params, *args, **kwargs)
      new_num_fun_eval = state.num_fun_eval + 1
      new_num_linesearch_iter = state.num_linesearch_iter
    delta_x = tree_sub(new_params, params)
    v = jac_rprod(delta_x)
    delta_g = tree_sub(new_value, value)
    denom = 1 / tree_vdot(v, delta_g)
    d = tree_scalar_mul(denom, v)
    c = tree_sub(delta_x, jac_prod(delta_g))

    last = (start + self.history_size) % self.history_size
    d_history = update_history(state.d_history, d, last)
    c_history = update_history(state.c_history, c, last)

    new_state = BroydenState(iter_num=state.iter_num + 1,
                             value=new_value,
                             stepsize=jnp.asarray(new_stepsize),
                             error=tree_l2_norm(new_value),
                             d_history=d_history,
                             c_history=c_history,
                             gamma=state.gamma,
                             aux=new_aux,
                             num_fun_eval=new_num_fun_eval,
                             num_linesearch_iter=new_num_linesearch_iter,
                             failed_linesearch=failed_linesearch)

    if self.verbose:
      self.log_info(
          new_state,
          error_name="Norm Output",
          additional_info={
              "Stepsize": new_stepsize,
              "Number Linesearch Iterations": 
              new_state.num_linesearch_iter - state.num_linesearch_iter
          }
      )
    return base.OptStep(params=new_params, state=new_state)

  def optimality_fun(self, params, *args, **kwargs):
    """Optimality function mapping compatible with ``@custom_root``."""
    value = self._value_fun(params, *args, **kwargs)
    return value

  def _value_fun(self, params, *args, **kwargs):
    if isinstance(params, base.OptStep):
      params = params.params
    value, _ = self._value_with_aux(params, *args, **kwargs)
    return value

  def __post_init__(self):
    super().__post_init__()

    if self.has_aux:
      fun_ = self.fun
    else:
      fun_ = lambda p, *a, **kw: (self.fun(p, *a, **kw), None)

    self._value_with_aux = fun_

    self.reference_signature = self.fun

    if self.history_size is None:
      self.history_size = self.maxiter
