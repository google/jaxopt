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

"""BFGS"""

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
from jaxopt._src.zoom_linesearch import zoom_linesearch
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_l2_norm
from jaxopt.tree_util import tree_sub
from jaxopt._src.tree_util import tree_single_dtype
from jaxopt._src.scipy_wrappers import make_onp_to_jnp
from jaxopt._src.scipy_wrappers import pytree_topology_from_example


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
  stepsize: float
  error: float
  H: jnp.ndarray
  aux: Optional[Any] = None


@dataclass(eq=False)
class BFGS(base.IterativeSolver):
  """BFGS solver.

  BFGS is not meant to be used with high-dimensional problems (use LBFGS in this
  case).

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
      line search or "zoom" for zoom line search.
    maxls: maximum number of iterations to use in the line search.
    decrease_factor: factor by which to decrease the stepsize during line search
      (default: 0.8).
    increase_factor: factor by which to increase the stepsize during line search
      (default: 1.5).
    max_stepsize: upper bound on stepsize.
    min_stepsize: lower bound on stepsize.

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
    Algorithm 6.1 (page 140).
  """

  fun: Callable
  value_and_grad: bool = False
  has_aux: bool = False

  maxiter: int = 500
  tol: float = 1e-3

  stepsize: Union[float, Callable] = 0.0
  linesearch: str = "zoom"
  condition: str = "strong-wolfe"
  maxls: int = 15
  decrease_factor: float = 0.8
  increase_factor: float = 1.5
  max_stepsize: float = 1.0
  # FIXME: should depend on whether float32 or float64 is used.
  min_stepsize: float = 1e-6

  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None

  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"

  verbose: bool = False

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
    if self.has_aux:
      _, aux = self.fun(init_params, *args, **kwargs)
    else:
      aux = None

    dtype = tree_single_dtype(init_params)
    flat_init_params = pytree_to_flat_array(init_params, dtype)

    return BfgsState(iter_num=jnp.asarray(0),
                     value=jnp.asarray(jnp.inf),
                     stepsize=jnp.asarray(self.max_stepsize, dtype=dtype),
                     error=jnp.asarray(jnp.inf),
                     H=jnp.eye(len(flat_init_params), dtype=dtype),
                     aux=aux)

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
    (value, aux), grad = self._value_and_grad_with_aux(params, *args, **kwargs)

    pytree_topology = pytree_topology_from_example(params)
    flat_array_to_pytree = make_onp_to_jnp(pytree_topology)
    dtype = tree_single_dtype(params)
    flat_grad = pytree_to_flat_array(grad, dtype)

    descent_direction = flat_array_to_pytree(-_dot(state.H, flat_grad))

    if not isinstance(self.stepsize, Callable) and self.stepsize <= 0:
      # with line search

      if self.linesearch == "backtracking":
        ls = BacktrackingLineSearch(fun=self._value_and_grad_fun,
                                    value_and_grad=True,
                                    maxiter=self.maxls,
                                    decrease_factor=self.decrease_factor,
                                    max_stepsize=self.max_stepsize,
                                    condition=self.condition,
                                    jit=self.jit,
                                    unroll=self.unroll)
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
        new_params = ls_state.params
        new_grad = ls_state.grad

      elif self.linesearch == "zoom":
        ls_state = zoom_linesearch(f=self._value_and_grad_with_aux,
                                   xk=params, pk=descent_direction,
                                   old_fval=value, gfk=grad, maxiter=self.maxls,
                                   value_and_grad=True, has_aux=True,
                                   args=args, kwargs=kwargs)
        new_value = ls_state.f_k
        new_stepsize = ls_state.a_k
        new_grad = ls_state.g_k
        # FIXME: zoom_linesearch currently doesn't return new_params
        # so we have to recompute it.
        t = new_stepsize.astype(tree_single_dtype(params))
        new_params = tree_add_scalar_mul(params, t, descent_direction)

      else:
        raise ValueError("Invalid name in 'linesearch' option.")

    else:
      # without line search
      if isinstance(self.stepsize, Callable):
        new_stepsize = self.stepsize(state.iter_num)
      else:
        new_stepsize = self.stepsize

      new_params = tree_add_scalar_mul(params, new_stepsize, descent_direction)
      # FIXME: this requires a second function call per iteration.
      new_value, new_grad = self._value_and_grad_fun(new_params, *args,
                                                     **kwargs)

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

    new_state = BfgsState(iter_num=state.iter_num + 1,
                          value=new_value,
                          stepsize=jnp.asarray(new_stepsize),
                          error=tree_l2_norm(new_grad),
                          H=new_H,
                          # FIXME: we should return new_aux here but
                          # BacktrackingLineSearch currently doesn't support
                          # an aux.
                          aux=aux)

    return base.OptStep(params=new_params, state=new_state)

  def optimality_fun(self, params, *args, **kwargs):
    """Optimality function mapping compatible with ``@custom_root``."""
    return self._value_and_grad_fun(params, *args, **kwargs)[1]

  def _value_and_grad_fun(self, params, *args, **kwargs):
    (value, aux), grad = self._value_and_grad_with_aux(params, *args, **kwargs)
    return value, grad

  def __post_init__(self):
    _, _, self._value_and_grad_with_aux = \
      base._make_funs_with_aux(fun=self.fun,
                               value_and_grad=self.value_and_grad,
                               has_aux=self.has_aux)

    self.reference_signature = self.fun
