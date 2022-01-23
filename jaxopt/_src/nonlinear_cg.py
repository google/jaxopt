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

"""Nonlinear conjugate gradient algorithm"""

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt._src.backtracking_linesearch import BacktrackingLineSearch
from jaxopt.tree_util import tree_vdot
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_sub
from jaxopt.tree_util import tree_div
from jaxopt.tree_util import tree_l2_norm


class NonlinearCGState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  stepsize: float
  error: float
  value: float
  grad: any
  descent_direction: jnp.ndarray
  aux: Optional[Any] = None


@dataclass(eq=False)
class NonlinearCG(base.IterativeSolver):
  """
    Nonlinear Conjugate Gradient Solver.
  Attributes:
    fun: a smooth function of the form ``fun(x, *args, **kwargs)``.
    method: which variant to calculate the beta parameter in Nonlinear CG.
      "polak-ribiere", "fletcher-reeves", "hestenes-stiefel"
      (default: "polak-ribiere")
    has_aux: whether function fun outputs one (False) or more values (True).
      When True it will be assumed by default that fun(...)[0] is the objective.
    maxiter: maximum number of proximal gradient descent iterations.
    tol: tolerance of the stopping criterion.
    maxls: maximum number of iterations to use in the line search.
    decrease_factor: factor by which to decrease the stepsize during line search
      (default: 0.8).
    increase_factor: factor by which to increase the stepsize during line search
      (default: 1.2).
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
    Algorithm 5.4 (page 121).
  """

  fun: Callable
  has_aux: bool = False

  maxiter: int = 100
  tol: float = 1e-3

  method: str = "polak-ribiere"  # same as SciPy
  condition: str = "strong-wolfe"
  maxls: int = 15
  decrease_factor: float = 0.8
  increase_factor: float = 1.2
  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None

  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"

  verbose: int = 0

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
    value, grad = self._value_and_grad_fun(init_params, *args, **kwargs)

    return NonlinearCGState(iter_num=jnp.asarray(0),
                            stepsize=jnp.asarray(1.0),
                            error=jnp.asarray(jnp.inf),
                            value=value,
                            grad=grad,
                            descent_direction=tree_scalar_mul(-1.0, grad))

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
    value, grad, descent_direction = state.value, state.grad, state.descent_direction
    init_stepsize = state.stepsize * self.increase_factor
    ls = BacktrackingLineSearch(fun=self._value_and_grad_fun,
                                value_and_grad=True,
                                maxiter=self.maxls,
                                decrease_factor=self.decrease_factor,
                                condition=self.condition)
    new_stepsize, ls_state = ls.run(init_stepsize=init_stepsize,
                                    params=params,
                                    value=value,
                                    grad=grad,
                                    *args, **kwargs)

    new_params = tree_add_scalar_mul(params, new_stepsize, descent_direction)
    (new_value, new_aux), new_grad = self._value_and_grad_with_aux(new_params, *args, **kwargs)

    if self.method == "polak-ribiere":
      # See Numerical Optimization, second edition, equation (5.44).
      gTg = tree_vdot(grad, grad)
      gTg = jnp.where(gTg >= eps, gTg, eps)
      new_beta = tree_div(tree_vdot(new_grad, tree_sub(new_grad, grad)), gTg)
      new_beta = jax.nn.relu(new_beta)
    elif self.method == "fletcher-reeves":
      # See Numerical Optimization, second edition, equation (5.41a).
      gTg = tree_vdot(grad, grad)
      gTg = jnp.where(gTg >= eps, gTg, eps)
      new_beta = tree_div(tree_vdot(new_grad, new_grad), gTg)
    elif self.method == 'hestenes-stiefel':
      # See Numerical Optimization, second edition, equation (5.45).
      grad_diff = tree_sub(new_grad, grad)
      dTg = tree_vdot(descent_direction, grad_diff)
      dTg = jnp.where(dTg >= eps, dTg, eps)
      new_beta = tree_div(tree_vdot(new_grad, grad_diff), dTg)
    else:
      raise ValueError("method should be either 'polak-ribiere', 'fletcher-reeves', or 'hestenes-stiefel'")

    new_descent_direction = tree_add_scalar_mul(tree_scalar_mul(-1, new_grad), new_beta, descent_direction)
    new_state = NonlinearCGState(iter_num=state.iter_num + 1,
                                 stepsize=jnp.asarray(new_stepsize),
                                 error=tree_l2_norm(grad),
                                 value=new_value,
                                 grad=new_grad,
                                 descent_direction=new_descent_direction,
                                 aux=new_aux)

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
