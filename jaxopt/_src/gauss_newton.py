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

"""Gauss-Newton algorithm in JAX."""

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Union

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt._src import linear_solve
from jaxopt.tree_util import tree_l2_norm, tree_sub


class GaussNewtonState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  residual: Any
  value: Any
  delta: Any
  error: float
  gradient: Any
  aux: Optional[Any] = None


@dataclass(eq=False)
class GaussNewton(base.IterativeSolver):
  """Gauss-Newton nonlinear least-squares solver.

  Given the residual function ``f(x): R^n -> R^m``, where ``f(x) =
  residual_fun(x, *args, **kwargs)``, ``GaussNewton`` finds a local minimum of
  the cost function ``argmin_x 0.5 * sum(f(x) ** 2)``.

  Attributes:
    residual_fun: a smooth function of the form
      ``residual_fun(x, *args, **kwargs)``.
    maxiter: maximum number of iterations.
    tol: tolerance.
    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.
    has_aux: whether ``residual_fun`` outputs auxiliary data or not.
    verbose: whether to print information on every iteration or not.
    jit: whether to JIT-compile the bisection loop (default: True).
    unroll: whether to unroll the bisection loop (default: "auto").
  """
  residual_fun: Callable
  maxiter: int = 30
  tol: float = 1e-5
  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None
  has_aux: bool = False
  verbose: Union[bool, int] = False
  jit: bool = True
  unroll: base.AutoOrBoolean = "auto"

  def init_state(self,
                 init_params: Any,
                 *args,
                 **kwargs) -> GaussNewtonState:
    """Initialize the solver state.

    Args:
      init_params: pytree containing the initial parameters.
      *args: additional positional arguments to be passed to ``residual_fun``.
      **kwargs: additional keyword arguments to be passed to ``residual_fun``.
    Returns:
      state
    """
    # Compute actual values of state variables at init_param
    residual, aux = self._fun_with_aux(init_params, *args, **kwargs)
    matvec = lambda v: self._jtj_op(init_params, v, *args, **kwargs)
    gradient = self._jt_op(init_params, residual, *args, **kwargs)

    return GaussNewtonState(iter_num=jnp.asarray(0),
                            error=jnp.asarray(jnp.inf),
                            residual=residual,
                            value=0.5 * jnp.sum(jnp.square(residual)),
                            delta=init_params,
                            gradient=gradient,
                            aux=aux)

  def update(self,
             params,
             state: NamedTuple,
             *args,
             **kwargs) -> base.OptStep:
    """Performs one iteration of the least-squares solver.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
    Returns:
      (params, state)
    """
    residual, aux = self._fun_with_aux(params, *args, **kwargs)
    matvec = lambda v: self._jtj_op(params, v, *args, **kwargs)
    gradient = self._jt_op(params, residual, *args, **kwargs)

    delta_params = linear_solve.solve_cg(matvec, gradient)
    params = tree_sub(params, delta_params)
    value = 0.5 * jnp.sum(jnp.square(residual))

    state = GaussNewtonState(iter_num=state.iter_num + 1,
                             error=tree_l2_norm(delta_params),
                             residual=residual,
                             value=value,
                             delta=delta_params,
                             gradient=gradient,
                             aux=aux)

    if self.verbose:
      self.log_info(
          state,
          error_name="Norm GN Update",
          additional_info={"Objective Value": value}
      )
    return base.OptStep(params=params, state=state)

  def __post_init__(self):
    super().__post_init__()

    if self.has_aux:
      self._fun_with_aux = self.residual_fun
      self._fun = lambda *a, **kw: self._fun_with_aux(*a, **kw)[0]
    else:
      self._fun = self.residual_fun
      self._fun_with_aux = lambda *a, **kw: (self.residual_fun(*a, **kw),
                                             None)
    # We need this definition in the base solver run function
    def optimality_fun(params, *args, **kwargs):
      residual = self._fun(params, *args, **kwargs)
      return self._jt_op(params, residual, *args, **kwargs)
    self.optimality_fun = optimality_fun

  def _jtj_op(self, params, vec, *args, **kwargs):
    """Product with J.T J"""
    fun_with_args = lambda p: self._fun(p, *args, **kwargs)
    return jax.vjp(fun_with_args, params)[1](
        jax.jvp(fun_with_args, (params,), (vec,))[1])[0]

  def _jt_op(self, params, residual, *args, **kwargs):
    """Product with J.T"""
    fun_with_args = lambda p: self._fun(p, *args, **kwargs)
    return jax.vjp(fun_with_args, params)[1](residual)[0]
