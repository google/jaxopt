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
"""Levenberg-Marquardt algorithm in JAX."""

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Union

from dataclasses import dataclass

from typing_extensions import Literal
import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt._src.linear_solve import solve_cg
from jaxopt._src.linear_solve import solve_cholesky
from jaxopt._src.linear_solve import solve_inv
from jaxopt._src.tree_util import tree_l2_norm, tree_inf_norm, tree_sub, tree_add, tree_mul


class LevenbergMarquardtState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  damping_factor: float
  increase_factor: float
  residual: Any
  loss: Any
  delta: Any
  error: float
  gradient: Any
  aux: Optional[Any] = None


@dataclass(eq=False)
class LevenbergMarquardt(base.IterativeSolver):
  """Levenberg-Marquardt nonlinear least-squares solver.

    Given the residual function `func` (x): R^n -> R^m, `least_squares` finds a
    local minimum of the cost function F(x):

    ```
    argmin_x F(x) = 0.5 * sum(f_i(x)**2), i = 0, ..., m - 1
    f(x) = func(x, *args)
    ```

    If stop_criterion is 'madsen-nielsen', the convergence is achieved once the
    coeff update satisfies ``||dcoeffs||_2 <= xtol * (||coeffs||_2 + xtol) `` or
    the gradient satisfies ``||grad(f)||_inf <= gtol``.

  Attributes:
    residual_fun: a smooth function of the form ``residual_fun(x, *args,
      **kwargs)``.
    maxiter: maximum increase_factormber of iterations.
    damping_parameter: The parameter which adds a correction to the equation
      derived for updating the coefficients using Gauss-Newton method. Please
      see section 3.2. of K. Madsen et al. in the book "Methods for nonlinear
      least squares problems" for more information.
    stop_criterion: The criterion to use for the convergence of the while loop.
      e.g., for 'madsen-nielsen' the criteria is to satisfy the two equations
      for delta_params and gradient that is mentioned above. If 'grad-l2' is
      selected, the convergence is achieved if l2 of gradient is smaller or
      equal to tol.
    tol: tolerance.
    xtol: float, optional The convergence tolerance for the second norm of the
      coefficient update.
    gtol: float, optional The convergence tolerance for the inf norm of the
      residual gradient.
    solver: str, optional The solver to use when finding delta_params, the
      update to the params in each iteration. This is done through solving a
      system of linear equation Ax=b. 'cholesky' (Cholesky factorization), 'inv'
      (explicit multiplication with matrix inverse). The user can provide custom
      solvers, for example using jaxopt.linear_solve.solve_cg which are more
      scalable for runtime but take longer compilations. 'cholesky' is
      faster than 'inv' since it uses the symmetry feature of A.
    geodesic: bool, if we would like to include the geodesic acceleration when
      solving for the delta_params in every iteration.
    contribution_ratio_threshold: float, the threshold for acceleration/velocity
      ratio. We update the parameters in the algorithm only if the ratio is
      smaller than this threshold value.
    implicit_diff: bool, whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.
    verbose: bool, whether to print error on every iteration or not.
      Warning: verbose=True will automatically disable jit.
    jit: whether to JIT-compile the bisection loop (default: "auto").
    unroll: whether to unroll the bisection loop (default: "auto").

  Reference: This algorithm is for finding the best fit parameters based on the
    algorithm 6.18 provided by K. Madsen & H. B. Nielsen in the book
    "Introduction to Optimization and Data Fitting".
  """
  residual_fun: Callable

  maxiter: int = 30

  damping_parameter: float = 1e-6
  stop_criterion: Literal['grad-l2-norm', 'madsen-nielsen'] = 'grad-l2-norm'
  tol: float = 1e-3
  xtol: float = 1e-3
  gtol: float = 1e-3

  solver: Union[Literal['cholesky', 'inv'], Callable] = solve_cg
  geodesic: bool = False
  contribution_ratio_threshold = 0.75

  verbose: bool = False
  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None
  has_aux: bool = False
  jit: base.AutoOrBoolean = 'auto'
  unroll: base.AutoOrBoolean = 'auto'

  # We are overriding the _cond_fun of the base solver to enable stopping based
  # on gradient or delta_params
  def _cond_fun(self, inputs):
    params, state = inputs[0]
    if self.verbose:
      print(f' l2 norm of gradient: {state.error}')
    if self.stop_criterion == 'madsen-nielsen':
      tree_mul_term = self.xtol * (tree_l2_norm(params) - self.xtol)
      return jnp.all(jnp.array([
        tree_inf_norm(state.gradient) > self.gtol,
        tree_l2_norm(state.delta) > tree_mul_term
      ]))
    elif self.stop_criterion == 'grad-l2-norm':
      return state.error > self.tol
    else:
      raise NotImplementedError

  def init_state(self, init_params: Any, *args,
                 **kwargs) -> LevenbergMarquardtState:
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
    gradient = self._jt_op(init_params, residual, *args, **kwargs)

    # found = jnp.linalg.norm(grad, jnp.inf) <= gtol
    jtj_diag = self._jtj_diag_op(init_params, *args, **kwargs)
    damping_factor = self.damping_parameter * jnp.max(jtj_diag)
    delta_params = jnp.zeros_like(init_params)

    return LevenbergMarquardtState(
        iter_num=jnp.asarray(0),
        damping_factor=damping_factor,
        increase_factor=2,
        error=tree_l2_norm(gradient),
        residual=residual,
        loss=0.5 * jnp.dot(residual, residual),
        delta=delta_params,
        gradient=gradient,
        aux=aux)

  def update_state_using_gain_ratio(self, gain_ratio, contribution_ratio_diff,
                                    gain_ratio_test_init_state, *args,
                                    **kwargs):
    """The function to return state variables based on gain ratio.

      Please see by page 120-121 of the book "Introduction to Optimization and
      Data Fitting" by K. Madsen & H. B. Nielsen for details.
    """

    def gain_ratio_test_true_func(params, damping_factor, increase_factor,
                                  residual, gradient, updated_params, aux):

      params = updated_params

      residual, aux = self._fun_with_aux(params, *args, **kwargs)

      # Calculate gradient based on Eq. 6.6 of "Introduction to optimization
      # and data fitting" g=JT * r, where J is jacobian and r is residual.
      gradient = self._jt_op(params, residual, *args, **kwargs)

      damping_factor = damping_factor * jax.lax.max(1 / 3, 1 -
                                                    (2 * gain_ratio - 1)**3)
      increase_factor = 2

      return params, damping_factor, increase_factor, gradient, residual, aux

    def gain_ratio_test_false_func(params, damping_factor, increase_factor,
                                   residual, gradient, updated_params, aux):

      damping_factor = damping_factor * increase_factor
      increase_factor = 2 * increase_factor
      return params, damping_factor, increase_factor, gradient, residual, aux

    # Calling the jax condition function:
    # Note that only the parameters that are used in the rest of the program
    # are returned by this function.

    gain_ratio_test_is_met = jnp.logical_and(gain_ratio > 0.0,
                                             contribution_ratio_diff <= 0.0)

    gain_ratio_test_is_met_ret = gain_ratio_test_true_func(
        *gain_ratio_test_init_state)
    gain_ratio_test_not_met_ret = gain_ratio_test_false_func(
        *gain_ratio_test_init_state)

    gain_ratio_test_is_met_ret = jax.tree_map(
        lambda x: gain_ratio_test_is_met * x, gain_ratio_test_is_met_ret)

    gain_ratio_test_not_met_ret = jax.tree_map(
        lambda x: (1.0 - gain_ratio_test_is_met) * x,
        gain_ratio_test_not_met_ret)

    params, damping_factor, increase_factor, gradient, residual, aux = (
        jax.tree_map(lambda x, y: x + y, gain_ratio_test_is_met_ret,
                     gain_ratio_test_not_met_ret))

    return params, damping_factor, increase_factor, gradient, residual, aux

  def update_state_using_delta_params(self, loss_curr, params, delta_params,
                                      contribution_ratio_diff, damping_factor,
                                      increase_factor, gradient, residual, aux,
                                      *args, **kwargs):
    """The function to return state variables based on delta_params.

    Define the functions required for the major conditional of the algorithm,
    which checks the magnitude of dparams and checks if it is small enough.
    for the value of dparams.
    """

    updated_params = params + delta_params

    residual_next = self._fun(updated_params, *args, **kwargs)

    # Calculate denominator of the gain ratio based on Eq. 6.16, "Introduction
    # to optimization and data fitting", L(0)-L(hlm)=0.5*hlm^T*(mu*hlm-g).
    gain_ratio_denom = 0.5 * jnp.dot(
        jnp.transpose(delta_params), (damping_factor * delta_params - gradient))

    # Current value of loss function F=0.5*||f||^2.
    loss_next = 0.5 * jnp.dot(residual_next, residual_next)

    gain_ratio = (loss_curr - loss_next) / gain_ratio_denom

    gain_ratio_test_init_state = (params, damping_factor, increase_factor,
                                  residual, gradient, updated_params, aux)

    # Calling the jax condition function:
    # Note that only the parameters that are used in the rest of the program
    # are returned by this function.

    params, damping_factor, increase_factor, gradient, residual, aux = (
        self.update_state_using_gain_ratio(gain_ratio, contribution_ratio_diff,
                                           gain_ratio_test_init_state, *args,
                                           **kwargs))

    return params, damping_factor, increase_factor, gradient, residual, aux

  def update(self, params, state: NamedTuple, *args, **kwargs) -> base.OptStep:
    """Performs one iteration of the least-squares solver.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.

    Returns:
      (params, state)
    """

    # Current value of the loss function F=0.5*||f||^2.
    loss_curr = state.loss

    matvec = lambda v: self._jtj_op(params, v, *args, **kwargs)

    if isinstance(self.solver, Callable):
      velocity = self.solver(
          matvec, state.gradient, ridge=state.damping_factor, init=state.delta)
      delta_params = velocity
      if self.geodesic:
        # jtrpp=JT * r" - J is jacobian and r" is second order dir. derivative
        rpp = self._d2fvv_op(params, delta_params, delta_params, *args,
                             **kwargs)
        jtrpp = self._jt_op(params, rpp, *args, **kwargs)
        acceleration = self.solver(matvec, jtrpp, ridge=state.damping_factor)
        delta_params += 0.5*acceleration
    elif self.solver == 'cholesky':
      velocity = solve_cholesky(
          matvec, state.gradient, ridge=state.damping_factor)
      delta_params = velocity
      if self.geodesic:
        # jtrpp=JT * r" - J is jacobian and r" is second order dir. derivative
        rpp = self._d2fvv_op(params, delta_params, delta_params, *args, **kwargs)
        jtrpp = self._jt_op(params, rpp, *args, **kwargs)
        acceleration = 0.5*solve_cholesky(matvec, jtrpp, ridge=state.damping_factor)
        delta_params += acceleration
    elif self.solver == 'inv':
      velocity = solve_inv(
          matvec, state.gradient, ridge=state.damping_factor)
      delta_params = velocity
      if self.geodesic:
        # jtrpp=JT * r" - J is jacobian and r" is second order dir. derivative
        rpp = self._d2fvv_op(params, delta_params, delta_params, *args,
                             **kwargs)
        jtrpp = self._jt_op(params, rpp, *args, **kwargs)
        acceleration = solve_inv(matvec, jtrpp, ridge=state.damping_factor)
        delta_params += 0.5*acceleration
    if self.geodesic:
      contribution_ratio_diff = jnp.linalg.norm(acceleration) / jnp.linalg.norm(
          velocity) - self.contribution_ratio_threshold
    else:
      contribution_ratio_diff = 0.0

    delta_params = -delta_params

    # Checking if the dparams satisfy the "sufficiently small" criteria.
    params, damping_factor, increase_factor, gradient, residual, aux = (
        self.update_state_using_delta_params(loss_curr, params, delta_params,
                                             contribution_ratio_diff,
                                             state.damping_factor,
                                             state.increase_factor,
                                             state.gradient, state.residual,
                                             state.aux, *args, **kwargs))

    state = LevenbergMarquardtState(
        iter_num=state.iter_num + 1,
        damping_factor=damping_factor,
        increase_factor=increase_factor,
        error=tree_l2_norm(gradient),
        residual=residual,
        loss=0.5 * jnp.dot(residual, residual),
        delta=delta_params,
        gradient=gradient,
        aux=aux)

    return base.OptStep(params=params, state=state)

  def __post_init__(self):
    if self.has_aux:
      self._fun_with_aux = self.residual_fun
      self._fun = lambda *a, **kw: self._fun_with_aux(*a, **kw)[0]
    else:
      self._fun = self.residual_fun
      self._fun_with_aux = lambda *a, **kw: (self.residual_fun(*a, **kw), None)

  def optimality_fun(self, params, *args, **kwargs):
    """Optimality function mapping compatible with ``@custom_root``."""
    residual = self._fun(params, *args, **kwargs)
    return self._jt_op(params, residual, *args, **kwargs)

  def _jt_op(self, params, residual, *args, **kwargs):
    """Product of J^T and residual -- J: jacobian of fun at params."""
    fun_with_args = lambda p: self._fun(p, *args, **kwargs)
    _, vjpfun = jax.vjp(fun_with_args, params)
    jt_op_val, = vjpfun(residual)
    return jt_op_val

  def _jtj_op(self, params, vec, *args, **kwargs):
    """Product of J^T.J with vec using vjp & jvp, where J is jacobian of fun at params."""
    fun_with_args = lambda p: self._fun(p, *args, **kwargs)
    _, vjpfun = jax.vjp(fun_with_args, params)
    _, jvp_val = jax.jvp(fun_with_args, (params,), (vec,))
    jtj_op_val, = vjpfun(jvp_val)
    return jtj_op_val

  def _jtj_diag_op(self, params, *args, **kwargs):
    """Diagonal elements of J^T.J, where J is jacobian of fun at params."""
    diag_op = lambda v: jnp.dot(
        jnp.transpose(v), self._jtj_op(params, v, *args, **kwargs))
    return jax.vmap(diag_op)(jnp.eye(len(params))).T

  def _d2fvv_op(self, primals, tangents1, tangents2, *args, **kwargs):
    """Product with d2f.v1v2."""
    fun_with_args = lambda p: self._fun(p, *args, **kwargs)
    g = lambda pr: jax.jvp(fun_with_args, (pr,), (tangents1,))[1]
    return jax.jvp(g, (primals,), (tangents2,))[1]
