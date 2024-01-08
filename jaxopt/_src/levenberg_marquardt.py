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

import math
from typing import Any
from typing import Callable
from typing import Literal
from typing import NamedTuple
from typing import Optional
from typing import Union

from dataclasses import dataclass
import warnings

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from jaxopt._src import base
from jaxopt._src.linear_solve import solve_cg
from jaxopt._src.linear_solve import solve_cholesky
from jaxopt._src.linear_solve import solve_inv
from jaxopt._src.linear_solve import solve_lu
from jaxopt._src.linear_solve import solve_qr
from jaxopt._src.tree_util import tree_l2_norm, tree_inf_norm, tree_sub, tree_add, tree_mul


class LevenbergMarquardtState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  damping_factor: float
  increase_factor: float
  residual: Any
  value: Any
  delta: Any
  error: float
  gradient: Any
  jac: Any
  jt: Any
  jtj: Any
  hess_res: Any
  aux: Optional[Any] = None

@dataclass(eq=False)
class LevenbergMarquardt(base.IterativeSolver):
  """Levenberg-Marquardt (LM) nonlinear least-squares solver.

    Given the residual function `func` (x): R^n -> R^m, `least_squares` finds a
    local minimum of the cost function F(x):

    ```
    argmin_x F(x) = 0.5 * sum(f_i(x)**2), i = 0, ..., m - 1
    f(x) = func(x, *args)
    ```

    This results in solving the following normal equation:
    ```
    (J^T @ J + µ I) @ ∆params = -J^T @ f(x)
    ```
    we use this method as recommended by K. Madsen & H. B. Nielsen in the book
    "Introduction to Optimization and Data Fitting" page 122.


    If stop_criterion is 'madsen-nielsen', the convergence is achieved once the
    coeff update satisfies ``||dcoeffs||_2 <= xtol * (||coeffs||_2 + xtol) `` or
    the gradient satisfies ``||grad(f)||_inf <= gtol``.

    It is possible to find the LM step by solving the following equation:
    ```
    [     J    ]              [ -f(x) ]
    |          | @ ∆params =  |       |
    [ Sqrt(µ)I ]              [   0   ]
    ```
    If LHS matrix is multiplied by its transpose, one can show that solving this
    equation is the same as the normal equation. This is the approach we take if
    QR factorization method is chosen. Note that since we are not multipyling the
    LHS matrix by its transpose, the condition number of the LHS matrix is more
    accurate and therefore this method can be used for ill-conditioned Jacobians.
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
      system of linear equation Ax=b. 'cholesky', 'lu', or 'qr' factorizations,
      'inv' (explicit multiplication with matrix inverse). Note that the inverse
      approach is the most expensive and least accurate and is just given as an
      option for legacy reasons. The user can provide custom solvers, for example
      using jaxopt.linear_solve.solve_cg which are more scalable for runtime but
      take longer compilations. Cholesky is faster than inverse since it uses
      the symmetry feature of A. QR factorization is preferred for ill-posed
      stiff problems due to not solving the normal equation.
    geodesic: bool, if we would like to include the geodesic acceleration when
      solving for the delta_params in every iteration.
    contribution_ratio_threshold: float, the threshold for acceleration/velocity
      ratio. We update the parameters in the algorithm only if the ratio is
      smaller than this threshold value.
    verbose: bool, whether to print information on every iteration or not.
    jac_fun: Callable, a function to calculate the Jacobian. If not None, this
      function is used instead of directly calculating it using ``jax.jacfwd``.
    materialize_jac: bool, whether to materialize Jacobian. If this option is
      True, Jacobian is either calculated using ``jax.jacfwd`` or obtained from
      ``jac_fun`` and all variables depending on it, such as J^T or J^T.J are
      obtained directly. If False, all of the Jacobian dependent variables are
      indirectly obtained using operators on the basis of ``jax.jvp`` and
      ``jax.vjp``.
    implicit_diff: bool, whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.
    has_aux: whether ``residual_fun`` outputs auxiliary data or not.
    jit: whether to JIT-compile the bisection loop (default: True).
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

  solver: Union[Literal['cholesky', 'lu', 'qr', 'inv'], Callable] = solve_cg
  geodesic: bool = False
  contribution_ratio_threshold = 0.75

  verbose: Union[bool, int] = False
  jac_fun: Optional[Callable[..., jnp.ndarray]] = None
  materialize_jac: bool = False
  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None
  has_aux: bool = False
  jit: bool = True
  unroll: base.AutoOrBoolean = 'auto'

  # We are overriding the _cond_fun of the base solver to enable stopping based
  # on gradient or delta_params
  def _cond_fun(self, inputs):
    params, state = inputs[0]
    if self.verbose:
      print_iteration(state)
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

    if self.materialize_jac:
      jac = self._jac_fun(init_params, *args, **kwargs)
      jt = jac.T
      jtj = jt @ jac
      gradient = jt @ residual
      damping_factor = self.damping_parameter * jnp.max(jnp.diag(jtj))
      if self.geodesic:
        hess_res = self._hess_res_fun(init_params, *args, **kwargs)
      else:
        hess_res = None
    else:
      jac = None
      jt = None
      jtj = None
      hess_res = None
      gradient = self._jt_op(init_params, residual, *args, **kwargs)
      jtj_diag = self._jtj_diag_op(init_params, *args, **kwargs)
      damping_factor = self.damping_parameter * jnp.max(jtj_diag)

    delta_params = jnp.zeros_like(init_params)

    return LevenbergMarquardtState(
        iter_num=jnp.asarray(0),
        damping_factor=damping_factor,
        increase_factor=2,
        error=tree_l2_norm(gradient),
        residual=residual,
        value=0.5 * jnp.sum(jnp.square(residual)),
        delta=delta_params,
        gradient=gradient,
        jac=jac,
        jt=jt,
        jtj=jtj,
        hess_res=hess_res,
        aux=aux)

  def update_state_using_gain_ratio(self, gain_ratio, contribution_ratio_diff,
                                    gain_ratio_test_init_state, *args,
                                    **kwargs):
    """The function to return state variables based on gain ratio.
      Please see by page 120-121 of the book "Introduction to Optimization and
      Data Fitting" by K. Madsen & H. B. Nielsen for details.
    """

    def gain_ratio_test_true_func(params, damping_factor,
                                  increase_factor, residual, gradient, jac, jt, jtj,
                                  hess_res, updated_params, aux):

      params = updated_params

      residual, aux = self._fun_with_aux(params, *args, **kwargs)

      # Calculate gradient based on Eq. 6.6 of "Introduction to optimization
      # and data fitting" g=JT * r, where J is jacobian and r is residual.
      # TODO: QR factorization solver doesn't require jt and jtj. Can be
      # skipped for increasing efficiency, if they are not required by user.
      if self.materialize_jac:
        # Calculate Jacobian and it's transpose based on the updated coeffs.
        jac = self._jac_fun(params, *args, **kwargs)
        jt = jac.T
        #  J^T.J is the gauss newton approximate hessian.
        jtj = jt @ jac
        gradient = jt @ residual
        if self.geodesic:
          hess_res = self._hess_res_fun(params, *args, **kwargs)
        else:
          hess_res = None
      else:
        jac = None
        jt = None
        jtj = None
        hess_res = None
        gradient = self._jt_op(params, residual, *args, **kwargs)

      damping_factor = damping_factor * jax.lax.max(1 / 3, 1 -
                                                    (2 * gain_ratio - 1)**3)
      increase_factor = 2

      return params, damping_factor, increase_factor, residual, gradient, jac, jt, jtj, hess_res, aux

    def gain_ratio_test_false_func(params, damping_factor,
                                   increase_factor, residual, gradient, jac, jt, jtj,
                                   hess_res, updated_params, aux):
      damping_factor = jnp.minimum(damping_factor * increase_factor, self.damping_factor_max)
      increase_factor = jnp.minimum(2 * increase_factor, self.increase_factor_max)
      return params, damping_factor, increase_factor, residual, gradient, jac, jt, jtj, hess_res, aux

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

    params, damping_factor, increase_factor, residual, gradient, jac, jt, jtj, hess_res, aux = jax.tree_map(
        lambda x, y: x + y, gain_ratio_test_is_met_ret,
        gain_ratio_test_not_met_ret)

    return params, damping_factor, increase_factor, residual, gradient, jac, jt, jtj, hess_res, aux

  def update_state_using_delta_params(self, loss_curr, params, delta_params,
                                      contribution_ratio_diff, damping_factor,
                                      increase_factor, residual, gradient, jac,
                                      jt, jtj, hess_res, aux, *args, **kwargs):
    """The function to return state variables based on delta_params.

    Define the functions required for the major conditional of the algorithm,
    which checks the magnitude of dparams and checks if it is small enough.
    for the value of dparams.
    """

    updated_params = params + delta_params

    residual_next = self._fun(updated_params, *args, **kwargs)

    # Calculate denominator of the gain ratio based on Eq. 6.16, "Introduction
    # to optimization and data fitting", L(0)-L(hlm)=0.5*hlm^T*(mu*hlm-g).
    gain_ratio_denom = 0.5 * delta_params.T @ (
        damping_factor * delta_params - gradient)

    # Current value of loss function F=0.5*||f||^2.
    loss_next = 0.5 * jnp.sum(jnp.square(residual_next))

    gain_ratio = (loss_curr - loss_next) / gain_ratio_denom

    gain_ratio_test_init_state = (params, damping_factor, increase_factor,
                                  residual, gradient, jac, jt, jtj, hess_res,
                                  updated_params, aux)

    # Calling the jax condition function:
    # Note that only the parameters that are used in the rest of the program
    # are returned by this function.

    params, damping_factor, increase_factor, residual, gradient, jac, jt, jtj, hess_res, aux = (
        self.update_state_using_gain_ratio(gain_ratio, contribution_ratio_diff,
                                           gain_ratio_test_init_state, *args,
                                           **kwargs))

    return params, damping_factor, increase_factor, residual, gradient, jac, jt, jtj, hess_res, aux

  def update(self, params, state: NamedTuple, *args, **kwargs) -> base.OptStep:
    """Performs one iteration of the least-squares solver.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.

    Returns:
      (params, state)
    """

    # Current value of the loss function F=0.5*||f||^2.
    loss_curr = state.value

    # TODO: clean up the linear_solver to take matrix directly and then clean up
    # linear equation solves below for materialize_jac=True to call them.

    # For geodesic acceleration, we calculate  jtrpp=JT * r",
    # where J is jacobian and r" is second order directional derivative.
    if self.materialize_jac:
      # Note that instead of taking the inverse of jtj_corr and multiply that
      # by state.gradient, we prefer to use `jsp.solve`, which uses Cholesky of
      # jtj_corr and uses that to obtain velocity. This has the advantage of
      # lower number of floating point operations and therefore less numerical
      # error which can be helpful for the case of single precision arithmatics.
      if self.solver == 'qr':
        damping_term = jnp.sqrt(state.damping_factor) * jnp.identity(params.size)
        aug_jac = jnp.vstack([state.jac, damping_term])
        aug_res = jnp.hstack([state.residual, jnp.zeros(params.size)])
        q, r = jsp.linalg.qr(aug_jac, mode='economic')
        velocity = jsp.linalg.solve_triangular(r, q.T @ aug_res)
      else:
        damping_term = state.damping_factor * jnp.identity(params.size)
        jtj_corr = state.jtj + damping_term
        if self.solver == 'lu':
          velocity = jnp.linalg.solve(jtj_corr, state.gradient)
        elif self.solver == 'inv':
          jtj_corr_inv = jnp.linalg.inv(jtj_corr)
          velocity = jnp.dot(jtj_corr_inv, state.gradient)
        else:
          velocity = jsp.linalg.solve(jtj_corr, state.gradient, assume_a='pos')
      delta_params = velocity
      if self.geodesic:
        rpp = (state.hess_res @ velocity) @ velocity
        # Note the same as above here that we could use inverse of jtj_corr but
        # chose to use solve for higher performance and lower numerical error.
        if self.solver == 'qr':
          aug_rpp = jnp.hstack([rpp, jnp.zeros(params.size)])
          acceleration = jsp.linalg.solve_triangular(r, q.T @ aug_rpp)
        elif self.solver == 'lu':
          acceleration = jnp.linalg.solve(jtj_corr, state.jt) @ rpp
        elif self.solver == 'inv':
          acceleration = jnp.dot(jtj_corr_inv, state.jt) @ rpp
        else:
          acceleration = jsp.linalg.solve(jtj_corr, state.jt, assume_a='pos') @ rpp

        delta_params += 0.5*acceleration
    else:
      matvec = lambda v: self._jtj_op(params, v, *args, **kwargs)

      velocity, acceleration, delta_params = self._solve_linear_eqs(
        matvec, state, params, *args, **kwargs
      )

    if self.geodesic:
      contribution_ratio_diff = jnp.linalg.norm(acceleration) / jnp.linalg.norm(
          velocity) - self.contribution_ratio_threshold
    else:
      contribution_ratio_diff = 0.0

    # Negative coefficient is due to the sign of the RHS vector in the update equation
    # (J^T @ J + µ I) @ ∆params = -J^T @ f(x).
    delta_params = -delta_params

    # Checking if the dparams satisfy the "sufficiently small" criteria.
    params, damping_factor, increase_factor, residual, gradient, jac, jt, jtj, hess_res, aux = (
        self.update_state_using_delta_params(loss_curr, params, delta_params,
                                             contribution_ratio_diff,
                                             state.damping_factor,
                                             state.increase_factor,
                                             state.residual, state.gradient,
                                             state.jac, state.jt, state.jtj,
                                             state.hess_res, state.aux, *args, **kwargs))

    new_value = 0.5 * jnp.sum(jnp.square(residual))
    state = LevenbergMarquardtState(
        iter_num=state.iter_num + 1,
        damping_factor=damping_factor,
        increase_factor=increase_factor,
        error=tree_l2_norm(gradient),
        residual=residual,
        value=new_value,
        delta=delta_params,
        gradient=gradient,
        jac=jac,
        jt=jt,
        jtj=jtj,
        hess_res=hess_res,
        aux=aux)

    if self.verbose:
      self.log_info(
          state,
          error_name="Gradient Norm",
          additional_info={
              "Objective Value": new_value,
              "Damping Factor": damping_factor
          }
      )
    return base.OptStep(params=params, state=state)

  def __post_init__(self):
    super().__post_init__()

    if isinstance(self.solver, Callable):
      self.solver_fn = self.solver
    elif self.solver == 'cholesky':
      self.solver_fn = solve_cholesky
    elif self.solver == 'lu':
      self.solver_fn = solve_lu
    elif self.solver == 'qr':
      self.solver_fn = solve_qr
    elif self.solver == 'inv':
      self.solver_fn = solve_inv
    else:
      raise NotImplementedError
    if self.has_aux:
      self._fun_with_aux = self.residual_fun
      self._fun = lambda *a, **kw: self._fun_with_aux(*a, **kw)[0]
    else:
      self._fun = self.residual_fun
      self._fun_with_aux = lambda *a, **kw: (self.residual_fun(*a, **kw), None)

    # Define maximum value for the damping factor and increase factor. This is
    # particularly helpful for X32 usage.
    self.damping_factor_max = jnp.asarray(2**32, dtype=float)
    self.increase_factor_max = jnp.asarray(2**32, dtype=float)

    # For geodesic acceleration, we define Hessian of the residual function.
    if self.materialize_jac:
      if self.jac_fun is None:
        self._jac_fun = jax.jacfwd(self._fun, argnums=(0))
        if self.geodesic:
          self._hess_res_fun = jax.jacfwd(
              jax.jacfwd(self._fun, argnums=(0)), argnums=(0))
    else:
      if self.jac_fun:
        self._jac_fun = self.jac_fun
        if self.geodesic:
          self._hess_res_fun = jax.jacfwd(self.jac_fun, argnums=(0))
      if self.solver == 'qr':
        raise ValueError("QR factorization solver materializes Jacobian to solve the augmented "
                         "Jacobian equation instead of the normal equation. Hence, this solver "
                         "choice is inconsistent with materialize_jac=False.")
      if self.solver in ['lu', 'cholesky', 'inv']:
        warnings.warn(f"The linear solver {self.solver} that requires materialization of "
                      "J^T.J matrix is used with materialize_jac=False, which may cause a "
                      "computational overhead. Consider using either a matrix-free iterative "
                      "solver such as cg or bicg or using materialize_jac=True.", category=UserWarning)

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
    diag_op = lambda v: v.T @ self._jtj_op(params, v, *args, **kwargs)
    return jax.vmap(diag_op)(jnp.eye(len(params))).T

  def _d2fvv_op(self, primals, tangents1, tangents2, *args, **kwargs):
    """Product with d2f.v1v2."""
    fun_with_args = lambda p: self._fun(p, *args, **kwargs)
    g = lambda pr: jax.jvp(fun_with_args, (pr,), (tangents1,))[1]
    return jax.jvp(g, (primals,), (tangents2,))[1]

  def _solve_linear_eqs(self, matvec, state, params, *args, **kwargs):
    """Solves the linear equations to obtain velocity and acceleration."""
    if isinstance(self.solver, Callable):
      solver_options = {'ridge': state.damping_factor, 'init': state.delta}
    else:
      solver_options = {'ridge': state.damping_factor}
    velocity = self.solver_fn(matvec, state.gradient, **solver_options)
    delta_params = velocity
    if self.geodesic:
      rpp = self._d2fvv_op(params, velocity, velocity, *args, **kwargs)
      jtrpp = self._jt_op(params, rpp, *args, **kwargs)
      acceleration = self.solver_fn(matvec, jtrpp, ridge=state.damping_factor)
      delta_params += 0.5*acceleration
    else:
      acceleration = jnp.zeros_like(velocity)

    return (velocity, acceleration, delta_params)


def print_iteration(state: LevenbergMarquardtState):
  jax.debug.print("Iteration: {iter}, Value: {value}, ||Gradient||: {error}, Damping Factor: {damp}",
                iter=state.iter_num, value=state.value, error=state.error, damp=state.damping_factor)
