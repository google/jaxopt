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

"""Limited-memory BFGS with box constraints."""

# This is based on:
# [1] R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
# Constrained Optimization, (1995), SIAM Journal on Scientific and Statistical
# Computing, 16, 5, pp. 1190-1208.
# [2] J. Nocedal and S. Wright.  Numerical Optimization, second edition.

import dataclasses
import inspect
import warnings
from typing import Any, Callable, NamedTuple, Optional, Union

import jax
from jax import numpy as jnp

from jaxopt._src import base
from jaxopt._src import projection
from jaxopt._src.lbfgs import init_history
from jaxopt._src.lbfgs import update_history
from jaxopt._src.linesearch_util import _init_stepsize
from jaxopt._src.linesearch_util import _setup_linesearch

from jaxopt._src.tree_util import tree_single_dtype
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_inf_norm
from jaxopt.tree_util import tree_map
from jaxopt.tree_util import tree_sub
from jaxopt.tree_util import tree_vdot

import numpy as np


def _flatten_and_concat(tree: Any, batch_ndims: int = 0):
  """Flattens a pytree and concatenates leaves along the last dimension."""
  r = jax.tree_util.tree_map(
      lambda x: jnp.reshape(x, jnp.array(x).shape[:batch_ndims] + (-1,)), tree
  )
  return jax.tree_util.tree_reduce(
      lambda *args: jnp.concatenate(args, axis=-1), r
  )


def _split_and_pack_like(x: jnp.ndarray, tree: Any):
  """Splits an array and packs the components like the provided pytree."""
  treedef = jax.tree_util.tree_structure(tree)
  if jax.tree_util.treedef_is_leaf(treedef):
    return jnp.reshape(x, tree.shape)
  sizes = jax.tree_util.tree_map(jnp.size, tree)
  flat_sizes, treedef = jax.tree_util.tree_flatten(sizes)
  splits = jnp.split(x, np.cumsum(flat_sizes)[:-1])
  return jax.tree_util.tree_map(lambda y, p: jnp.reshape(y, p.shape),
                                jax.tree_util.tree_unflatten(treedef, splits),
                                tree)


def _get_error(x: Any, grad: Any, lower: Any, upper: Any):
  """Computes the error for determining convergence."""
  err = tree_map(
      lambda y, g, l, u: jnp.clip(y - g, l, u) - y, x, grad, lower, upper
  )
  return tree_inf_norm(err)


def _find_cauchy_point(
    x: jnp.ndarray,
    grad: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    theta: jnp.ndarray,
    w: jnp.ndarray,
    m: jnp.ndarray,
):
  """Finds the Cauchy point.

  The Cauchy point is the first local minimizer of the quadratic model along the
  piecewise linear path obtained by projecting points along the steepest descent
  direction. This function implements Algorithm CP in [1].

  Args:
    x: Parameters.
    grad: Gradients with respect to parameters.
    lower: Parameter lower bounds.
    upper: Parameter upper bounds.
    theta: Scaling parameter on the identity matrix for the initial Hessian
      approximation.
    w: `W_k` matrix in equation 3.3 of [1], computed from the correction
      matrices.
    m: `M_k` matrix in equation 3.4 of [1], computed from the correction
      matrices.

  Returns:
    x_cauchy: The Cauchy point.
    c: Vector to initialize subspace minimization.
    active_set_mask: Boolean mask where `True` indicates that the coordinate is
      in the active set (not equal to the lower or upper bound).
  """
  # TODO(emilyaf, srvasude): Consider a cheaper Cauchy point approximation:
  # https://yunfei.work/lbfgsb/lbfgsb_tech_report.pdf
  eps = np.finfo(x.dtype).eps
  t = jnp.where(
      jnp.abs(grad) < eps, jnp.inf,
      jnp.where(grad < 0., (x - upper) / grad, (x - lower) / grad))
  d = jnp.where(t < eps, 0., -grad)
  x_bound = jnp.where(d > 0., upper, jnp.where(d < 0., lower, x))

  # Sort coordinates by the distance from the bounds, divided by the gradient.
  t_ind = jnp.argsort(t, axis=-1)
  t_sorted = t[t_ind]
  dt = jnp.diff(jnp.pad(t_sorted, (1, 0), "constant"))

  # Begin the loop at the first coordinate that is not at a bound.
  active_set_mask_sorted = t_sorted > eps
  start_ind = jnp.argmax(
      jnp.concatenate([active_set_mask_sorted,
                       jnp.ones([1]).astype(jnp.bool_)],
                      axis=0))

  init_c = jnp.zeros(m.shape[-1:], dtype=m.dtype)
  init_p = jnp.dot(w.T, d)
  init_df = -jnp.dot(d, d)
  init_ddf = -theta * init_df - jnp.dot(jnp.matmul(m, init_p), init_p)
  init_state = (start_ind, init_df, init_ddf, active_set_mask_sorted, init_c,
                init_p)

  def _cond(state):
    return (-state[1] / state[2] >= dt[state[0]]) & (state[0] < x.shape[-1])

  def _body(args):
    i, df, ddf, mask, c, p = args
    j = t_ind[i]  # index of the unsorted array
    c_new = c + dt[i] * p
    df_new = (
        df
        + dt[i] * ddf
        + grad[j] ** 2
        + theta * grad[j] * (x_bound[j] - x[j])
        - grad[j] * jnp.dot(w[j], jnp.matmul(m, c_new))
    )
    ddf_new = (
        ddf
        - theta * grad[j] ** 2
        - 2.0 * grad[j] * jnp.dot(w[j], jnp.matmul(m, p))
        - grad[j] ** 2 * jnp.dot(w[j], jnp.matmul(m, w[j]))
    )
    ddf_new = jnp.maximum(eps, ddf_new)
    p_new = p + grad[j] * w[j]
    return (i + 1, df_new, ddf_new, mask.at[i].set(False), c_new, p_new)

  i, df, ddf, mask_sorted, c, p = jax.lax.while_loop(_cond, _body, init_state)
  dt_min = jnp.maximum(-df / ddf, jnp.zeros([], dtype=m.dtype))
  dt_min = jnp.where(jnp.isnan(dt_min), jnp.zeros([], dtype=m.dtype), dt_min)
  t_old = (
      jax.lax.cond(
          i > 0, lambda: t_sorted[i - 1], lambda: jnp.zeros([], dtype=m.dtype)
      )
      + dt_min
  )
  active_set_mask = mask_sorted[jnp.argsort(t_ind)]
  x_cauchy = jnp.where(active_set_mask, x + t_old * d, x_bound)
  return x_cauchy, c + dt_min * p, active_set_mask


def _minimize_subspace(
    x, grad, lower, upper, x_cauchy, c, theta, w, m, active_set_mask
):
  """Direct primal method of subspace minimization from [1]."""
  w_masked = w * active_set_mask[:, jnp.newaxis].astype(w.dtype)
  r_c = (
      grad + theta * (x_cauchy - x) - jnp.dot(w_masked, jnp.matmul(m, c))
  )  # eq. 5.4

  # TODO(emilyaf): Implement the method from [1] for a large number of variables
  # and few active constraints.
  v = jnp.dot(w_masked.T, r_c)
  v = jnp.matmul(m, v)
  n = jnp.matmul(w_masked.T, w_masked) / theta
  n = jnp.eye(m.shape[-1], dtype=w.dtype) - jnp.matmul(m, n)
  v = jnp.linalg.solve(n, v)
  du = -r_c / theta - jnp.matmul(w_masked, v) / theta**2

  # TODO(emilyaf, srvasude): Investigate whether to instead truncate
  # `x_cauchy + alpha_star * du` at the boundary, following
  # https://dl.acm.org/doi/abs/10.1145/2049662.2049669.
  alpha = jnp.maximum((upper - x_cauchy) / du, (lower - x_cauchy) / du)
  alpha = jnp.where(active_set_mask & (jnp.abs(du) > 0.), alpha, 1.)
  alpha_star = jnp.min(alpha, axis=-1)
  alpha_star = jnp.minimum(alpha_star, 1.)
  return jnp.where(active_set_mask, x_cauchy + alpha_star * du, x_cauchy)


class LbfgsbState(NamedTuple):
  """Named tuple containing state information."""

  iter_num: int
  value: float
  grad: Any
  stepsize: float
  error: float
  s_history: Any
  y_history: Any
  theta: jnp.ndarray
  num_updates: jnp.ndarray
  aux: Optional[Any] = None
  failed_linesearch: bool = False

  num_fun_eval: int = 0
  num_grad_eval: int = 0
  num_linesearch_iter: int = 0


@dataclasses.dataclass(eq=False)
class LBFGSB(base.IterativeSolver):
  """L-BFGS-B solver.

  L-BFGS-B is a version of L-BFGS that incorporates box constraints on
  variables.

  Attributes:
    fun: a smooth function of the form ``fun(x, *args, **kwargs)``.
    value_and_grad: whether ``fun`` just returns the value (False) or both the
      value and gradient (True). See base.make_funs_with_aux for details.
    has_aux: whether ``fun`` outputs auxiliary data or not. If ``has_aux`` is
      False, ``fun`` is expected to be  scalar-valued. If ``has_aux`` is True,
      then we have one of the following two cases. If ``value_and_grad`` is
      False, the output should be ``value, aux = fun(...)``. If ``value_and_grad
      == True``, the output should be ``(value, aux), grad = fun(...)``. At each
      iteration of the algorithm, the auxiliary outputs are stored in
      ``state.aux``.
    maxiter: maximum number of proximal gradient descent iterations.
    tol: tolerance of the stopping criterion.
    stepsize: a stepsize to use (if <= 0, use backtracking line search), or a
      callable specifying the **positive** stepsize to use at each iteration.
    linesearch_init: strategy for line-search initialization. By default, it
      will use "increase", which will increased the step-size by a factor of
      `increase_factor` at each iteration if the step-size is larger than
      `min_stepsize`, and set it to `max_stepsize` otherwise. Other choices are
      "max", that initializes the step-size to `max_stepsize` at every
      iteration, and "current", that uses the step-size from the previous
      iteration.
    stop_if_linesearch_fails: whether to stop iterations if the line search
      fails. When True, this matches the behavior of core JAX.
    condition: Deprecated. Condition used to select the stepsize when using
      backtracking linesearch
    maxls: maximum number of iterations to use in the line search.
    decrease_factor: Deprecated. factor by which to decrease the stepsize during
      backtracking line search (default: 0.8).
    increase_factor: factor by which to increase the stepsize during line search
      (default: 1.5).
    max_stepsize: upper bound on stepsize.
    min_stepsize: lower bound on stepsize guess at start of each linesearch run.
    history_size: size of the memory to use.
    use_gamma: whether to initialize the Hessian approximation with gamma *
      theta, where gamma is chosen following equation (7.20) of 'Numerical
      Optimization' [2]. If use_gamma is set to False, theta is used as
      initialization.
    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.
    jit: whether to JIT-compile the optimization loop (default: True).
    unroll: whether to unroll the optimization loop (default: "auto").
    verbose: if set to True or 1 prints the information at each step of 
      the solver, if set to 2, print also the information of the linesearch.
  """

  fun: Callable  # pylint: disable=g-bare-generic
  value_and_grad: Union[bool, Callable] = False
  has_aux: bool = False

  maxiter: int = 50
  tol: float = 1e-3

  stepsize: Union[float, Callable[[Any], float]] = 0.0
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

  theta: float = 1.0
  history_size: int = 10
  use_gamma: bool = True

  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable[[Any], Any]] = None

  jit: bool = True
  unroll: base.AutoOrBoolean = "auto"

  verbose: Union[bool, int] = False

  def _cond_fun(self, inputs):
    _, state = inputs[0]
    # We continue the optimization loop while the error tolerance is not met
    # and either failed linesearch is disallowed or linesearch hasn't failed.
    return (state.error > self.tol) & jnp.logical_or(
        not self.stop_if_linesearch_fails, ~state.failed_linesearch)

  def init_state(
      self,
      init_params: Any,
      bounds: Optional[Any],
      *args,
      **kwargs) -> LbfgsbState:
    """Initialize the solver state.

    Args:
      init_params: pytree containing the initial parameters.
      bounds: an optional tuple `(lb, ub)` of pytrees with structure identical
        to `init_params`, representing box constraints.
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
          iter_num=init_params.state.iter_num,
          stepsize=init_params.state.stepsize,
          num_updates=init_params.state.num_updates,
          theta=init_params.state.theta,
      )
      init_params = init_params.params
      dtype = tree_single_dtype(init_params)
    else:
      dtype = tree_single_dtype(init_params)
      state_kwargs = dict(
          s_history=init_history(init_params, self.history_size),
          y_history=init_history(init_params, self.history_size),
          iter_num=jnp.asarray(0),
          stepsize=jnp.asarray(self.max_stepsize, dtype=dtype),
          num_updates=jnp.asarray(0),
          theta=jnp.asarray(self.theta, dtype=dtype),
      )

    (value, aux), grad = self._value_and_grad_with_aux(
        init_params, *args, **kwargs
    )
    if bounds is None:
      bounds = (tree_map(lambda x: -jnp.inf * jnp.ones_like(x), init_params),
                tree_map(lambda x: jnp.inf * jnp.ones_like(x), init_params))
    init_error = _get_error(init_params, grad, *bounds)
    return LbfgsbState(
        value=value,
        grad=grad,
        error=init_error,
        **state_kwargs,
        aux=aux,
        failed_linesearch=jnp.asarray(False),
        num_fun_eval=jnp.array(1, base.NUM_EVAL_DTYPE),
        num_grad_eval=jnp.array(1, base.NUM_EVAL_DTYPE),
        num_linesearch_iter=np.array(0, base.NUM_EVAL_DTYPE)
    )

  def update(
      self,
      params: Any,
      state: LbfgsbState,
      bounds: Optional[Any],
      *args,
      **kwargs) -> base.OptStep:
    """Performs one iteration of LBFGS.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      bounds: an optional tuple `(lb, ub)` of pytrees with structure identical
        to `init_params`, representing box constraints.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.

    Returns:
      (params, state)
    """
    if isinstance(params, base.OptStep):
      params = params.params
    if bounds is None:
      bounds = (tree_map(lambda x: -jnp.inf * jnp.ones_like(x), params),
                tree_map(lambda x: jnp.inf * jnp.ones_like(x), params))
    lower, upper = bounds
    x_flat = _flatten_and_concat(params)
    g_flat = _flatten_and_concat(state.grad)
    lower_flat = _flatten_and_concat(lower)
    upper_flat = _flatten_and_concat(upper)
    s_history_flat = _flatten_and_concat(state.s_history, batch_ndims=1)
    y_history_flat = _flatten_and_concat(state.y_history, batch_ndims=1)

    # Construct the limited-memory BFGS matrix from [1], section 3.
    w_flat = jnp.transpose(
        jnp.concatenate([y_history_flat, state.theta * s_history_flat], axis=0))
    s_dot_yt = jnp.matmul(s_history_flat, jnp.transpose(y_history_flat))

    # Pad extra history dimensions with constants on the diagonal to ensure
    # invertability while maintaining constant array sizes for JIT compilation.
    # The corresponding dimensions of the inverted matrix are multiplied by zero
    # downstream. If possible, the constants are chosen to equal an existing
    # nonzero element of the diagonal, for numerical stability.
    history_mask = jnp.arange(self.history_size) >= state.num_updates
    diagonal_ones = history_mask.astype(x_flat.dtype)
    i, j = jnp.diag_indices(self.history_size)
    prev_ind = state.num_updates % self.history_size - 1
    zeros = jnp.zeros_like(s_dot_yt)
    fill_diag_syt = jnp.where(state.num_updates > 0, s_dot_yt[prev_ind,
                                                              prev_ind], 1.)
    diag_adj = zeros.at[i, j].set(diagonal_ones * fill_diag_syt)

    # (m, m) tril with zeros on the diagonal
    lower_tril = jnp.tril(s_dot_yt, -1)

    # (m, m) diagonal
    diag = -1. * jnp.diag(jnp.diag(s_dot_yt)) + diag_adj

    s_dot_st = jnp.matmul(s_history_flat, jnp.transpose(s_history_flat))
    fill_diag_sst = jnp.where(state.num_updates > 0, s_dot_st[prev_ind,
                                                              prev_ind], 1.)
    sst_adj = zeros.at[i, j].set(diagonal_ones * fill_diag_sst) * state.theta
    m_inv = jnp.concatenate(  # Equation 3.4 of [1].
        [
            jnp.concatenate([diag, jnp.transpose(lower_tril)], axis=1),
            jnp.concatenate([lower_tril, sst_adj + state.theta * s_dot_st],
                            axis=1)
        ],
        axis=0)
    m = jnp.linalg.inv(m_inv)

    x_cauchy, c, active_ind = _find_cauchy_point(
        x_flat, g_flat, lower_flat, upper_flat, state.theta, w_flat, m
    )
    x_subspace_min = _minimize_subspace(
        x_flat,
        g_flat,
        lower_flat,
        upper_flat,
        x_cauchy,
        c,
        state.theta,
        w_flat,
        m,
        active_ind,
    )

    descent_direction = _split_and_pack_like(x_subspace_min - x_flat, params)

    use_linesearch = (not isinstance(self.stepsize, Callable) and
                      self.stepsize <= 0.)
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
          value=state.value,
          grad=state.grad,
          descent_direction=descent_direction,
          fun_args=args,
          fun_kwargs=kwargs,
      )
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
      new_params = tree_map(jnp.clip, new_params, lower, upper)

      (new_value, new_aux), new_grad = self._value_and_grad_with_aux(
          new_params, *args, **kwargs
      )
      new_num_grad_eval = state.num_grad_eval + 1
      new_num_fun_eval = state.num_fun_eval + 1
      new_num_linesearch_iter = state.num_linesearch_iter
      failed_linesearch = jnp.asarray(False)

    s = tree_sub(new_params, params)
    y = tree_sub(new_grad, state.grad)
    curvature = tree_vdot(y, s)

    if self.use_gamma:
      gamma_inv = tree_vdot(y, y) / curvature
    else:
      gamma_inv = jnp.ones([], dtype=curvature.dtype)

    history_ind = state.num_updates % self.history_size
    (new_s_history, new_y_history, new_theta, new_num_updates) = (
        jax.lax.cond(
            curvature > 0.0,
            lambda sh, yh: (  # pylint: disable=g-long-lambda
                update_history(sh, s, history_ind),
                update_history(yh, y, history_ind),
                gamma_inv * self.theta,
                state.num_updates + 1,
            ),
            lambda sh, yh: (  # pylint: disable=g-long-lambda
                sh,
                yh,
                state.theta,
                state.num_updates,
            ),
            state.s_history,
            state.y_history,
        )
    )

    error = _get_error(new_params, new_grad, lower, upper)
    new_state = LbfgsbState(
        iter_num=state.iter_num + 1,
        value=new_value,
        grad=new_grad,
        stepsize=new_stepsize,
        error=error,
        s_history=new_s_history,
        y_history=new_y_history,
        num_updates=new_num_updates,
        theta=new_theta,
        aux=new_aux,
        failed_linesearch=failed_linesearch,
        num_grad_eval=new_num_grad_eval,
        num_fun_eval=new_num_fun_eval,
        num_linesearch_iter=new_num_linesearch_iter,
    )

    if self.verbose:
      self.log_info(
          new_state,
          error_name="Projected Gradient Norm",
          additional_info={
              "Objective Value": new_value,
              "Stepsize": new_stepsize,
              "Number Linesearch Iterations": 
              new_state.num_linesearch_iter - state.num_linesearch_iter
          }
      )
    return base.OptStep(new_params, new_state)

  def _fixed_point_fun(self, sol, bounds, args, kwargs):
    step = tree_sub(sol, self._grad_fun(sol, *args, **kwargs))
    return projection.projection_box(step, bounds)

  def optimality_fun(self, sol, bounds, *args, **kwargs):
    """Optimality function mapping compatible with `@custom_root`."""
    if bounds is None:
      return self._value_and_grad_fun(sol, *args, **kwargs)[1]
    fp = self._fixed_point_fun(sol, bounds, args, kwargs)
    return tree_sub(fp, sol)

  def _value_and_grad_fun(self, params, *args, **kwargs):
    if isinstance(params, base.OptStep):
      params = params.params
    (value, _), grad = self._value_and_grad_with_aux(params, *args, **kwargs)
    return value, grad

  def _grad_fun(self, params, *args, **kwargs):
    return self._value_and_grad_fun(params, *args, **kwargs)[1]

  def __post_init__(self):
    super().__post_init__()

    _fun_with_aux, _, self._value_and_grad_with_aux = base._make_funs_with_aux(
        fun=self.fun,
        value_and_grad=self.value_and_grad,
        has_aux=self.has_aux,
    )

    # Sets up reference signature.
    fun = getattr(self.fun, "subfun", self.fun)
    signature = inspect.signature(fun)
    parameters = list(signature.parameters.values())
    new_param = inspect.Parameter(name="bounds",
                                  kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
    parameters.insert(1, new_param)
    self.reference_signature = inspect.Signature(parameters)


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
        verbose=int(self.verbose)-1,
    )

    self.run_ls = linesearch_solver.run

    if self.condition is not None:
      warnings.warn("Argument condition is deprecated", DeprecationWarning)
    if self.decrease_factor is not None:
      warnings.warn(
          "Argument decrease_factor is deprecated", DeprecationWarning
      )
