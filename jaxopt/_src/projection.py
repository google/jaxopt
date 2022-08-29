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

"""Projection operators."""

from functools import partial
from typing import Any
from typing import Callable
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from jaxopt._src.bisection import Bisection
from jaxopt._src.eq_qp import EqualityConstrainedQP
from jaxopt._src.lbfgs import LBFGS
from jaxopt._src.osqp import OSQP, BoxOSQP
from jaxopt._src import tree_util


def projection_non_negative(x: Any, hyperparams=None) -> Any:
  r"""Projection onto the non-negative orthant:

  .. math::

    \underset{p}{\text{argmin}} ~ ||x - p||_2^2 \quad
    \textrm{subject to} \quad p \ge 0

  Args:
    x: pytree to project.
    hyperparams: ignored.
  Returns:
    projected pytree, with the same structure as ``x``.
  """
  del hyperparams  # Not used.
  return tree_util.tree_map(jax.nn.relu, x)


def _clip_safe(x, lower, upper):
  return jnp.clip(jnp.asarray(x), lower, upper)


def projection_box(x: Any, hyperparams: Tuple) -> Any:
  r"""Projection onto box constraints:

  .. math::

    \underset{p}{\text{argmin}} ~ ||x - p||_2^2 \quad \textrm{subject to} \quad
    \text{lower} \le p \le \text{upper}

  Args:
    x: pytree to project.
    hyperparams: a tuple ``hyperparams = (lower, upper)``, where ``lower`` and
      ``upper`` can be either scalar values or pytrees of the same structure as
      ``x``.
  Returns:
    projected pytree, with the same structure as ``x``.
  """
  lower, upper = hyperparams
  return tree_util.tree_map(_clip_safe, x, lower, upper)


def projection_hypercube(x: Any, unit: float = 1.0) -> Any:
  r"""Projection onto the (unit) hypercube:

  .. math::

    \underset{p}{\text{argmin}} ~ ||x - p||_2^2 \quad \textrm{subject to} \quad
    0 \le p \le \text{unit}

  This is a convenience wrapper around
  :func:`projection_box <jaxopt.projection.projection_box>`.

  Args:
    x: pytree to project.
    unit: a float value, defaults to 1.0.
  Returns:
    projected pytree, with the same structure as ``x``.
  """
  return projection_box(x, (0.0, unit))


@jax.custom_jvp
def _projection_unit_simplex(x: jnp.ndarray) -> jnp.ndarray:
  """Projection onto the unit simplex."""
  s = 1.0
  n_features = x.shape[0]
  u = jnp.sort(x)[::-1]
  cumsum_u = jnp.cumsum(u)
  ind = jnp.arange(n_features) + 1
  cond = s / ind + (u - cumsum_u / ind) > 0
  idx = jnp.count_nonzero(cond)
  return jax.nn.relu(s / idx + (x - cumsum_u[idx - 1] / idx))


@_projection_unit_simplex.defjvp
def _projection_unit_simplex_jvp(primals, tangents):
  x, = primals
  x_dot, = tangents
  primal_out = _projection_unit_simplex(x)
  supp = primal_out > 0
  card = jnp.count_nonzero(supp)
  tangent_out = supp * x_dot - (jnp.dot(supp, x_dot) / card) * supp
  return primal_out, tangent_out


def projection_simplex(x: jnp.ndarray, value: float = 1.0) -> jnp.ndarray:
  r"""Projection onto a simplex:

  .. math::

    \underset{p}{\text{argmin}} ~ ||x - p||_2^2 \quad \textrm{subject to} \quad
    p \ge 0, p^\top 1 = \text{value}

  By default, the projection is onto the probability simplex.

  Args:
    x: vector to project, an array of shape (n,).
    value: value p should sum to (default: 1.0).
  Returns:
    projected vector, an array with the same shape as ``x``.
  """
  if value is None:
    value = 1.0
  return value * _projection_unit_simplex(x / value)


@partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def _projection_unit_sparse_simplex(
    x: jnp.ndarray, max_nz: int,
    use_approx_max_nz: bool = False) -> jnp.ndarray:
  """Projection onto the unit simplex with cardinality constraint (maximum number of non-zero elements)."""

  # Top max_nz values (in an decreasing order) and their corresponding indices
  if use_approx_max_nz:
    max_nz_values, max_nz_indices = jax.lax.approx_max_k(x, max_nz)
  else:
    max_nz_values, max_nz_indices = jax.lax.top_k(x, max_nz)

  # Projection the sorted top k values onto the unit simplex
  cumsum_max_nz_values = jnp.cumsum(max_nz_values)
  ind = jnp.arange(max_nz) + 1
  cond = 1 / ind + (max_nz_values - cumsum_max_nz_values / ind) > 0
  idx = jnp.count_nonzero(cond)
  max_nz_simplex_projection = jax.nn.relu(
      1 / idx + (max_nz_values - cumsum_max_nz_values[idx - 1] / idx))

  # Put the projection of max_nz_values to their original indices;
  # set all other indices zero.
  sparse_simplex_projection = jnp.sum(
      max_nz_simplex_projection[ :, jnp.newaxis] * jax.nn.one_hot(
          max_nz_indices, len(x), dtype=x.dtype), axis=0)

  return  sparse_simplex_projection

@_projection_unit_sparse_simplex.defjvp
def _projection_unit_sparse_simplex_jvp(
    max_nz, use_approx_max_nz, primals, tangents):
  x, = primals
  x_dot, = tangents
  primal_out = _projection_unit_sparse_simplex(x, max_nz, use_approx_max_nz)
  supp = primal_out > 0
  card = jnp.count_nonzero(supp)
  tangent_out = supp * x_dot - (jnp.dot(supp, x_dot) / card) * supp
  return primal_out, tangent_out


def projection_sparse_simplex(
    x: jnp.ndarray, max_nz: int,
    use_approx_max_nz: bool = False, value: float = 1.0) -> jnp.ndarray:
  r"""Projection onto the simplex with cardinality constraint (maximum number of non-zero elements).

  .. math::
    \underset{p}{\text{argmin}} ~ ||x - p||_2^2 \quad \textrm{subject to} \quad
    p \ge 0, p^\top 1 = \text{value}, ||p||_0 \le \text{max_nz}

  Args:
    x: vector to project, an array of shape (n,).
    max_nz: max nonzero values to keep
    use_approx_max_nz: when set to True, use `jax.lax.approx_max_k` to return
      max values and their indices in an approximate manner (default: False).
    value: value p should sum to (default: 1.0).
  Returns:
    projected vector, an array with the same shape as ``x``.

  References:
    Sparse projections onto the simplex
    Anastasios Kyrillidis, Stephen Becker, Volkan Cevher and, Christoph Koch
    ICML 2013
    https://arxiv.org/abs/1206.1529
  """
  if value is None:
    value = 1.0
  return value * _projection_unit_sparse_simplex(
      x / value, max_nz=max_nz, use_approx_max_nz=use_approx_max_nz)


def projection_l1_sphere(x: jnp.ndarray, value: float = 1.0) -> jnp.ndarray:
  r"""Projection onto the l1 sphere:

  .. math::

    \underset{y}{\text{argmin}} ~ ||x - y||_2^2 \quad \textrm{subject to} \quad
    ||y||_1 = \text{value}

  Args:
    x: array to project.
    value: radius of the sphere.

  Returns:
    output array, with the same shape as ``x``.
  """
  return jnp.sign(x) * projection_simplex(jnp.abs(x), value)


def projection_l1_ball(x: jnp.ndarray, max_value: float = 1.0) -> jnp.ndarray:
  r"""Projection onto the l1 ball:

  .. math::

    \underset{y}{\text{argmin}} ~ ||x - y||_2^2 \quad \textrm{subject to} \quad
    ||y||_1 \le \text{max_value}

  Args:
    x: array to project.
    max_value: radius of the ball.

  Returns:
    output array, with the same structure as ``x``.
  """
  l1_norm = jax.numpy.linalg.norm(x, ord=1)
  return jax.lax.cond(l1_norm <= max_value,
                      lambda _: x,
                      lambda _: projection_l1_sphere(x, max_value),
                      operand=None)


def projection_l2_sphere(x: Any, value: float = 1.0) -> Any:
  r"""Projection onto the l2 sphere:

  .. math::

    \underset{y}{\text{argmin}} ~ ||x - y||_2^2 \quad \textrm{subject to} \quad
    ||y||_2 = \text{value}

  Args:
    x: pytree to project.
    value: radius of the sphere.

  Returns:
    output pytree, with the same structure as ``x``.
  """
  factor = value / tree_util.tree_l2_norm(x)
  return tree_util.tree_scalar_mul(factor, x)


def projection_l2_ball(x: Any, max_value: float = 1.0) -> Any:
  r"""Projection onto the l2 ball:

  .. math::

    \underset{y}{\text{argmin}} ~ ||x - y||_2^2 \quad \textrm{subject to} \quad
    ||y||_2 \le \text{max_value}

  Args:
    x: pytree to project.
    max_value: radius of the ball.

  Returns:
    output pytree, with the same structure as ``x``.
  """
  l2_norm = tree_util.tree_l2_norm(x)
  factor = max_value / l2_norm
  return jax.lax.cond(l2_norm <= max_value,
                      lambda _: x,
                      lambda _: tree_util.tree_scalar_mul(factor, x),
                      operand=None)


def projection_linf_ball(x: Any, max_value: float = 1.0) -> Any:
  r"""Projection onto the l-infinity ball:

  .. math::

    \underset{y}{\text{argmin}} ~ ||x - y||_2^2 \quad \textrm{subject to} \quad
    ||y||_{\infty} \le \text{max_value}

  Args:
    x: pytree to project.
    max_value: radius of the ball.

  Returns:
    output pytree, with the same structure as ``x``.
  """
  return projection_box(x, (-max_value, max_value))


def projection_hyperplane(x: jnp.ndarray, hyperparams: Tuple) -> jnp.ndarray:
  r"""Projection onto a hyperplane:

  .. math::

    \underset{y}{\text{argmin}} ~ ||x - y||_2^2 \quad \textrm{subject to} \quad
    a^\top y = b

  Args:
    x: pytree to project.
    hyperparams: tuple ``hyperparams = (a, b)``, where ``a`` is a pytree with
                 the same structure as ``x`` and ``b`` is a scalar.

  Returns:
    output array, with the same shape as ``x`.
  """
  a, b = hyperparams
  scale = (tree_util.tree_vdot(a, x) -b) / tree_util.tree_vdot(a, a)
  return tree_util.tree_add_scalar_mul(x, -scale, a)


def projection_halfspace(x: jnp.ndarray, hyperparams: Tuple) -> jnp.ndarray:
  r"""Projection onto a halfspace:

  .. math::

    \underset{y}{\text{argmin}} ~ ||x - y||_2^2 \quad \textrm{subject to} \quad
    a^\top y \le b

  Args:
    x: pytree to project.
    hyperparams: tuple ``hyperparams = (a, b)``, where ``a`` is a pytree with
                 the same structure as ``x`` and ``b`` is a scalar.

  Returns:
    output array, with same shape as ``x``.
  """
  a, b = hyperparams
  scale = jax.nn.relu(tree_util.tree_vdot(a, x) -b) / tree_util.tree_vdot(a, a)
  return tree_util.tree_add_scalar_mul(x, -scale, a)


def projection_affine_set(x: jnp.ndarray, hyperparams: Tuple) -> jnp.ndarray:
  r"""Projection onto an affine set:

  .. math::

    \underset{y}{\text{argmin}} ~ ||x - y||_2^2 \quad \textrm{subject to} \quad
    A y = b

  Args:
    x: array to project.
    hyperparams: tuple ``hyperparams = (A, b)``, where ``A`` is a matrix and
      ``b`` is a vector.

  Returns:
    output array, with the same shape as ``x``.
  """
  # TODO: support matvec for A
  A, b = hyperparams
  matvec_Q = lambda _, vec: vec
  osqp = EqualityConstrainedQP(matvec_Q=matvec_Q)
  hyperparams = dict(params_obj=(None, -x), params_eq=(A, b))
  kkt_sol = osqp.run(**hyperparams).params
  return kkt_sol.primal


def projection_polyhedron(x: jnp.ndarray, hyperparams: Tuple,
                          check_feasible=True) -> jnp.ndarray:
  r"""Projection onto a polyhedron:

  .. math::

    \underset{y}{\text{argmin}} ~ ||x - y||_2^2 \quad \textrm{subject to} \quad
    A y = b, G y \le h

  Args:
    x: pytree to project.
    hyperparams: tuple ``hyperparams = (A, b, G, h)``, where ``A`` is a matrix,
      ``b`` is a vector, ``G`` is a matrix and ``h`` is a vector.
    check_feasible: if True (default: True) check the non emptyness of the polyhedron,
      which disables jit compilation. If False, the function is jit compiled.

  Returns:
    output array, with the same shape as ``x``.
  """
  # todo: support matvecs for A and G
  A, b, G, h = hyperparams
  matvec_Q = lambda _, vec: vec
  osqp = OSQP(matvec_Q=matvec_Q, check_primal_dual_infeasability=check_feasible)
  # check feasability by default; currently there is no way to return the info to the user inside @jit.
  hyperparams = dict(params_obj=(None, -x), params_eq=(A, b), params_ineq=(G, h))
  kkt_sol, state = osqp.run(**hyperparams)
  if check_feasible and state.status in [BoxOSQP.PRIMAL_INFEASIBLE, BoxOSQP.DUAL_INFEASIBLE]:
    raise ValueError("The polyhedron is empty.")
  return kkt_sol.primal


def _optimality_fun_proj_box_sec(tau, x, hyperparams):
  # An optimal solution has the form
  # p_i = clip(w_i * tau + x_i, alpha_i, beta_i) for all i
  # where tau is the root of fun(tau, hyperparams) = dot(w, p) - c = 0.
  alpha, beta, w, c = hyperparams
  p = jnp.clip(w * tau + x, alpha, beta)
  return jnp.dot(w, p) - c


def _root_proj_box_sec(x, hyperparams):
  alpha, beta, w, _ = hyperparams
  lower = jax.lax.stop_gradient(jnp.min((alpha - x) / w))
  upper = jax.lax.stop_gradient(jnp.max((beta - x) / w))
  bisect = Bisection(optimality_fun=_optimality_fun_proj_box_sec,
                     lower=lower,
                     upper=upper,
                     check_bracket=False)
  return bisect.run(x=x, hyperparams=hyperparams).params


def projection_box_section(x: jnp.ndarray,
                           hyperparams: Tuple,
                           check_feasible: bool = False):
  r"""Projection onto a box section:

  .. math::

    \underset{p}{\text{argmin}} ~ ||x - p||_2^2 \quad \textrm{subject to} \quad
    \alpha \le p \le \beta, p^\top w = c

  The problem is infeasible if :math:`w^\top \alpha > c` or
  if :math:`w^\top \beta < c`.

  Args:
    x: vector to project, an array of shape (n,).
    hyperparams: tuple of parameters, ``hyperparams = (alpha, beta, w, c)``,
      where ``w``, ``alpha`` and ``beta`` are of the same shape as ``x``,
      and ``c`` is a scalar. Moreover, ``w`` should be positive.
    check_feasible: whether to check feasibility or not.
      If True, function cannot be jitted.
  Returns:
    projected vector, an array with the same shape as ``x``.
  """

  alpha, beta, w, c = hyperparams

  if check_feasible:
    if jnp.dot(w, alpha) > c:
      raise ValueError("alpha should satisfy dot(w, alpha) <= c")

    if jnp.dot(w, beta) < c:
      raise ValueError("beta should satisfy dot(w, beta) >= c")

  return jnp.clip(w * _root_proj_box_sec(x, hyperparams) + x, alpha, beta)


def _max_l2(x, marginal_b, gamma):
  scale = gamma * marginal_b
  x_scale = x / scale
  p = projection_simplex(x_scale)
  # From Danskin's theorem, we do not need to backpropagate
  # through projection_simplex.
  p = jax.lax.stop_gradient(p)
  return jnp.dot(x, p) - 0.5 * scale * jnp.dot(p, p)


def _max_ent(x, marginal_b, gamma):
  return gamma * logsumexp(x / gamma) - gamma * jnp.log(marginal_b)


_max_l2_vmap = jax.vmap(_max_l2, in_axes=(1, 0, None))
_max_l2_grad_vmap = jax.vmap(jax.grad(_max_l2), in_axes=(1, 0, None))

_max_ent_vmap = jax.vmap(_max_ent, in_axes=(1, 0, None))
_max_ent_grad_vmap = jax.vmap(jax.grad(_max_ent), in_axes=(1, 0, None))


def _delta_l2(x, gamma=1.0):
  # Solution to Eqn. (6) in https://arxiv.org/abs/1710.06276 with squared l2
  # regularization (see Table 1 in the paper).
  z = (0.5 / gamma) * jnp.dot(jax.nn.relu(x), jax.nn.relu(x))
  return z


def _delta_ent(x, gamma):
  # Solution to Eqn. (6) in https://arxiv.org/abs/1710.06276 with negative
  # entropy regularization.
  return gamma * jnp.exp((x / gamma) - 1).sum()

_delta_l2_vmap = jax.vmap(_delta_l2, in_axes=(1, None))
_delta_l2_grad_vmap = jax.vmap(jax.grad(_delta_l2), in_axes=(1, None))

_delta_ent_vmap = jax.vmap(_delta_ent, in_axes=(1, None))
_delta_ent_grad_vmap = jax.vmap(jax.grad(_delta_ent), in_axes=(1, None))


def _make_semi_dual(max_vmap, gamma=1.0):
  # Semi-dual objective, see equation (10) in
  # https://arxiv.org/abs/1710.06276
  def fun(alpha, cost_matrix, marginals_a, marginals_b):
    X = alpha[:, jnp.newaxis] - cost_matrix
    ret = jnp.dot(marginals_b, max_vmap(X, marginals_b, gamma))
    ret -= jnp.dot(alpha, marginals_a)
    return ret
  return fun


def _make_dual(delta_vmap, gamma):
  r"""Make the objective function of dual variables.

  Args:
    delta_vmap: The smoothed version of delta function, acting on each column of
      its matrix-valued input.
    gamma: A regularization constant.
  Returns:
    A cost function of dual variables. Cf. Equation (7) in
      https://arxiv.org/abs/1710.06276
  """

  def fun(alpha_beta, cost_matrix, marginals_a, marginals_b):
    alpha, beta = alpha_beta
    alpha_column = alpha[:, jnp.newaxis]
    beta_row = beta[jnp.newaxis, :]
    # Make a dual constraint matrix, whose (i,j)-th entry is
    # alpha[i] + beta[j] - c[i,j]. JAXopt solvers minimize functions hence
    # the sign is the opposite of Eqn (7)"
    dual_constraint_matrix = alpha_column + beta_row - cost_matrix
    delta_dual_constraints = delta_vmap(dual_constraint_matrix, gamma)
    dual_loss = delta_dual_constraints.sum() - jnp.dot(
        alpha, marginals_a) - jnp.dot(beta, marginals_b)
    return dual_loss
  return fun


def _regularized_transport_semi_dual(cost_matrix,
                           marginals_a,
                           marginals_b,
                           make_solver,
                           max_vmap,
                           max_grad_vmap,
                           gamma=1.0):

  r"""Regularized transport in the semi-dual formulation.

  Args:
    cost_matrix: The cost matrix of size (m, n).
    marginals_a: The marginals of size (m,)
    marginals_b: The marginals of size (n,)
    make_solver: A function that makes the optimization algorithm
    max_vmap:  A function that computes the regularized max on columns of
      its matrix-valued input
    max_grad_vmap:  A function that computes gradient of regularized max
      on columns of its matrix-valued input
    gamma: A parameter that controls the strength of regularization.

  Returns:
    The optimized plan. See the text under Eqn. (10) of
      https://arxiv.org/abs/1710.06276
  """
  size_a, size_b = cost_matrix.shape

  if len(marginals_a.shape) >= 2:
    raise ValueError("marginals_a should be a vector.")

  if len(marginals_b.shape) >= 2:
    raise ValueError("marginals_b should be a vector.")

  if size_a != marginals_a.shape[0] or size_b != marginals_b.shape[0]:
    raise ValueError("cost_matrix and marginals must have matching shapes.")

  if make_solver is None:
    make_solver = lambda fun: LBFGS(fun=fun, tol=1e-3, maxiter=500,
                                    linesearch="zoom")

  semi_dual = _make_semi_dual(max_vmap, gamma=gamma)
  solver = make_solver(semi_dual)
  alpha_init = jnp.zeros(size_a)

  # Optimal dual potentials.
  alpha = solver.run(alpha_init,
                     cost_matrix=cost_matrix,
                     marginals_a=marginals_a,
                     marginals_b=marginals_b).params

  # Optimal primal transportation plan.
  X = alpha[:, jnp.newaxis] - cost_matrix
  P = max_grad_vmap(X, marginals_b, gamma).T * marginals_b

  return P


def _regularized_transport_dual(cost_matrix,
                                marginals_a,
                                marginals_b,
                                make_solver,
                                delta_vmap,
                                delta_grad_vmap,
                                gamma=1.0):
  r"""Regularized transport in the dual formulation.

  Args:
    cost_matrix: The cost matrix of size (m, n).
    marginals_a: The marginals of size (m,)
    marginals_b: The marginals of size (n,)
    make_solver: A function that makes the optimization algorithm
    delta_vmap:  A function that computes the regularized delta on columns of
      its matrix-valued input
    delta_grad_vmap:  A function that computes gradient of regularized delta
      on columns of its matrix-valued input
    gamma: A parameter that controls the strength of regularization.

  Returns:
    The optimized plan. See the text under Eqn. (7) of
      https://arxiv.org/abs/1710.06276

  """

  size_a, size_b = cost_matrix.shape

  if len(marginals_a.shape) >= 2:
    raise ValueError("marginals_a should be a vector.")

  if len(marginals_b.shape) >= 2:
    raise ValueError("marginals_b should be a vector.")

  if size_a != marginals_a.shape[0] or size_b != marginals_b.shape[0]:
    raise ValueError("cost_matrix and marginals must have matching shapes.")

  if make_solver is None:
    make_solver = lambda fun: LBFGS(fun=fun, tol=1e-3, maxiter=500,
                                    linesearch="zoom")

  dual = _make_dual(delta_vmap, gamma=gamma)
  solver = make_solver(dual)
  alpha_beta_init = (jnp.zeros(size_a), jnp.zeros(size_b))

  # Optimal dual potentials.
  alpha_beta = solver.run(init_params=alpha_beta_init,
                          cost_matrix=cost_matrix,
                          marginals_a=marginals_a,
                          marginals_b=marginals_b).params

  # Optimal primal transportation plan.
  alpha, beta = alpha_beta
  alpha_column = alpha[:, jnp.newaxis]
  beta_row = beta[jnp.newaxis, :]
  # The (i,j)-th entry of dual_constraint_matrix is alpha[i] + beta[j] - c[i,j].
  dual_constraint_matrix = alpha_column + beta_row - cost_matrix
  plan = delta_grad_vmap(dual_constraint_matrix, gamma).T
  return plan


def projection_transport(sim_matrix: jnp.ndarray,
                         marginals: Tuple,
                         make_solver: Callable = None,
                         use_semi_dual: bool = True):
  r"""Projection onto the transportation polytope.

  We solve

  .. math::

    \underset{P \ge 0}{\text{argmin}} ~ ||P - S||_2^2 \quad
    \textrm{subject to} \quad P^\top \mathbf{1} = a, P \mathbf{1} = b

  or equivalently

  .. math::

    \underset{P \ge 0}{\text{argmin}} ~ \langle P, C \rangle
    + \frac{1}{2} \|P\|_2^2 \quad
    \textrm{subject to} \quad P^\top \mathbf{1} = a, P \mathbf{1} = b

  where :math:`S` is a similarity matrix, :math:`C` is a cost matrix
  and :math:`S = -C`.

  This implementation solves the semi-dual (see equation 10 in reference below)
  using LBFGS but the solver can be overidden using the ``make_solver`` option.

  For an entropy-regularized version, see
  :func:`kl_projection_transport <jaxopt.projection.kl_projection_transport>`.

  Args:
    sim_matrix: similarity matrix, shape=(size_a, size_b).
    marginals: a tuple (marginals_a, marginals_b),
      where marginals_a has shape=(size_a,) and
      marginals_b has shape=(size_b,).
    make_solver: a function of the form make_solver(fun),
      for creating an iterative solver to minimize fun.
    use_semi_dual: if true, use the semi-dual formulation in
      Equation (10) of https://arxiv.org/abs/1710.06276. Otherwise, use
      the dual-formulation in Equation (7).

  Returns:
    plan: transportation matrix, shape=(size_a, size_b).
  References:
    Smooth and Sparse Optimal Transport.
    Mathieu Blondel, Vivien Seguy, Antoine Rolet.
    In Proceedings of Artificial Intelligence and Statistics (AISTATS), 2018.
    https://arxiv.org/abs/1710.06276
  """
  marginals_a, marginals_b = marginals

  if use_semi_dual:
    plan = _regularized_transport_semi_dual(cost_matrix=-sim_matrix,
                                  marginals_a=marginals_a,
                                  marginals_b=marginals_b,
                                  make_solver=make_solver,
                                  max_vmap=_max_l2_vmap,
                                  max_grad_vmap=_max_l2_grad_vmap)
  else:
    plan = _regularized_transport_dual(cost_matrix=-sim_matrix,
                                       marginals_a=marginals_a,
                                       marginals_b=marginals_b,
                                       make_solver=make_solver,
                                       delta_vmap=_delta_l2_vmap,
                                       delta_grad_vmap=_delta_l2_grad_vmap)
  return plan


def kl_projection_transport(sim_matrix: jnp.ndarray,
                            marginals: Tuple,
                            make_solver: Callable = None,
                            use_semi_dual: bool = True):

  r"""Kullback-Leibler projection onto the transportation polytope.

  We solve

  .. math::
    \underset{P > 0}{\text{argmin}} ~ \text{KL}(P, \exp(S)) \quad
    \textrm{subject to} \quad P^\top \mathbf{1} = a, P \mathbf{1} = b

  or equivalently

  .. math::

    \underset{P > 0}{\text{argmin}} ~ \langle P, C \rangle
    + \langle P, \log P \rangle \quad
    \textrm{subject to} \quad P^\top \mathbf{1} = a, P \mathbf{1} = b

  where :math:`S` is a similarity matrix, :math:`C` is a cost matrix
  and :math:`S = -C`.

  This implementation solves the semi-dual (see equation 10 in reference below)
  using LBFGS but the solver can be overidden using the ``make_solver`` option.

  For an l2-regularized version, see
  :func:`projection_transport <jaxopt.projection.projection_transport>`.

  Args:
    sim_matrix: similarity matrix, shape=(size_a, size_b).
    marginals: a tuple (marginals_a, marginals_b),
      where marginals_a has shape=(size_a,) and
      marginals_b has shape=(size_b,).
    make_solver: a function of the form make_solver(fun),
      for creating an iterative solver to minimize fun.
    use_semi_dual: if true, use the semi-dual formulation in
      Equation (10) of https://arxiv.org/abs/1710.06276. Otherwise, use
      the dual-formulation in Equation (7).
  Returns:
    plan: transportation matrix, shape=(size_a, size_b).
  References:
    Smooth and Sparse Optimal Transport.
    Mathieu Blondel, Vivien Seguy, Antoine Rolet.
    In Proceedings of Artificial Intelligence and Statistics (AISTATS), 2018.
    https://arxiv.org/abs/1710.06276
  """
  marginals_a, marginals_b = marginals

  if use_semi_dual:
    plan = _regularized_transport_semi_dual(
        cost_matrix=-sim_matrix,
        marginals_a=marginals_a,
        marginals_b=marginals_b,
        make_solver=make_solver,
        max_vmap=_max_ent_vmap,
        max_grad_vmap=_max_ent_grad_vmap)
  else:
    plan = _regularized_transport_dual(
        cost_matrix=-sim_matrix,
        marginals_a=marginals_a,
        marginals_b=marginals_b,
        make_solver=make_solver,
        delta_vmap=_delta_ent_vmap,
        delta_grad_vmap=_delta_ent_grad_vmap)
  return plan


def projection_birkhoff(sim_matrix: jnp.ndarray,
                        make_solver: Callable = None,
                        use_semi_dual: bool = True):

  r"""Projection onto the Birkhoff polytope, the set of doubly stochastic
  matrices.

  This function is a special case of
  :func:`projection_transport <jaxopt.projection.projection_transport>`.

  Args:
    sim_matrix: similarity matrix, shape=(size, size).
    make_solver: a function of the form make_solver(fun),
      for creating an iterative solver to minimize fun.
    use_semi_dual: if true, use the semi-dual formulation in
      Equation (10) of https://arxiv.org/abs/1710.06276. Otherwise, use
      the dual-formulation in Equation (7).
  Returns:
    P: doubly-stochastic matrix, shape=(size, size).
  """
  marginals_a = jnp.ones(sim_matrix.shape[0])
  marginals_b = jnp.ones(sim_matrix.shape[1])
  return projection_transport(sim_matrix=sim_matrix,
                              marginals=(marginals_a, marginals_b),
                              make_solver=make_solver,
                              use_semi_dual=use_semi_dual)


def kl_projection_birkhoff(sim_matrix: jnp.ndarray,
                           make_solver: Callable = None,
                           use_semi_dual: bool = True):

  r"""Kullback-Leibler projection onto the Birkhoff polytope,
  the set of doubly stochastic matrices.

  This function is a special case of
  :func:`kl_projection_transport <jaxopt.projection.kl_projection_transport>`.

  Args:
    sim_matrix: similarity matrix, shape=(size, size).
    make_solver: a function of the form make_solver(fun),
      for creating an iterative solver to minimize fun.
    use_semi_dual: if true, use the semi-dual formulation in
      Equation (10) of https://arxiv.org/abs/1710.06276. Otherwise, use
      the dual-formulation in Equation (7).
  Returns:
    P: doubly-stochastic matrix, shape=(size, size).
  """
  marginals_a = jnp.ones(sim_matrix.shape[0])
  marginals_b = jnp.ones(sim_matrix.shape[1])
  return kl_projection_transport(sim_matrix=sim_matrix,
                                 marginals=(marginals_a, marginals_b),
                                 make_solver=make_solver,
                                 use_semi_dual=use_semi_dual)
