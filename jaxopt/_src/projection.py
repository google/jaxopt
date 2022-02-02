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

from typing import Any
from typing import Tuple

import jax
import jax.numpy as jnp

from jaxopt._src.bisection import Bisection
from jaxopt._src.osqp import OSQP, BoxOSQP
from jaxopt._src.eq_qp import EqualityConstrainedQP
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
  return tree_util.tree_multimap(_clip_safe, x, lower, upper)


@jax.custom_jvp
def _projection_unit_simplex(x: jnp.ndarray) -> jnp.ndarray:
  """Projection onto the unit simplex."""
  s = 1.0
  n_features = x.shape[0]
  u = jnp.sort(x)[::-1]
  cssv = jnp.cumsum(u) - s
  ind = jnp.arange(n_features) + 1
  cond = u - cssv / ind > 0
  idx = jnp.count_nonzero(cond)
  threshold = cssv[idx - 1] / idx.astype(x.dtype)
  return jax.nn.relu(x - threshold)


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
  if check_feasible and state.status == BoxOSQP.PRIMAL_INFEASIBLE:
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
