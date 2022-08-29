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

from absl.testing import absltest

import jax
import jax.numpy as jnp

from jaxopt import projection
from jaxopt._src import test_util
from jaxopt._src.lbfgs import LBFGS
from jaxopt import OSQP

import numpy as onp


class ProjectionTest(test_util.JaxoptTestCase):

  def test_projection_non_negative(self):
    x = jnp.array([-1.0, 2.0, 3.0])
    expected = jnp.array([0, 2.0, 3.0])
    self.assertArraysEqual(projection.projection_non_negative(x), expected)
    self.assertArraysEqual(projection.projection_non_negative((x, x)),
                           (expected, expected))
    # test with nested pytree
    tree_x = (-1.0, {"k1": 1.0, "k2": (1.0, 1.0)}, 1.0)
    tree_expected = (0.0, {"k1": 1.0, "k2": (1.0, 1.0)}, 1.0)
    self.assertAllClose(projection.projection_non_negative(tree_x),
                        tree_expected)

  def test_projection_box(self):
    x = jnp.array([-1.0, 2.0, 3.0])
    expected = jnp.array([0, 2.0, 2.0])

    L, U = 0.0, 2.0

    # lower and upper values are scalars.
    self.assertArraysEqual(projection.projection_box(x, (L, U)), expected)
    params = ((L, L), (U, U))
    self.assertArraysEqual(projection.projection_box((x, x), params),
                           (expected, expected))

    # lower and upper are pytrees. Note: Does not work when U and L are scalars!
    tree_x = (-1.0, {"k1": 2.0, "k2": (2.0, 3.0)}, 3.0)
    tree_expected = (0.0, {"k1": 2.0, "k2": (2.0, 2.0)}, 2.0)
    U_tree = (2.0, {"k1": 2.0, "k2": (2.0, 2.0)}, 2.0)
    L_tree = (0.0, {"k1": 0.0, "k2": (0.0, 0.0)}, 0.0)
    self.assertArraysEqual(projection.projection_box(tree_x, [L_tree, U_tree]),
                           tree_expected)

    # lower and upper values are arrays.
    params = (jnp.ones(len(x)) * L, jnp.ones(len(x)) * U)
    self.assertArraysEqual(projection.projection_box(x, params), expected)
    params = ((params[0], params[0]), (params[1], params[1]))
    self.assertArraysEqual(projection.projection_box((x, x), params),
                           (expected, expected))

  def test_projection_hypercube(self):
    x = jnp.array([-1.0, 2.0, 0.5])

    expected = jnp.array([0, 1.0, 0.5])
    self.assertArraysEqual(projection.projection_hypercube(x), expected)

    expected = jnp.array([0, 0.8, 0.5])
    self.assertArraysEqual(projection.projection_hypercube(x, 0.8), expected)


  def test_projection_simplex(self):
    rng = onp.random.RandomState(0)

    for scale in [1.0, 1e9]:
      for _ in range(10):
        x = scale * rng.randn(50).astype(onp.float32)

        p = projection.projection_simplex(x)
        self.assertAllClose(jnp.sum(p), 1.0)
        self.assertTrue(jnp.all(0 <= p))
        self.assertTrue(jnp.all(p <= 1))

        p = projection.projection_simplex(x, 0.8)
        self.assertAllClose(jnp.sum(p), 0.8)
        self.assertTrue(jnp.all(0 <= p))
        self.assertTrue(jnp.all(p <= 0.8))

    # Check that -inf is supported.
    p = projection.projection_simplex(jnp.array([0.0, 0.0, -jnp.inf]))
    self.assertAllClose(p, jnp.array([0.5, 0.5, 0.0]))

  def test_projection_simplex_jacobian(self):
    rng = onp.random.RandomState(0)

    for scale in [1.0, 1e9]:
      x = scale * rng.rand(5).astype(onp.float32)
      v = scale * rng.randn(5).astype(onp.float32)

      J_rev = jax.jacrev(projection.projection_simplex)(x)
      J_fwd = jax.jacfwd(projection.projection_simplex)(x)

      # Check against theoretical expression.
      p = projection.projection_simplex(x)
      support = (p > 0).astype(jnp.int32)
      cardinality = jnp.count_nonzero(support)
      J_true = jnp.diag(support) - jnp.outer(support, support) / cardinality
      self.assertArraysAllClose(J_true, J_fwd)
      self.assertArraysAllClose(J_true, J_rev)

      # Check vector Jacobian product.
      vJ, = jax.vjp(projection.projection_simplex, x)[1](v)
      self.assertArraysAllClose(vJ, jnp.dot(v, J_true))

      # Check vector Jacobian product.
      Jv = jax.jvp(projection.projection_simplex, (x,), (v,))[1]
      self.assertArraysAllClose(Jv, jnp.dot(J_true, v))

  def test_projection_simplex_vmap(self):
    rng = onp.random.RandomState(0)
    X = rng.randn(3, 50).astype(onp.float32)

    # Check with default s=1.0.
    P = jax.vmap(projection.projection_simplex)(X)
    self.assertArraysAllClose(jnp.sum(P, axis=1), jnp.ones(len(X)))
    self.assertTrue(jnp.all(0 <= P))
    self.assertTrue(jnp.all(P <= 1))

    # Check with s=0.8.
    P = jax.vmap(projection.projection_simplex)(X, 0.8 * jnp.ones(len(X)))
    self.assertArraysAllClose(jnp.sum(P, axis=1), 0.8 * jnp.ones(len(X)))
    self.assertTrue(jnp.all(0 <= P))
    self.assertTrue(jnp.all(P <= 0.8))

  def test_projection_simplex_vmap_diff(self):
    proj = jax.vmap(projection.projection_simplex)

    def fun(X):
      return jnp.sum(proj(X) ** 2)

    rng = onp.random.RandomState(0)
    X = rng.rand(4, 5).astype(onp.float32)
    U = rng.rand(4, 5)
    U /= onp.sqrt(onp.sum(U ** 2))
    U = U.astype(onp.float32)

    eps = 1e-3
    dir_deriv_num = (fun(X + eps * U) - fun(X - eps * U)) / (2 * eps)
    dir_deriv = jnp.vdot(jax.grad(fun)(X), U)
    self.assertAllClose(dir_deriv, dir_deriv_num, atol=1e-3)

  def test_projection_l1_sphere(self):
    rng = onp.random.RandomState(0)

    x = rng.randn(50).astype(onp.float32)

    p = projection.projection_l1_sphere(x)
    self.assertAllClose(jnp.sum(jnp.abs(p)), 1.0)

    radius = 3.21
    p = projection.projection_l1_sphere(x, radius)
    self.assertAllClose(jnp.sum(jnp.abs(p)), radius)

  def _check_projection_ball(self, ball_fun, sphere_fun, norm, check_pytree):
    rng = onp.random.RandomState(0)

    x = rng.randn(50).astype(onp.float32)
    norm_value = jnp.linalg.norm(x, ord=norm)

    # x is already in the ball
    big_radius = norm_value * 2
    p = ball_fun(x, big_radius)
    self.assertArraysAllClose(x, p)
    J = jax.jacrev(ball_fun)(x, big_radius)
    self.assertArraysAllClose(J, jnp.eye(len(x)))
    J = jax.jacrev(ball_fun, argnums=1)(x, big_radius)
    self.assertArraysAllClose(J, jnp.zeros_like(x))

    # x is at the boundary of the ball
    p = ball_fun(x, norm_value)
    self.assertArraysAllClose(x, p)
    J = jax.jacrev(ball_fun)(x, norm_value)
    self.assertArraysAllClose(J, jnp.eye(len(x)))
    J = jax.jacrev(ball_fun, argnums=1)(x, norm_value)
    self.assertArraysAllClose(J, jnp.zeros_like(x))

    # x is outside of the ball
    small_radius = norm_value / 2
    p = ball_fun(x, small_radius)
    self.assertAllClose(jnp.linalg.norm(p, ord=norm), small_radius)
    J = jax.jacrev(ball_fun)(x, small_radius)
    J2 = jax.jacrev(sphere_fun)(x, small_radius)
    self.assertArraysAllClose(J, J2)
    J = jax.jacrev(ball_fun, argnums=1)(x, small_radius)
    J2 = jax.jacrev(sphere_fun, argnums=1)(x, small_radius)
    self.assertArraysAllClose(J, J2)

    if check_pytree:
      # x is a pytree
      pytree = (x[:25], x[25:])
      p = ball_fun(pytree, norm_value)
      self.assertArraysAllClose(x[:25], p[0])
      self.assertArraysAllClose(x[25:], p[1])

  def test_projection_l1_ball(self):
    self._check_projection_ball(projection.projection_l1_ball,
                                projection.projection_l1_sphere,
                                norm=1, check_pytree=False)

  def test_projection_l2_sphere(self):
    rng = onp.random.RandomState(0)

    x = rng.randn(50).astype(onp.float32)

    p = projection.projection_l2_sphere(x)
    self.assertAllClose(jnp.sum(p ** 2), 1.0)

    radius = 3.21
    p = projection.projection_l2_sphere(x, radius)
    self.assertAllClose(jnp.sqrt(jnp.sum(p ** 2)), radius)

  def test_projection_l2_ball(self):
    self._check_projection_ball(projection.projection_l2_ball,
                                projection.projection_l2_sphere,
                                norm=2, check_pytree=True)

  def test_projection_linf_ball(self):
    rng = onp.random.RandomState(0)

    x = rng.randn(50).astype(onp.float32)
    linf_norm = jnp.max(jnp.abs(x))

    p = projection.projection_linf_ball(x, linf_norm)
    self.assertArraysAllClose(x, p)

    p = projection.projection_linf_ball(x, linf_norm * 0.5)
    self.assertAllClose(jnp.max(jnp.abs(p)), linf_norm * 0.5)

  def test_projection_hyperplane(self):
    rng = onp.random.RandomState(0)

    x = rng.randn(50).astype(onp.float32)
    a = rng.randn(50).astype(onp.float32)
    b = 1.0

    p = projection.projection_hyperplane(x, (a, b))
    self.assertAllClose(jnp.dot(a, p), b)

    tree_x = (1.0, {"k1": 2.0, "k2": (2.0, 1.0)}, 2.0)
    tree_a = (1.0, {"k1": 1.0, "k2": (1.0, 0.0)}, 1.0)
    p = projection.projection_hyperplane(tree_x, (tree_a, b))
    expected_p = (-0.5, {"k1": 0.5, "k2": (0.5, 1.0)}, 0.5)
    self.assertAllClose(expected_p, p)

  def test_projection_halfspace(self):
    # For b very large, halfplane projection of x is x.
    rng = onp.random.RandomState(0)
    tree_x = (rng.randn(1), {"k1": rng.randn(1), "k2": (rng.randn(1), rng.randn(1))}, rng.randn(1))
    tree_a = (rng.randn(1), {"k1": rng.randn(1), "k2": (rng.randn(1), rng.randn(1))}, rng.randn(1))
    b = 10000
    p = projection.projection_halfspace(tree_x, (tree_a, b))
    self.assertAllClose(tree_x, p)

    # For b very negative, the halfplane projection of x is on the hyperplane <a,x> =b.
    b = -10000
    p = projection.projection_halfspace(tree_x, (tree_a, b))
    p_hyper = projection.projection_hyperplane(tree_x, (tree_a, b))
    self.assertAllClose(p_hyper, p)

  def test_projection_affine_set(self):
    rng = onp.random.RandomState(0)

    x = rng.randn(5).astype(onp.float32)
    A = rng.randn(3, 5).astype(onp.float32)
    b = rng.randn(3).astype(onp.float32)
    params = (A, b)

    p = projection.projection_affine_set(x, params)
    self.assertAllClose(jnp.dot(A, p), b, atol=1e-3, rtol=1e-3)

    jac_fun = jax.jacrev(projection.projection_affine_set, argnums=0)
    jac_x = jac_fun(x, params)
    self.assertEqual(jac_x.shape, (5, 5))

    jac_fun = jax.jacrev(projection.projection_affine_set, argnums=1)
    jac_A, jac_b = jac_fun(x, params)
    self.assertEqual(jac_A.shape, (5, 3, 5))
    self.assertEqual(jac_b.shape, (5, 3))

  def test_projection_polyhedron(self):
    rng = onp.random.RandomState(0)

    x = rng.randn(5).astype(onp.float32)
    A = rng.randn(3, 5).astype(onp.float32)
    b = rng.randn(3).astype(onp.float32)
    G = rng.randn(2, 5).astype(onp.float32)
    h = rng.randn(2).astype(onp.float32)
    params = (A, b, G, h)

    p = projection.projection_polyhedron(x, params)
    self.assertAllClose(jnp.dot(A, p), b, atol=1e-3, rtol=1e-3)

    jac_fun = jax.jacrev(projection.projection_polyhedron, argnums=0)
    jac_x = jac_fun(x, params)
    self.assertEqual(jac_x.shape, (5, 5))

    jac_fun = jax.jacrev(projection.projection_polyhedron, argnums=1)
    jac_A, jac_b, jac_G, jac_h = jac_fun(x, params)
    self.assertEqual(jac_A.shape, (5, 3, 5))
    self.assertEqual(jac_b.shape, (5, 3))
    self.assertEqual(jac_G.shape, (5, 2, 5))
    self.assertEqual(jac_h.shape, (5, 2))

  def test_projection_polyhedron_infeasible(self):
    def proj(x):
      A = jnp.array([[1.0, 1.0],[0.0, 1.0],[-1.0, 1.0]])
      b = jnp.array([1.0, 1.0, -1.0])
      G = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
      h = jnp.array([0.0, 0.0])
      return projection.projection_polyhedron(x, hyperparams=(A, b, G, h))

    self.assertRaises(ValueError, proj, jnp.zeros(2))

  def test_projection_box_section(self):
    x = jnp.array([1.2, 3.0, 0.7])
    w = jnp.array([1.0, 2.0, 3.0])
    alpha = jnp.zeros_like(x)
    beta = jnp.ones_like(x) * 0.8
    c = 1.0
    params = (alpha, beta, w, c)
    p = projection.projection_box_section(x, params)
    self.assertAllClose(jnp.dot(w, p), c, atol=1e-5)
    self.assertTrue(jnp.all(alpha <= p))
    self.assertTrue(jnp.all(p <= beta))

    Q = jnp.eye(len(x))
    A = jnp.array([w])
    b = jnp.array([c])
    G = jnp.concatenate((-jnp.eye(len(x)), jnp.eye(len(x))))
    h = jnp.concatenate((alpha, beta))
    hyperparams = dict(params_obj=(Q, -x), params_eq=(A, b), params_ineq=(G, h))
    osqp = OSQP()
    p2 = osqp.run(**hyperparams).params.primal
    self.assertArraysAllClose(p, p2, atol=1e-3)

  def test_projection_box_section_infeasible(self):
    x = jnp.array([1.2, 3.0, 0.7])
    w = jnp.array([1.0, 2.0, 3.0])
    alpha = jnp.zeros_like(x)
    beta = jnp.ones_like(x)
    params = (alpha + 1.5, beta, w, 1.0)
    self.assertRaises(ValueError, projection.projection_box_section, x, params,
                      True)
    params = (alpha, beta - 1.5, w, 1.0)
    self.assertRaises(ValueError, projection.projection_box_section, x, params,
                      True)

  def test_projection_box_section_simplex(self):
    rng = onp.random.RandomState(0)
    x = jnp.array(rng.rand(5).astype(onp.float32))
    w = jnp.ones_like(x)
    alpha = jnp.zeros_like(x)
    beta = jnp.ones_like(x)
    c = 1.0
    params = (alpha, beta, w, c)

    p = projection.projection_simplex(x)
    p2 = projection.projection_box_section(x, params)
    self.assertArraysAllClose(p, p2, atol=1e-4)

    J = jax.jacrev(projection.projection_simplex)(x)
    J2 = jax.jacrev(projection.projection_box_section)(x, params)
    self.assertArraysAllClose(J, J2)

  def _sinkhorn(self, cost_matrix, marginals_a, marginals_b, maxiter=1000):
    eps = 1.0
    K = onp.exp(-cost_matrix / eps)
    f = onp.zeros_like(marginals_a)
    for it in range(maxiter):
      exp_f = onp.exp(f / eps)
      g = eps * onp.log(marginals_b) - eps * onp.log(K.T.dot(exp_f))
      exp_g = onp.exp(g / eps)
      f = eps * onp.log(marginals_a) - eps * onp.log(K.dot(exp_g))
    exp_f = onp.exp(f / eps)
    return exp_f[:, onp.newaxis] * K * exp_g

  def test_projection_transport(self):
    rng = onp.random.RandomState(0)
    sim_matrix = rng.rand(5, 6)
    marginals_a = rng.rand(5)
    marginals_a /= jnp.sum(marginals_a)
    marginals_b = rng.rand(6)
    marginals_b /= jnp.sum(marginals_b)
    make_solver = lambda fun: LBFGS(
        fun=fun, tol=1e-12, maxiter=10000, linesearch="zoom")

    for use_semi_dual in [False, True]:
      T1 = projection.projection_transport(
          sim_matrix, (marginals_a, marginals_b), make_solver,
          use_semi_dual=use_semi_dual)
      T2 = projection.kl_projection_transport(
          sim_matrix, (marginals_a, marginals_b), make_solver,
          use_semi_dual=use_semi_dual)

      for T in (T1, T2):
        self.assertEqual(T.shape, (5, 6))
        self.assertArraysAllClose(jnp.sum(T, axis=1), marginals_a, atol=1e-3)
        self.assertArraysAllClose(jnp.sum(T, axis=0), marginals_b, atol=1e-5)
        self.assertAllClose(jnp.sum(T), 1.0)

      # Compare with Sinkhorn.
      T3 = self._sinkhorn(cost_matrix=-sim_matrix, marginals_a=marginals_a,
                          marginals_b=marginals_b)
      self.assertArraysAllClose(T2, T3, atol=1e-4)

  def test_projection_birkhoff(self):
    rng = onp.random.RandomState(0)

    sim_matrix = rng.rand(5, 5)
    sim_matrix = 0.5 * (sim_matrix + sim_matrix.T)
    P1 = projection.projection_birkhoff(sim_matrix)
    P2 = projection.kl_projection_birkhoff(sim_matrix)

    # Check that P is a doubly-stochastic matrix.
    for P in (P1, P2):
      self.assertEqual(P.shape, (5, 5))
      self.assertArraysAllClose(jnp.sum(P, axis=1), jnp.ones(5), atol=1e-3)
      self.assertArraysAllClose(jnp.sum(P, axis=0), jnp.ones(5))

    # Check that input = output if input is already a doubly stochastic matrix.
    doubly_stochastic_matrix = jnp.array([[0.6, 0.4], [0.4, 0.6]])
    solution1 = projection.projection_birkhoff(doubly_stochastic_matrix)
    self.assertArraysAllClose(doubly_stochastic_matrix, solution1)

if __name__ == '__main__':
  absltest.main()