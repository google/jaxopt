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
from jax import test_util as jtu
import jax.numpy as jnp

from jaxopt import projection
from jaxopt import QuadraticProgramming
from jaxopt._src import quadratic_prog as _quadratic_prog

import numpy as onp


class QuadraticProgTest(jtu.JaxTestCase):

  def test_matvec_and_rmatvec(self):
    rng = onp.random.RandomState(0)
    A = rng.randn(5, 4)
    matvec = lambda x: jnp.dot(A, x)
    x = rng.randn(4)
    y = rng.randn(5)
    mv_A, rmv_A = _quadratic_prog._matvec_and_rmatvec(matvec, x, y)
    self.assertArraysAllClose(mv_A, jnp.dot(A, x))
    self.assertArraysAllClose(rmv_A, jnp.dot(A.T, y))

  def _check_derivative_A_and_b(self, solver, params, A, b):
    def fun(A, b):
      # reduce the primal variables to a scalar value for test purpose.
      hyperparams = dict(params_obj=params["params_obj"],
                         params_eq=(A, b),
                         params_ineq=params["params_ineq"])
      return jnp.sum(solver.run(**hyperparams).params[0])

    # Derivative w.r.t. A.
    rng = onp.random.RandomState(0)
    V = rng.rand(*A.shape)
    V /= onp.sqrt(onp.sum(V ** 2))
    eps = 1e-4
    deriv_jax = jnp.vdot(V, jax.grad(fun)(A, b))
    deriv_num = (fun(A + eps * V, b) - fun(A - eps * V, b)) / (2 * eps)
    self.assertAllClose(deriv_jax, deriv_num, atol=1e-3)

    # Derivative w.r.t. b.
    v = rng.rand(*b.shape)
    v /= onp.sqrt(onp.sum(b ** 2))
    eps = 1e-4
    deriv_jax = jnp.vdot(v, jax.grad(fun, argnums=1)(A, b))
    deriv_num = (fun(A, b + eps * v) - fun(A, b - eps * v)) / (2 * eps)
    self.assertAllClose(deriv_jax, deriv_num, atol=1e-3)

  def test_qp_eq_and_ineq(self):
    Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
    c = jnp.array([1.0, 1.0])
    A = jnp.array([[1.0, 1.0]])
    b = jnp.array([1.0])
    G = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
    h = jnp.array([0.0, 0.0])
    qp = QuadraticProgramming()
    hyperparams = dict(params_obj=(Q, c), params_eq=(A, b), params_ineq=(G, h))
    sol = qp.run(**hyperparams).params
    self.assertAllClose(qp.l2_optimality_error(sol, **hyperparams), 0.0)
    self._check_derivative_A_and_b(qp, hyperparams, A, b)

  def test_qp_eq_only(self):
    Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
    c = jnp.array([1.0, 1.0])
    A = jnp.array([[1.0, 1.0]])
    b = jnp.array([1.0])
    qp = QuadraticProgramming()
    hyperparams = dict(params_obj=(Q, c), params_eq=(A, b), params_ineq=None)
    sol = qp.run(**hyperparams).params
    self.assertAllClose(qp.l2_optimality_error(sol, **hyperparams), 0.0)
    self._check_derivative_A_and_b(qp, hyperparams, A, b)

  def test_projection_hyperplane(self):
    x = jnp.array([1.0, 2.0])
    a = jnp.array([-0.5, 1.5])
    b = 0.3
    # Find ||y-x||^2 such that jnp.dot(y, a) = b.
    expected = projection.projection_hyperplane(x, (a, b))

    matvec_Q = lambda params_Q, u: u
    matvec_A = lambda params_A, u: jnp.dot(a, u).reshape(1)
    qp = QuadraticProgramming(matvec_Q=matvec_Q, matvec_A=matvec_A)
    # In this example, params_Q = params_A = None.
    hyperparams = dict(params_obj=(None, -x),
                       params_eq=(None, jnp.array([b])))
    sol = qp.run(**hyperparams).params
    primal_sol = sol[0]
    self.assertArraysAllClose(primal_sol, expected)
    self.assertAllClose(qp.l2_optimality_error(sol, **hyperparams), 0.0)

  def test_projection_simplex(self):
    def _projection_simplex_qp(x, s=1.0):
      Q = jnp.eye(len(x))
      A = jnp.array([jnp.ones_like(x)])
      b = jnp.array([s])
      G = -jnp.eye(len(x))
      h = jnp.zeros_like(x)
      hyperparams = dict(params_obj=(Q, -x), params_eq=(A, b),
                         params_ineq=(G, h))

      qp = QuadraticProgramming()
      # Returns the primal solution only.
      return qp.run(**hyperparams).params[0]

    rng = onp.random.RandomState(0)
    x = jnp.array(rng.randn(10).astype(onp.float32))
    p = projection.projection_simplex(x)
    p2 = _projection_simplex_qp(x)
    self.assertArraysAllClose(p, p2, atol=1e-4)

    J = jax.jacrev(projection.projection_simplex)(x)
    J2 = jax.jacrev(_projection_simplex_qp)(x)
    self.assertArraysAllClose(J, J2, atol=1e-5)

  def test_eq_constrained_qp_with_pytrees(self):
    rng = onp.random.RandomState(0)
    Q = rng.randn(7, 7)
    Q = onp. dot(Q, Q.T)
    A = rng.randn(4, 7)

    tmp = rng.randn(7)
    # Must have the same pytree structure as the output of matvec_Q.
    c = (tmp[:3], tmp[3:])
    # Must have the same pytree structure as the output of matvec_A.
    b = rng.randn(4)

    def matvec_Q(Q, tup):
      x_ = jnp.concatenate(tup)
      res = jnp.dot(Q, x_)
      return res[:3], res[3:]

    def matvec_A(A, tup):
      x_ = jnp.concatenate(tup)
      return jnp.dot(A, x_)

    # With pytrees directly.
    hyperparams = dict(params_obj=(Q, c), params_eq=(A, b))
    qp = QuadraticProgramming(matvec_Q=matvec_Q, matvec_A=matvec_A)
    # sol.primal has the same pytree structure as the output of matvec_Q.
    # sol.dual_eq has the same pytree structure as the output of matvec_A.
    sol_pytree = qp.run(**hyperparams).params
    self.assertAllClose(qp.l2_optimality_error(sol_pytree, **hyperparams), 0.0,
                        atol=1e-4)

    # With flattened pytrees.
    hyperparams = dict(params_obj=(Q, jnp.concatenate(c)), params_eq=(A, b))
    qp = QuadraticProgramming()
    sol = qp.run(**hyperparams).params
    self.assertAllClose(qp.l2_optimality_error(sol, **hyperparams), 0.0,
                        atol=1e-4)

    # Check that the solutions match.
    self.assertArraysAllClose(jnp.concatenate(sol_pytree.primal), sol.primal,
                              atol=1e-4)
    self.assertArraysAllClose(sol_pytree.dual_eq, sol.dual_eq, atol=1e-4)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
