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

from jaxopt import implicit_diff
from jaxopt import projection
from jaxopt import quadratic_prog
from jaxopt import tree_util

import numpy as onp


class QuadraticProgTest(jtu.JaxTestCase):

  def _check_derivative_A_and_b(self, solver, params, A, b):
    def fun(A, b):
      # reduce the primal variables to a scalar value for test purpose.
      hyperparams = (params[0], (A, b), params[2])
      return jnp.sum(solver.run(hyperparams=hyperparams).params[0])

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
    qp = quadratic_prog.QuadraticProgramming()
    hyperparams = ((Q, c), (A, b), (G, h))
    sol = qp.run(hyperparams=hyperparams).params
    self.assertAllClose(qp.l2_optimality_error(sol, hyperparams), 0.0)
    self._check_derivative_A_and_b(qp, hyperparams, A, b)

  def test_qp_eq_only(self):
    Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
    c = jnp.array([1.0, 1.0])
    A = jnp.array([[1.0, 1.0]])
    b = jnp.array([1.0])
    qp = quadratic_prog.QuadraticProgramming()
    hyperparams = ((Q, c), (A, b), None)
    sol = qp.run(hyperparams=hyperparams).params
    self.assertAllClose(qp.l2_optimality_error(sol, hyperparams), 0.0)
    self._check_derivative_A_and_b(qp, hyperparams, A, b)

  def test_projection_simplex(self):
    def _projection_simplex_qp(x, s=1.0):
      Q = jnp.eye(len(x))
      A = jnp.array([jnp.ones_like(x)])
      b = jnp.array([s])
      G = -jnp.eye(len(x))
      h = jnp.zeros_like(x)
      hyperparams = ((Q, -x), (A, b), (G, h))

      qp = quadratic_prog.QuadraticProgramming()
      # Returns the primal solution only.
      return qp.run(hyperparams=hyperparams).params[0]

    rng = onp.random.RandomState(0)
    x = jnp.array(rng.randn(10).astype(onp.float32))
    p = projection.projection_simplex(x)
    p2 = _projection_simplex_qp(x)
    self.assertArraysAllClose(p, p2, atol=1e-4)

    J = jax.jacrev(projection.projection_simplex)(x)
    J2 = jax.jacrev(_projection_simplex_qp)(x)
    self.assertArraysAllClose(J, J2, atol=1e-5)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
