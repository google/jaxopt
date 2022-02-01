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

"""CVXPY tests."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import numpy as onp

from jaxopt import projection
from jaxopt import CvxpyQP
from jaxopt._src import test_util


class CvxpyQPTest(test_util.JaxoptTestCase):

  def _check_derivative_Q_c_A_b(self, solver, params, Q, c, A, b):
    def fun(Q, c, A, b):
      try:
        params_ineq = params["params_ineq"]
      except KeyError:
        params_ineq = None

      Q = 0.5 * (Q + Q.T)

      hyperparams = dict(params_obj=(Q, c),
                         params_eq=(A, b),
                         params_ineq=params_ineq)

      # reduce the primal variables to a scalar value for test purpose.
      return jnp.sum(solver.run(None, **hyperparams).params[0])

    # Derivative w.r.t. A.
    rng = onp.random.RandomState(0)
    V = rng.rand(*A.shape)
    V /= onp.sqrt(onp.sum(V ** 2))
    eps = 1e-4
    deriv_jax = jnp.vdot(V, jax.grad(fun, argnums=2)(Q, c, A, b))
    deriv_num = (fun(Q, c, A + eps * V, b) - fun(Q, c, A - eps * V, b)) / (2 * eps)
    self.assertAllClose(deriv_jax, deriv_num, atol=1e-3)

    # Derivative w.r.t. b.
    v = rng.rand(*b.shape)
    v /= onp.sqrt(onp.sum(v ** 2))
    eps = 1e-4
    deriv_jax = jnp.vdot(v, jax.grad(fun, argnums=3)(Q, c, A, b))
    deriv_num = (fun(Q, c, A, b + eps * v) - fun(Q, c, A, b - eps * v)) / (2 * eps)
    self.assertAllClose(deriv_jax, deriv_num, atol=1e-3)

    # Derivative w.r.t. Q
    W = rng.rand(*Q.shape)
    W /= onp.sqrt(onp.sum(W ** 2))
    eps = 1e-4
    deriv_jax = jnp.vdot(W, jax.grad(fun, argnums=0)(Q, c, A, b))
    deriv_num = (fun(Q + eps * W, c, A, b) - fun(Q - eps * W, c, A, b)) / (2 * eps)
    self.assertAllClose(deriv_jax, deriv_num, atol=1e-3)

    # Derivative w.r.t. c
    w = rng.rand(*c.shape)
    w /= onp.sqrt(onp.sum(w ** 2))
    eps = 1e-4
    deriv_jax = jnp.vdot(w, jax.grad(fun, argnums=1)(Q, c, A, b))
    deriv_num = (fun(Q, c + eps * w, A, b) - fun(Q, c - eps * w, A, b)) / (2 * eps)
    self.assertAllClose(deriv_jax, deriv_num, atol=1e-3)

  def test_qp_eq_and_ineq(self):
    Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
    c = jnp.array([1.0, 1.0])
    A = jnp.array([[1.0, 1.0]])
    b = jnp.array([1.0])
    G = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
    h = jnp.array([0.0, 0.0])
    qp = CvxpyQP()
    hyperparams = dict(params_obj=(Q, c), params_eq=(A, b), params_ineq=(G, h))
    sol = qp.run(None, **hyperparams).params
    self.assertAllClose(qp.l2_optimality_error(sol, **hyperparams), 0.0, atol=1e-4)
    self._check_derivative_Q_c_A_b(qp, hyperparams, Q, c, A, b)

  def test_projection_simplex(self):
    def _projection_simplex_qp(x, s=1.0):
      Q = jnp.eye(len(x))
      A = jnp.array([jnp.ones_like(x)])
      b = jnp.array([s])
      G = -jnp.eye(len(x))
      h = jnp.zeros_like(x)
      hyperparams = dict(params_obj=(Q, -x), params_eq=(A, b),
                         params_ineq=(G, h))

      qp = CvxpyQP()
      # Returns the primal solution only.
      return qp.run(None, **hyperparams).params[0]

    rng = onp.random.RandomState(0)
    x = jnp.array(rng.randn(10).astype(onp.float32))
    p = projection.projection_simplex(x)
    p2 = _projection_simplex_qp(x)
    self.assertArraysAllClose(p, p2)
    J = jax.jacrev(projection.projection_simplex)(x)
    J2 = jax.jacrev(_projection_simplex_qp)(x)
    self.assertArraysAllClose(J, J2, atol=1e-5)


if __name__ == '__main__':
  absltest.main()
