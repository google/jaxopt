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

import jax
from jax import test_util as jtu
import jax.numpy as jnp

from jaxopt import PseudoInversePreconditionedEqQP
from jaxopt import EqualityConstrainedQP
import numpy as onp


class PreconditionedEqualityConstrainedQPTest(jtu.JaxTestCase):
  def _check_derivative_Q_c_A_b(self, solver, Q, c, A, b):
    def fun(Q, c, A, b):
      Q = 0.5 * (Q + Q.T)

      hyperparams = dict(params_obj=(Q, c), params_eq=(A, b))
      # reduce the primal variables to a scalar value for test purpose.
      return jnp.sum(solver.run(**hyperparams).params[0])

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

  def test_pseudoinverse_preconditioner(self):
    Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
    c = jnp.array([1.0, 1.0])
    A = jnp.array([[1.0, 1.0]])
    b = jnp.array([1.0])
    qp = EqualityConstrainedQP(tol=1e-7)
    preconditioned_qp = PseudoInversePreconditionedEqQP(qp)
    params_obj = (Q, c)
    params_eq = (A, b)
    params_precond = preconditioned_qp.init_params(params_obj, params_eq)
    hyperparams = dict(
      params_obj=params_obj,
      params_eq=params_eq,
    )
    sol = preconditioned_qp.run(**hyperparams, params_precond=params_precond).params
    self.assertAllClose(qp.l2_optimality_error(sol, **hyperparams), 0.0)
    self._check_derivative_Q_c_A_b(preconditioned_qp, Q, c, A, b)
