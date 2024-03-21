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
from absl.testing import parameterized

from functools import partial

import jax.random
import jax.numpy as jnp
import numpy as onp

from jaxopt._src import test_util
from jax.test_util import check_grads
from jaxopt import NNLS, NMF
from sklearn.decomposition import NMF as sk_NMF


class NNLSTest(test_util.JaxoptTestCase):

  @parameterized.product(jit=[True, False])
  def test_random_nnls(self, jit):
    n, m = 20, 10
    rank = 5
    onp.random.seed(654)
    U = jax.nn.relu(onp.random.randn(n, rank))
    W_0 = jax.nn.relu(onp.random.randn(m, rank))
    Y = U @ W_0.T

    hyper_params = dict(params_obj=(Y, U), params_eq=None, params_ineq=None)
    tol = 1e-4
    solver = NNLS(tol=tol, jit=jit, rho_policy='adaptive')
    init_params = solver.init_params(**hyper_params)
    sol, state = solver.run(init_params=init_params, **hyper_params)

    # Check that the solution is close to the original.
    W_sol = sol.primal
    self.assertAllClose(W_0, W_sol, atol=1e-3, rtol=1e-3)
    self.assertLessEqual(state.error, tol)

  @parameterized.product(rho_policy=[0.001, 0.01, 0.1, 'osqp', 'adaptive'])
  def test_implicit_diff(self, rho_policy):
    n, m = 20, 10
    rank = 5
    onp.random.seed(654)
    U = jax.nn.relu(onp.random.randn(n, rank)).astype(jnp.float64)
    W_0 = jax.nn.relu(onp.random.randn(m, rank)).astype(jnp.float64)
    noise = 0.1*onp.random.randn(n, m)
    Y = jnp.array(U @ W_0.T + noise, dtype=jnp.float64)

    def run_solver(Y, U):
      hyper_params = dict(params_obj=(Y, U), params_eq=None, params_ineq=None)
      tol = 1e-6
      solver = NNLS(tol=tol, implicit_diff=True, cg_tol=1e-8, rho_policy=rho_policy)
      init_params = solver.init_params(**hyper_params)
      kkt_sol, _ = solver.run(init_params=init_params, **hyper_params)
      return kkt_sol

    # Check auto-differentiation.
    atol = 1e-2
    rtol = 1e-2
    eps = 1e-3
    run_solver_U = lambda U: run_solver(Y, U).primal
    run_solver_Y = lambda Y: run_solver(Y, U).primal
    check_grads(run_solver_U, args=(U,), order=1, modes=['rev'], eps=eps, atol=atol, rtol=rtol)
    check_grads(run_solver_Y, args=(Y,), order=1, modes=['rev'], eps=eps, atol=atol, rtol=rtol)


class NMFTest(test_util.JaxoptTestCase):

    @parameterized.named_parameters(
      {'testcase_name': 'small', 'size_a': 12, 'size_b': 6, 'rank': 3},
      {'testcase_name': 'big', 'size_a': 128, 'size_b': 64, 'rank': 12}
    )
    def test_random_problem(self, size_a, size_b, rank):
      onp.random.seed(8989)  # for Sklearn's internal randomness
      initializer = jax.nn.initializers.glorot_normal()  # ensures good scaling for big matrices : more stability
      key, subkey = jax.random.split(jax.random.PRNGKey(9898))
      H1 = jax.nn.relu(initializer(key, (size_a, rank), jnp.float32))
      H2 = jax.nn.relu(initializer(subkey, (size_b, rank), jnp.float32))
      Y = H1 @ H2.T

      tol = 1e-4

      nmf = NMF(rank=rank, init='sklearn_nndsvda', jit=True, tol=tol,
                nnls_solver=NNLS(tol=1e-4, cg_tol=1e-7))
      kkt_sol = nmf.init_params(Y)
      kkt_sol, state = nmf.run(kkt_sol, Y)
      h1, h2 = kkt_sol.primal

      atol = 1e-2
      rtol = 1e-2

      self.assertLessEqual(state.error, tol)
      
      sk_nmf = sk_NMF(n_components=rank, init=None, tol=tol, random_state=42)
      sk_h1 = sk_nmf.fit_transform(Y)
      sk_h2 = sk_nmf.components_.T

      self.assertAllClose(sk_h1 @ sk_h2.T, h1 @ h2.T, rtol=rtol, atol=atol)
      self.assertAllClose(Y, h1 @ h2.T, rtol=rtol, atol=atol)

      test_ground_truth = False
      if test_ground_truth:  # non convex: often fails to find the true optimum.
        self.assertAllClose(sk_h1, H1, rtol=rtol, atol=atol)
        self.assertAllClose(sk_h2, H2, rtol=rtol, atol=atol)
        self.assertAllClose(h1, H1, rtol=rtol, atol=atol)
        self.assertAllClose(h2, H2, rtol=rtol, atol=atol)

    def test_nmf_decreasing(self):
      size_a = 20
      size_b = 10
      rank = 5
      onp.random.seed(8989)  # for Sklearn's internal randomness
      initializer = jax.nn.initializers.glorot_normal()  # ensures good scaling for big matrices : more stability
      key, subkey = jax.random.split(jax.random.PRNGKey(9898))
      H1 = jax.nn.relu(initializer(key, (size_a, rank), jnp.float32))
      H2 = jax.nn.relu(initializer(subkey, (size_b, rank), jnp.float32))
      Y = H1 @ H2.T
 
      tol = 1e-3
      nmf = NMF(rank=rank, tol=tol, maxiter=10,
                nnls_solver=NNLS(tol=1e-6, cg_tol=1e-8))

      @jax.jit
      def jitted_update(kkt_sol, state, Y):
        return nmf.update(kkt_sol, state, Y, None, None)

      kkt_sol = nmf.init_params(Y)
      state = nmf.init_state(kkt_sol, Y)
      for _ in range(nmf.maxiter):
        kkt_sol, new_state = jitted_update(kkt_sol, state, Y)
        self.assertLessEqual(new_state.error, state.error)
        state = new_state
        if state.error < tol:
          break


if __name__ == '__main__':
  jax.config.update("jax_enable_x64", True)
  absltest.main()
