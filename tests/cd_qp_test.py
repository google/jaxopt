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

from absl.testing import absltest

import jax
import jax.numpy as jnp

from jaxopt import BoxCDQP
from jaxopt._src import test_util

import numpy as onp


def _cd_qp(Q, c, l, u, tol, maxiter, verbose=0):
  """Pure NumPy implementation for test purposes."""
  x = onp.zeros(Q.shape[0])

  for it in range(maxiter):
    error = 0

    for i in range(len(x)):
      g_i = onp.dot(Q[i], x) + c[i]
      h_i = Q[i, i]

      if h_i == 0:
        continue

      x_i_new = onp.clip(x[i] - g_i / h_i, l[i], u[i])
      delta_i = x_i_new - x[i]
      error += onp.abs(delta_i)
      x[i] = x_i_new

    if verbose:
      print(it + 1, error)

    if error <= tol:
      break

  return x


class CD_QP_Test(test_util.JaxoptTestCase):

  def setUp(self):
    rng = onp.random.RandomState(0)
    num_dim = 5
    M = rng.randn(num_dim, num_dim)
    self.Q = onp.dot(M, M.T)
    self.c = rng.randn(num_dim)
    self.l = rng.randn(num_dim)
    self.u = self.l + 5 * rng.rand(num_dim)
    self.params_obj = (self.Q, self.c)
    self.params_ineq = (self.l, self.u)

  def test_forward(self):
    sol_numpy = _cd_qp(self.Q, self.c, self.l, self.u,
                       tol=1e-3, maxiter=100, verbose=0)

    # Manual loop
    params = jnp.zeros_like(sol_numpy)

    cdqp = BoxCDQP()
    state = cdqp.init_state(params)

    for _ in range(5):
      params, state = cdqp.update(params, state, params_obj=self.params_obj,
                                  params_ineq=self.params_ineq)

    self.assertAllClose(state.error, 0.0)
    self.assertAllClose(params, sol_numpy)

    # Run call.
    params = jnp.zeros_like(sol_numpy)
    params, state = cdqp.run(params, params_obj=self.params_obj,
                             params_ineq=self.params_ineq)
    self.assertAllClose(state.error, 0.0)
    self.assertAllClose(params, sol_numpy)

  def test_backward(self):
    cdqp = BoxCDQP(implicit_diff=True)
    cdqp2 = BoxCDQP(implicit_diff=False)
    init_params = jnp.zeros(self.Q.shape[0])

    def wrapper(c):
      params_obj = (self.Q, c)
      return cdqp.run(init_params, params_obj=params_obj,
                      params_ineq=self.params_ineq).params

    def wrapper2(c):
      params_obj = (self.Q, c)
      return cdqp2.run(init_params, params_obj=params_obj,
                       params_ineq=self.params_ineq).params

    J = jax.jacobian(wrapper)(self.c)
    J2 = jax.jacobian(wrapper2)(self.c)
    self.assertAllClose(J, J2)


if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  #jax.config.update("jax_enable_x64", True)
  absltest.main()
