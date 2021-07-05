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

import jax
from jax import test_util as jtu
import jax.numpy as jnp

from jaxopt import implicit_diff2 as idf
from jaxopt import test_util2 as test_util

from sklearn import datasets

class ImplicitDiffTest(jtu.JaxTestCase):

  def test_root_vjp(self):
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    fun = test_util.ridge_objective  # fun(params, hyperparams, data)
    optimality_fun = jax.grad(fun)
    lam = 5.0
    sol = test_util.ridge_solver(X, y, lam)
    vjp = lambda g: idf.root_vjp(optimality_fun=optimality_fun,
                                 sol=sol,
                                 hyperparams=lam,
                                 cotangent=g,
                                 data=(X, y))
    I = jnp.eye(len(sol))
    J = jax.vmap(vjp)(I)
    J_num = test_util.ridge_solver_jac(X, y, lam, eps=1e-4)
    self.assertArraysAllClose(J, J_num, atol=1e-2)

  def test_root_jvp(self):
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    fun = test_util.ridge_objective  # fun(params, hyperparams, data)
    optimality_fun = jax.grad(fun)
    lam = 5.0
    sol = test_util.ridge_solver(X, y, lam)
    J = idf.root_jvp(optimality_fun=optimality_fun,
                     sol=sol,
                     hyperparams=lam,
                     tangent=1.0,
                     data=(X, y))
    J_num = test_util.ridge_solver_jac(X, y, lam, eps=1e-4)
    self.assertArraysAllClose(J, J_num, atol=1e-2)

  @parameterized.product(has_aux=[True, False])
  def test_custom_root_closed_over_data(self, has_aux):
    """ Test @custom_root with solver_fun(hyperparams)."""
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    def fun(params, hyperparams):
      return test_util.ridge_objective(params, hyperparams, (X, y))
    optimality_fun = jax.grad(fun)
    lam = 5.0
    @idf.custom_root(optimality_fun, has_aux=has_aux)
    def ridge_solver(hyperparams):
      ret = test_util.ridge_solver(X, y, hyperparams)
      if has_aux:
        return ret, None  # Return some dummy output for test purposes.
      else:
        return ret
    sol = ridge_solver(lam)
    if has_aux:
      J, _ = jax.jacobian(ridge_solver)(lam)
    else:
      J = jax.jacobian(ridge_solver)(lam)
    J_num = test_util.ridge_solver_jac(X, y, lam, eps=1e-4)
    self.assertArraysAllClose(J, J_num, atol=1e-2)

  @parameterized.product(has_aux=[True, False])
  def test_custom_root_with_init_and_data(self, has_aux):
    """ Test @custom_root with solver_fun(init_params, hyperparams, data)."""
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    fun = test_util.ridge_objective  # fun(params, hyperparams, data)
    optimality_fun = jax.grad(fun)
    lam = 5.0
    @idf.custom_root(optimality_fun, has_aux=has_aux)
    def ridge_solver(init_params, hyperparams, data):
      del init_params  # not used
      ret = test_util.ridge_solver(data[0], data[1], hyperparams)
      if has_aux:
        return ret, None  # Return some dummy output for test purposes.
      else:
        return ret
    if has_aux:
      J, _ = jax.jacobian(ridge_solver, argnums=1)(None, lam, data=(X, y))
    else:
      J = jax.jacobian(ridge_solver, argnums=1)(None, lam, data=(X, y))
    J_num = test_util.ridge_solver_jac(X, y, lam, eps=1e-4)
    self.assertArraysAllClose(J, J_num, atol=1e-2)


if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  # jax.config.update("jax_enable_x64", True)
  absltest.main(testLoader=jtu.JaxTestLoader())
