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
    data = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    fun = test_util.ridge_objective  # fun(params, hyperparams, data)
    optimality_fun = jax.grad(fun)
    hyperparams = 5.0
    sol = test_util.ridge_solver(hyperparams, data)
    vjp = lambda g: idf.root_vjp(optimality_fun=optimality_fun,
                                 sol=sol,
                                 hyperparams=hyperparams,
                                 cotangent=g,
                                 data=data)
    I = jnp.eye(len(sol))
    J = jax.vmap(vjp)(I)
    J_num = test_util.ridge_solver_jac(hyperparams, data, eps=1e-4)
    self.assertArraysAllClose(J, J_num, atol=1e-2)

  def test_root_jvp(self):
    data = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    fun = test_util.ridge_objective  # fun(params, hyperparams, data)
    optimality_fun = jax.grad(fun)
    hyperparams = 5.0
    sol = test_util.ridge_solver(hyperparams, data)
    J = idf.root_jvp(optimality_fun=optimality_fun,
                     sol=sol,
                     hyperparams=hyperparams,
                     tangent=1.0,
                     data=data)
    J_num = test_util.ridge_solver_jac(hyperparams, data, eps=1e-4)
    self.assertArraysAllClose(J, J_num, atol=1e-2)

  def test_custom_root(self):
    data = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    fun = test_util.ridge_objective  # fun(params, hyperparams, data)
    optimality_fun = jax.grad(fun)
    hyperparams = 5.0
    ridge_solver = idf.custom_root(optimality_fun)(test_util.ridge_solver)
    sol = ridge_solver(hyperparams, data)
    J = jax.jacobian(ridge_solver)(hyperparams, data)
    J_num = test_util.ridge_solver_jac(hyperparams, data, eps=1e-4)
    self.assertArraysAllClose(J, J_num, atol=1e-2)

  def test_custom_root_closed_over_data(self):
    data = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    def fun(params, hyperparams):
      return test_util.ridge_objective(params, hyperparams, data)
    optimality_fun = jax.grad(fun)
    hyperparams = 5.0
    @idf.custom_root(optimality_fun)
    def ridge_solver(hyperparams):
      return test_util.ridge_solver(hyperparams, data)
    sol = ridge_solver(hyperparams)
    J = jax.jacobian(ridge_solver)(hyperparams)
    J_num = test_util.ridge_solver_jac(hyperparams, data, eps=1e-4)
    self.assertArraysAllClose(J, J_num, atol=1e-2)

  def test_custom_root_has_aux(self):
    data = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    fun = test_util.ridge_objective  # fun(params, hyperparams, data)
    optimality_fun = jax.grad(fun)
    hyperparams = 5.0
    @idf.custom_root(optimality_fun, has_aux=True)
    def ridge_solver(hyperparams, data):
      return test_util.ridge_solver(hyperparams, data), None
    J, _ = jax.jacobian(ridge_solver)(hyperparams, data)
    J_num = test_util.ridge_solver_jac(hyperparams, data, eps=1e-4)
    self.assertArraysAllClose(J, J_num, atol=1e-2)

  def test_custom_root_extra_args(self):
    data = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    fun = test_util.ridge_objective  # fun(params, hyperparams, data)
    optimality_fun = jax.grad(fun)
    hyperparams = 5.0
    @idf.custom_root(optimality_fun)
    def ridge_solver(hyperparams, data, extra, arg):
      return test_util.ridge_solver(hyperparams, data)
    J = jax.jacobian(ridge_solver)(hyperparams, data, None, None)
    J_num = test_util.ridge_solver_jac(hyperparams, data, eps=1e-4)
    self.assertArraysAllClose(J, J_num, atol=1e-2)

if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  # jax.config.update("jax_enable_x64", True)
  absltest.main(testLoader=jtu.JaxTestLoader())
