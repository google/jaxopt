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

import functools

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp

from jaxopt import implicit_diff as idf
from jaxopt._src import test_util

from sklearn import datasets


def ridge_objective(params, lam, X, y):
  residuals = jnp.dot(X, params) - y
  return 0.5 * jnp.mean(residuals ** 2) + 0.5 * lam * jnp.sum(params ** 2)


# def ridge_solver(init_params, lam, X, y):
def ridge_solver(init_params, lam, X, y):
  del init_params  # not used
  XX = jnp.dot(X.T, X)
  Xy = jnp.dot(X.T, y)
  I = jnp.eye(X.shape[1])
  return jnp.linalg.solve(XX + lam * len(y) * I, Xy)


def ridge_solver_with_kwargs(init_params, **kw):
  return ridge_solver(init_params, kw['lam'], kw['X'], kw['y'])


class ImplicitDiffTest(test_util.JaxoptTestCase):

  def test_root_vjp(self):
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    optimality_fun = jax.grad(ridge_objective)
    lam = 5.0
    sol = ridge_solver(None, lam, X, y)
    vjp = lambda g: idf.root_vjp(optimality_fun=optimality_fun,
                                 sol=sol,
                                 args=(lam, X, y),
                                 cotangent=g)[0]  # vjp w.r.t. lam
    I = jnp.eye(len(sol))
    J = jax.vmap(vjp)(I)
    J_num = test_util.ridge_solver_jac(X, y, lam, eps=1e-4)
    self.assertArraysAllClose(J, J_num, atol=5e-2)

  def test_root_jvp(self):
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    optimality_fun = jax.grad(ridge_objective)
    lam = 5.0
    sol = ridge_solver(None, lam, X, y)
    J = idf.root_jvp(optimality_fun=optimality_fun,
                     sol=sol,
                     args=(lam, X, y),
                     tangents=(1.0, jnp.zeros_like(X), jnp.zeros_like(y)))
    J_num = test_util.ridge_solver_jac(X, y, lam, eps=1e-4)
    self.assertArraysAllClose(J, J_num, atol=5e-2)

  def test_custom_root(self):
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    optimality_fun = jax.grad(ridge_objective)
    lam = 5.0
    ridge_solver_decorated = idf.custom_root(optimality_fun)(ridge_solver)
    sol = ridge_solver(None, lam=lam, X=X, y=y)
    sol_decorated = ridge_solver_decorated(None, lam=lam, X=X, y=y)
    self.assertArraysAllClose(sol, sol_decorated, atol=1e-4)
    J_num = test_util.ridge_solver_jac(X, y, lam, eps=1e-4)
    J = jax.jacrev(ridge_solver_decorated, argnums=1)(None, lam, X=X, y=y)
    self.assertArraysAllClose(J, J_num, atol=5e-2)

  def test_custom_root_with_has_aux(self):
    def ridge_solver_with_aux(init_params, lam, X, y):
      return ridge_solver(init_params, lam, X, y), None

    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    optimality_fun = jax.grad(ridge_objective)
    lam = 5.0
    decorator = idf.custom_root(optimality_fun, has_aux=True)
    ridge_solver_decorated = decorator(ridge_solver_with_aux)
    sol = ridge_solver(None, lam=lam, X=X, y=y)
    sol_decorated = ridge_solver_decorated(None, lam=lam, X=X, y=y)[0]
    self.assertArraysAllClose(sol, sol_decorated, atol=1e-4)
    J_num = test_util.ridge_solver_jac(X, y, lam, eps=1e-4)
    J, _ = jax.jacrev(ridge_solver_decorated, argnums=1)(None, lam, X=X, y=y)
    self.assertArraysAllClose(J, J_num, atol=5e-2)

  def test_custom_fixed_point(self):
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    grad_fun = jax.grad(ridge_objective)
    fp_fun = lambda x, *args: x - grad_fun(x, *args)
    fp_fun = functools.wraps(grad_fun)(fp_fun)  # carries grad_fun signature
    lam = 5.0
    ridge_solver_decorated = idf.custom_fixed_point(fp_fun)(ridge_solver)
    sol = ridge_solver(None, lam=lam, X=X, y=y)
    sol_decorated = ridge_solver_decorated(None, lam=lam, X=X, y=y)
    self.assertArraysAllClose(sol, sol_decorated, atol=1e-4)
    J_num = test_util.ridge_solver_jac(X, y, lam, eps=1e-4)
    J = jax.jacrev(ridge_solver_decorated, argnums=1)(None, lam, X=X, y=y)
    self.assertArraysAllClose(J, J_num, atol=5e-2)

  def test_custom_root_with_kwargs(self):
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    lam = 5.0
    w_ref = ridge_solver(None, lam, X, y)

    cr_solver = idf.custom_root(jax.grad(ridge_objective))(ridge_solver_with_kwargs)
    w = cr_solver(None, lam=lam, X=X, y=y)
    w_jit = jax.jit(cr_solver)(None, lam=lam, X=X, y=y)

    self.assertArraysAllClose(w_ref, w, atol=5e-2)
    self.assertArraysAllClose(w_jit, w, atol=5e-2)

  def test_custom_root_with_kwargs_unordered(self):
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    lam = 5.0
    w_ref = ridge_solver(None, lam, X, y)

    cr_solver = idf.custom_root(jax.grad(ridge_objective))(ridge_solver_with_kwargs)
    w = cr_solver(None, X=X, lam=lam, y=y)
    w_jit = jax.jit(cr_solver)(None, X=X, lam=lam, y=y)

    self.assertArraysAllClose(w_ref, w, atol=5e-2)
    self.assertArraysAllClose(w_jit, w, atol=5e-2)

  def test_custom_root_with_kwargs_grad(self):
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    lam = 5.0
    lam_grad_ref = jax.grad(lambda lam: ridge_solver(None, lam, X, y)[0])(lam)

    cr_solver = idf.custom_root(jax.grad(ridge_objective))(ridge_solver_with_kwargs)
    grad_fun = jax.grad(lambda lam: cr_solver(None, lam=lam, X=X, y=y)[0])
    lam_grad = grad_fun(lam)
    lam_grad_jit = jax.jit(grad_fun)(lam)

    self.assertArraysAllClose(lam_grad_ref, lam_grad, atol=5e-2)
    self.assertArraysAllClose(lam_grad_jit, lam_grad, atol=5e-2)

  def test_custom_root_with_kwargs_grad_unordered(self):
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    lam = 5.0
    lam_grad_ref = jax.grad(lambda lam: ridge_solver(None, lam, X, y)[0])(lam)

    cr_solver = idf.custom_root(jax.grad(ridge_objective))(ridge_solver_with_kwargs)
    grad_fun = jax.grad(lambda lam: cr_solver(None, X=X, lam=lam, y=y)[0])
    lam_grad = grad_fun(lam)
    lam_grad_jit = jax.jit(grad_fun)(lam)

    self.assertArraysAllClose(lam_grad_ref, lam_grad, atol=5e-2)
    self.assertArraysAllClose(lam_grad_jit, lam_grad, atol=5e-2)

if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  # jax.config.update("jax_enable_x64", True)
  absltest.main()
