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

from jaxopt import gradient_descent
from jaxopt import implicit_diff
from jaxopt import test_util
from jaxopt import tree_util as tu

from sklearn import datasets


class GradientDescentTest(jtu.JaxTestCase):

  def test_logreg_with_intercept(self):
    X, y = datasets.load_digits(return_X_y=True)
    lam = 1e2
    tol = 1e-3
    maxiter = 200
    atol = 1e-3
    fun = test_util.make_logreg_objective(X, y, fit_intercept=True)

    pytree_init = (jnp.zeros((X.shape[1], 10)), jnp.zeros(10))
    solver_fun = gradient_descent.make_solver_fun(fun=fun, init=pytree_init,
                                                  tol=tol, maxiter=maxiter)
    pytree_fit = solver_fun(params_fun=lam)

    # Check optimality conditions.
    pytree_grad = jax.grad(fun)(pytree_fit, lam)
    self.assertLessEqual(tu.tree_l2_norm(pytree_grad), tol)

    # Compare against sklearn.
    W_skl, b_skl = test_util.logreg_skl(X, y, lam, fit_intercept=True)
    self.assertArraysAllClose(pytree_fit[0], W_skl, atol=atol)
    self.assertArraysAllClose(pytree_fit[1], b_skl, atol=atol)

  def test_logreg_implicit_diff(self):
    X, y = datasets.load_digits(return_X_y=True)
    lam = float(X.shape[0])
    tol = 1e-3
    maxiter = 200
    fun = test_util.make_logreg_objective(X, y)

    jac_num = test_util.logreg_skl_jac(X, y, lam)
    W_skl = test_util.logreg_skl(X, y, lam)

    # Make sure the decorator works.
    solver_fun = gradient_descent.make_solver_fun(fun=fun, init=W_skl, tol=tol,
                                                  maxiter=maxiter,
                                                  acceleration=True,
                                                  implicit_diff=True)
    jac_custom = jax.jacrev(solver_fun)(lam)
    self.assertArraysAllClose(jac_num, jac_custom, atol=1e-2)

  def test_has_aux(self):
    """Test implicit differentiation works when has_aux set to True."""
    X, y = datasets.load_digits(return_X_y=True)
    lam = float(X.shape[0])
    tol = 1e-3
    maxiter = 200

    W_skl = test_util.logreg_skl(X, y, lam)

    fun_naked = test_util.make_logreg_objective(X, y)
    fun_aux = lambda x, par: (fun_naked(x, par), jnp.ones_like(x))

    # Run both solvers and check they match.
    jac_custom = []

    for has_aux, fun in zip([False, True], [fun_naked, fun_aux]):
      solver_fun = gradient_descent.make_solver_fun(
          fun=fun, init=W_skl, tol=tol,
          maxiter=maxiter, acceleration=True, implicit_diff=True,
          has_aux=has_aux)

      jac_custom.append(jax.jacrev(solver_fun)(lam))

    self.assertArraysAllClose(jac_custom[0], jac_custom[1], atol=1e-4)

  @parameterized.product(acceleration=[True, False])
  def test_logreg_forward_diff(self, acceleration):
    X, y = datasets.load_digits(return_X_y=True)
    lam = float(X.shape[0])
    tol = 1e-3 if acceleration else 5e-3
    maxiter = 200
    atol = 1e-3 if acceleration else 1e-1
    fun = test_util.make_logreg_objective(X, y)

    jac_lam = test_util.logreg_skl_jac(X, y, lam)

    # Compute the Jacobian w.r.t. lam via forward differentiation.
    W_init = jnp.zeros((X.shape[1], 10))
    solver_fun = gradient_descent.make_solver_fun(fun=fun, init=W_init, tol=tol,
                                                  maxiter=maxiter,
                                                  implicit_diff=False,
                                                  acceleration=acceleration)
    jac_lam2 = jax.jacrev(solver_fun)(lam)
    self.assertArraysAllClose(jac_lam, jac_lam2, atol=atol)

  def test_jit_and_vmap(self):
    test_util.test_logreg_jit_and_vmap(
        self,
        gradient_descent.make_solver_fun,
        implicit_diff.make_gradient_descent_fixed_point_fun,
        jnp.array([1.0, 10.0]),
        atol=1e-3)

if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  # jax.config.update("jax_enable_x64", True)
  absltest.main(testLoader=jtu.JaxTestLoader())
