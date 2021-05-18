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

import functools

import jax
from jax import test_util as jtu
import jax.numpy as jnp

from jaxopt import implicit_diff
from jaxopt import projection
from jaxopt import prox
from jaxopt import proximal_gradient
from jaxopt import test_util
from jaxopt import tree_util as tu

from sklearn import datasets
from sklearn import preprocessing


class ProximalGradientTest(jtu.JaxTestCase):

  @parameterized.product(acceleration=[True, False])
  def test_lasso(self, acceleration):
    X, y = datasets.load_boston(return_X_y=True)
    lam = 1.0
    tol = 1e-3 if acceleration else 5e-3
    maxiter = 200
    atol = 1e-2 if acceleration else 1e-1
    fun = test_util.make_least_squares_objective(X, y)

    # Check optimality conditions.
    w_init = jnp.zeros(X.shape[1])
    solver_fun = proximal_gradient.make_solver_fun(fun=fun, init=w_init,
                                                   prox=prox.prox_lasso,
                                                   tol=tol, maxiter=maxiter,
                                                   acceleration=acceleration)
    w_fit = solver_fun(params_fun=lam, params_prox=1.0)
    w_fit2 = prox.prox_lasso(w_fit - jax.grad(fun)(w_fit, lam), 1.0)
    self.assertLessEqual(jnp.sqrt(jnp.sum((w_fit - w_fit2)**2)), tol)

    # Compare against sklearn.
    w_skl = test_util.lasso_skl(X, y, lam)
    self.assertArraysAllClose(w_fit, w_skl, atol=atol)

  def test_lasso_with_intercept(self):
    X, y = datasets.load_boston(return_X_y=True)
    lam = 1.0
    tol = 1e-3
    maxiter = 200
    atol = 1e-2
    fun = test_util.make_least_squares_objective(X, y, fit_intercept=True)

    def _prox(pytree, params, scaling=1.0):
      w, b = pytree
      return prox.prox_lasso(w, params, scaling), b

    # Check optimality conditions.
    pytree_init = (jnp.zeros(X.shape[1]), 0.0)
    solver_fun = proximal_gradient.make_solver_fun(fun=fun, init=pytree_init,
                                                   prox=_prox, tol=tol,
                                                   maxiter=maxiter,
                                                   acceleration=True)
    pytree_fit = solver_fun(params_fun=1.0, params_prox=lam)
    pytree = tu.tree_sub(pytree_fit, jax.grad(fun)(pytree_fit, 1.0))
    pytree_fit2 = _prox(pytree, lam)
    pytree_fit_diff = tu.tree_sub(pytree_fit, pytree_fit2)
    self.assertLessEqual(tu.tree_l2_norm(pytree_fit_diff), tol)

    # Compare against sklearn.
    w_skl, b_skl = test_util.lasso_skl(X, y, lam, fit_intercept=True)
    self.assertArraysAllClose(pytree_fit[0], w_skl, atol=atol)
    self.assertAllClose(pytree_fit[1], b_skl, atol=atol)

  def test_has_aux(self):
    """Test implicit differentiation works when has_aux set to True."""
    X, y = datasets.load_boston(return_X_y=True)
    lam = 1.5
    tol = 1e-3
    maxiter = 200

    fun_naked = test_util.make_least_squares_objective(X, y)
    fun_aux = lambda x, par: (fun_naked(x, par), jnp.ones_like(x))

    # Run both solvers and check they match.
    jac_custom = []

    for has_aux, fun in zip([True, False], [fun_aux, fun_naked]):
      solver_fun = proximal_gradient.make_solver_fun(
          fun=fun, init=jnp.zeros((X.shape[1],)), prox=prox.prox_lasso,
          tol=tol, maxiter=maxiter, acceleration=True, implicit_diff=True,
          has_aux=has_aux)
      jac_custom.append(jax.jacrev(solver_fun)(lam, 1.0))

    self.assertArraysAllClose(jac_custom[0], jac_custom[1], atol=1e-4)

  def test_lasso_implicit_diff(self):
    """Test implicit differentiation of a single lambda parameter."""
    X, y = datasets.load_boston(return_X_y=True)
    lam = 1.5
    tol = 1e-3
    maxiter = 200

    fun = test_util.make_least_squares_objective(X, y)
    jac_num = test_util.lasso_skl_jac(X, y, lam)
    w_skl = test_util.lasso_skl(X, y, lam)

    solver_fun = proximal_gradient.make_solver_fun(fun=fun, init=w_skl,
                                                   prox=prox.prox_lasso,
                                                   tol=tol, maxiter=maxiter,
                                                   acceleration=True,
                                                   implicit_diff=True)
    jac_custom = jax.jacrev(solver_fun)(lam, 1.0)
    self.assertArraysAllClose(jac_num, jac_custom, atol=1e-3)

    jac_custom = jax.jacrev(solver_fun, argnums=1)(1.0, lam)
    self.assertArraysAllClose(jac_num, jac_custom, atol=1e-3)

  def test_lasso_implicit_diff_multi(self):
    """Test implicit differentiation of multiple lambda parameters."""
    X, y = datasets.make_regression(n_samples=50, n_features=3, random_state=0)

    # Use one lambda per feature.
    n_features = X.shape[1]
    lam = jnp.array([1.0, 5.0, 10.0])
    fun = test_util.make_least_squares_objective(X, y)

    # Compute solution.
    w_init = jnp.zeros_like(lam)
    tol = 1e-3
    maxiter = 200
    I = jnp.eye(n_features)
    solver_fun = proximal_gradient.make_solver_fun(fun=fun, init=w_init,
                                                   prox=prox.prox_lasso,
                                                   tol=tol, maxiter=maxiter,
                                                   acceleration=True,
                                                   implicit_diff=True)
    sol = solver_fun(1.0, lam)

    # Compute the Jacobian w.r.t. lam (params_prox) using VJPs.
    fixed_point_fun = implicit_diff.make_proximal_gradient_fixed_point_fun(
        fun, prox.prox_lasso)
    vjp_fun = lambda g: implicit_diff.fixed_point_vjp(
        fixed_point_fun=fixed_point_fun,
        sol=sol,
        params=(1.0, lam),
        cotangent=g)[1]
    jac_params_prox_from_vjp = jax.vmap(vjp_fun)(I)
    self.assertArraysEqual(jac_params_prox_from_vjp.shape,
                           (n_features, n_features))

    # Compute the Jacobian w.r.t. lam (params_prox) using JVPs.
    fixed_point_fun2 = lambda sol, pp: fixed_point_fun(sol, (1.0, pp))
    jvp_fun = lambda g: implicit_diff.fixed_point_jvp(
        fixed_point_fun=fixed_point_fun2,
        sol=sol,
        params=lam,
        tangent=g)
    jac_params_prox_from_jvp = jax.vmap(jvp_fun)(I)
    self.assertArraysAllClose(
        jac_params_prox_from_jvp, jac_params_prox_from_vjp, atol=tol)

    # Make sure the decorator works.
    jac_custom = jax.jacrev(solver_fun, argnums=1)(1.0, lam)
    self.assertArraysAllClose(jac_params_prox_from_vjp, jac_custom, atol=tol)

  @parameterized.product(acceleration=[True, False])
  def test_lasso_forward_diff(self, acceleration):
    raise absltest.SkipTest
    X, y = datasets.load_boston(return_X_y=True)
    lam = 1.0
    tol = 1e-4 if acceleration else 5e-3
    maxiter = 200
    fun = test_util.make_least_squares_objective(X, y)

    jac_lam = test_util.lasso_skl_jac(X, y, lam)

    # Compute the Jacobian w.r.t. lam via forward differentiation.
    w_init = jnp.zeros(X.shape[1])
    solver_fun = proximal_gradient.make_solver_fun(fun=fun, init=w_init,
                                                   prox=prox.prox_lasso,
                                                   tol=tol, maxiter=maxiter,
                                                   implicit_diff=False,
                                                   acceleration=acceleration)
    jac_lam2 = jax.jacrev(solver_fun)(lam, 1.0)
    self.assertArraysAllClose(jac_lam, jac_lam2, atol=1e-3)

  def test_elastic_net(self):
    X, y = datasets.load_boston(return_X_y=True)
    params_prox = (2.0, 0.8)
    tol = 1e-3
    maxiter = 200
    atol = 1e-3
    fun = test_util.make_least_squares_objective(X, y)

    # Check optimality conditions.
    w_init = jnp.zeros(X.shape[1])
    solver_fun = proximal_gradient.make_solver_fun(fun=fun, init=w_init,
                                                   prox=prox.prox_elastic_net,
                                                   tol=tol, maxiter=maxiter)
    w_fit = solver_fun(params_fun=1.0, params_prox=params_prox)
    w_fit2 = prox.prox_elastic_net(w_fit - jax.grad(fun)(w_fit, 1.0),
                                   params_prox)
    w_diff = tu.tree_sub(w_fit, w_fit2)
    self.assertLessEqual(jnp.sqrt(jnp.sum(w_diff**2)), tol)

    # Compare against sklearn.
    w_skl = test_util.enet_skl(X, y, params_prox)
    self.assertArraysAllClose(w_fit, w_skl, atol=atol)

  def test_elastic_net_implicit_diff(self):
    """Test implicit differentiation of a single lambda parameter."""
    X, y = datasets.load_boston(return_X_y=True)
    params_prox = (2.0, 0.8)
    tol = 1e-3
    maxiter = 200
    fun = test_util.make_least_squares_objective(X, y)

    jac_num_lam, jac_num_gam = test_util.enet_skl_jac(X, y, params_prox)
    w_skl = test_util.enet_skl(X, y, params_prox)
    solver_fun = proximal_gradient.make_solver_fun(fun=fun, init=w_skl,
                                                   prox=prox.prox_elastic_net,
                                                   tol=tol, maxiter=maxiter,
                                                   acceleration=True,
                                                   implicit_diff=True)
    jac_custom = jax.jacrev(solver_fun, argnums=1)(1.0, params_prox)
    self.assertArraysAllClose(jac_num_lam, jac_custom[0], atol=1e-3)
    self.assertArraysAllClose(jac_num_gam, jac_custom[1], atol=1e-3)

  def test_multiclass_svm_dual(self):
    X, y = datasets.make_classification(
        n_samples=20,
        n_features=5,
        n_informative=3,
        n_classes=3,
        random_state=0)
    # Transform labels to a one-hot representation.
    # Y has shape (n_samples, n_classes).
    Y = preprocessing.LabelBinarizer().fit_transform(y)
    lam = 10.0
    tol = 1e-2
    maxiter = 500
    atol = 1e-2
    fun = test_util.make_multiclass_linear_svm_objective(X, y)

    proj_vmap = jax.vmap(projection.projection_simplex)
    _prox = lambda x, params_prox, scaling=1.0: proj_vmap(x)

    n_samples, n_classes = Y.shape
    beta_init = jnp.ones((n_samples, n_classes)) / n_classes
    solver_fun = proximal_gradient.make_solver_fun(fun=fun, init=beta_init,
                                                   prox=_prox, stepsize=1e-2,
                                                   tol=tol, maxiter=maxiter)
    beta_fit = solver_fun(params_fun=lam)

    # Check optimality conditions.
    beta_fit2 = proj_vmap(beta_fit - jax.grad(fun)(beta_fit, lam))
    self.assertLessEqual(jnp.sqrt(jnp.sum((beta_fit - beta_fit2)**2)), tol)

    # Compare against sklearn.
    W_skl = test_util.multiclass_linear_svm_skl(X, y, lam)
    W_fit = jnp.dot(X.T, (Y - beta_fit)) / lam
    self.assertArraysAllClose(W_fit, W_skl, atol=atol)

  def test_multiclass_svm_dual_implicit_diff(self):
    X, y = datasets.make_classification(
        n_samples=20,
        n_features=5,
        n_informative=3,
        n_classes=3,
        random_state=0)
    X = preprocessing.Normalizer().fit_transform(X)
    # Transform labels to a one-hot representation.
    # Y has shape (n_samples, n_classes).
    Y = preprocessing.LabelBinarizer().fit_transform(y)
    lam = 10.0
    tol = 1e-3
    maxiter = 500
    atol = 1e-2
    fun = test_util.make_multiclass_linear_svm_objective(X, y)

    proj_vmap = jax.vmap(projection.projection_simplex)
    _prox = lambda x, params_prox, scaling=1.0: proj_vmap(x)

    def proximal_gradient_fun_dual(lam):
      n_samples, n_classes = Y.shape
      beta_init = jnp.ones((n_samples, n_classes)) / n_classes
      solver_fun = proximal_gradient.make_solver_fun(fun=fun, init= beta_init,
                                                     prox=_prox, stepsize=1e-2,
                                                     tol=tol, maxiter=maxiter)
      return solver_fun(params_fun=lam)

    def proximal_gradient_fun_primal(lam):
      beta_fit = proximal_gradient_fun_dual(lam)
      return jnp.dot(X.T, (Y - beta_fit)) / lam

    jac_primal = jax.jacrev(proximal_gradient_fun_primal)(lam)
    jac_num = test_util.multiclass_linear_svm_skl_jac(X, y, lam, eps=1e-3)
    self.assertArraysAllClose(jac_num, jac_primal, atol=5e-3)

  def test_jit_and_vmap(self):
    make_solver_fun = functools.partial(proximal_gradient.make_solver_fun,
                                        prox=prox.prox_lasso)
    make_fixed_point_fun = functools.partial(
        implicit_diff.make_proximal_gradient_fixed_point_fun,
        prox=prox.prox_lasso)
    # A list of (params_fun, params_prox) pairs.
    params_list = jnp.array([[1.0, 1.0], [1.0, 10.0]])
    test_util.test_logreg_jit_and_vmap(self, make_solver_fun,
                                       make_fixed_point_fun, params_list,
                                       unpack_params=True)

if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  # jax.config.update("jax_enable_x64", True)
  absltest.main(testLoader=jtu.JaxTestLoader())
