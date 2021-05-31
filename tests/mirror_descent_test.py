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

from typing import Any, Callable, Optional

from absl.testing import absltest
import jax
from jax import test_util as jtu
import jax.numpy as jnp
from jaxopt import mirror_descent
from jaxopt import test_util
from jaxopt import tree_util as tu
from sklearn import datasets
from sklearn import preprocessing


def kl_projection(x: Any, params_proj: Optional[Any] = None) -> Any:
  del params_proj
  return tu.tree_map(lambda t: jax.nn.softmax(t, -1), x)


# Generating function of the Bregman divergence.
kl_generating_fun = lambda x: -jnp.sum(jax.scipy.special.entr(x) + x)
# Row-wise mirror map.
kl_mapping_fun = jax.vmap(jax.grad(kl_generating_fun))


def make_stepsize_schedule(max_stepsize, n_steps, power=1.0) -> Callable:
  def stepsize_schedule(t: int) -> float:
    true_fn = lambda t: 1.0
    false_fn = lambda t: (n_steps / t) ** power
    decay_factor = jax.lax.cond(t <= n_steps, true_fn, false_fn, t)
    return decay_factor * max_stepsize
  return stepsize_schedule


class MirrorDescentTest(jtu.JaxTestCase):

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
    stepsize_schedule = make_stepsize_schedule(
        max_stepsize=5.0, n_steps=50, power=1.0)
    tol = 1e-2
    maxiter = 500
    atol = 1e-2
    fun = test_util.make_multiclass_linear_svm_objective(X, y)

    n_samples, n_classes = Y.shape
    beta_init = jnp.ones((n_samples, n_classes)) / n_classes
    solver_fun = mirror_descent.make_solver_fun(
        fun=fun,
        projection=kl_projection,
        mapping_fun=kl_mapping_fun,
        init=beta_init,
        stepsize=stepsize_schedule,
        tol=tol,
        maxiter=maxiter)
    beta_fit = solver_fun(params_fun=lam)

    # Check optimality conditions.
    beta_fit2 = kl_projection(
        kl_mapping_fun(beta_fit) - jax.grad(fun)(beta_fit, lam))
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
    stepsize_schedule = make_stepsize_schedule(
        max_stepsize=5.0, n_steps=50, power=1.0)
    tol = 1e-3
    maxiter = 500
    fun = test_util.make_multiclass_linear_svm_objective(X, y)

    def mirror_descent_fun_dual(lam):
      n_samples, n_classes = Y.shape
      beta_init = jnp.ones((n_samples, n_classes)) / n_classes
      solver_fun = mirror_descent.make_solver_fun(
          fun=fun,
          projection=kl_projection,
          mapping_fun=kl_mapping_fun,
          init=beta_init,
          stepsize=stepsize_schedule,
          tol=tol,
          maxiter=maxiter)
      return solver_fun(params_fun=lam)

    def mirror_descent_fun_primal(lam):
      beta_fit = mirror_descent_fun_dual(lam)
      return jnp.dot(X.T, (Y - beta_fit)) / lam

    jac_primal = jax.jacrev(mirror_descent_fun_primal)(lam)
    jac_num = test_util.multiclass_linear_svm_skl_jac(X, y, lam, eps=1e-3)
    self.assertArraysAllClose(jac_num, jac_primal, atol=5e-3)


if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  # jax.config.update("jax_enable_x64", True)
  absltest.main(testLoader=jtu.JaxTestLoader())
