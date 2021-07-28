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
from absl.testing import parameterized

import jax
from jax import test_util as jtu
import jax.numpy as jnp

from jaxopt import MirrorDescent
from jaxopt import objectives
from jaxopt._src import test_util
from jaxopt import tree_util as tu

from sklearn import datasets
from sklearn import preprocessing


def kl_projection(x: Any, hyperparams_proj: Optional[Any] = None) -> Any:
  del hyperparams_proj
  return tu.tree_map(lambda t: jax.nn.softmax(t, -1), x)


# Generating function of the Bregman divergence.
kl_generating_fun = lambda x: -jnp.sum(jax.scipy.special.entr(x) + x)
# Row-wise mirror map.
kl_mapping_fun = jax.vmap(jax.grad(kl_generating_fun))


def projection_grad_kl_stable(x: Any,
                              x_fun_grad: Any,
                              stepsize: float,
                              hyperparams_proj: Optional[Any] = None) -> Any:
  del hyperparams_proj
  # TODO(fllinares): generalize to arbitrary axes. How to pass axes?
  def _fn(x_i, g_i):
    g_i = jnp.where(x_i != 0, -stepsize * g_i, -jnp.inf)
    g_i = g_i - jnp.max(g_i, axis=-1, keepdims=True)
    y_i = x_i * jnp.exp(g_i)
    return y_i / jnp.sum(y_i, axis=-1, keepdims=True)
  return tu.tree_multimap(_fn, x, x_fun_grad)


def make_stepsize_schedule(max_stepsize, n_steps, power=1.0) -> Callable:
  def stepsize_schedule(t: int) -> float:
    true_fn = lambda t: 1.0
    false_fn = lambda t: (n_steps / t) ** power
    decay_factor = jax.lax.cond(t <= n_steps, true_fn, false_fn, t)
    return decay_factor * max_stepsize
  return stepsize_schedule


class MirrorDescentTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
      ('kl', None),
      ('kl_stable', projection_grad_kl_stable),
  )
  def test_multiclass_svm_dual(self, projection_grad):
    X, y = datasets.make_classification(
        n_samples=20,
        n_features=5,
        n_informative=3,
        n_classes=3,
        random_state=0)
    Y = preprocessing.LabelBinarizer().fit_transform(y)
    Y = jnp.asarray(Y)
    n_samples, n_classes = Y.shape
    fun = objectives.multiclass_linear_svm_dual
    lam = 10.0
    data = (X, Y)
    stepsize_schedule = make_stepsize_schedule(
        max_stepsize=5.0, n_steps=50, power=1.0)
    maxiter = 500
    tol = 1e-2
    atol = 1e-2

    beta_init = jnp.ones((n_samples, n_classes)) / n_classes
    if projection_grad is None:
      projection_grad = MirrorDescent.make_projection_grad(
          kl_projection, kl_mapping_fun)
    md = MirrorDescent(
        fun=fun,
        projection_grad=projection_grad,
        stepsize=stepsize_schedule,
        maxiter=maxiter,
        tol=tol,
        implicit_diff=True)
    beta_fit, info = md.run(beta_init, None, lam, data)

    # Check optimality conditions.
    self.assertLessEqual(info.error, tol)

    # Compare against sklearn.
    W_skl = test_util.multiclass_linear_svm_skl(X, y, lam)
    W_fit = jnp.dot(X.T, (Y - beta_fit)) / lam
    self.assertArraysAllClose(W_fit, W_skl, atol=atol)

  @parameterized.named_parameters(
      ('kl', None),
      ('kl_stable', projection_grad_kl_stable),
  )
  def test_multiclass_svm_dual_implicit_diff(self, projection_grad):
    X, y = datasets.make_classification(
        n_samples=20,
        n_features=5,
        n_informative=3,
        n_classes=3,
        random_state=0)
    X = preprocessing.Normalizer().fit_transform(X)
    Y = preprocessing.LabelBinarizer().fit_transform(y)
    Y = jnp.asarray(Y)
    n_samples, n_classes = Y.shape
    fun = objectives.multiclass_linear_svm_dual
    lam = 10.0
    data = (X, Y)
    stepsize_schedule = make_stepsize_schedule(
        max_stepsize=5.0, n_steps=50, power=1.0)
    maxiter = 500
    tol = 1e-3

    beta_init = jnp.ones((n_samples, n_classes)) / n_classes
    if projection_grad is None:
      projection_grad = MirrorDescent.make_projection_grad(
          kl_projection, kl_mapping_fun)
    md = MirrorDescent(
        fun=fun,
        projection_grad=projection_grad,
        stepsize=stepsize_schedule,
        maxiter=maxiter,
        tol=tol,
        implicit_diff=True)

    def mirror_descent_fun_primal(lam):
      beta_fit, _ = md.run(beta_init, None, lam, data)
      return jnp.dot(X.T, (Y - beta_fit)) / lam

    jac_primal = jax.jacrev(mirror_descent_fun_primal)(lam)
    jac_num = test_util.multiclass_linear_svm_skl_jac(X, y, lam, eps=1e-3)
    self.assertArraysAllClose(jac_num, jac_primal, atol=5e-3)


if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  # jax.config.update("jax_enable_x64", True)
  absltest.main(testLoader=jtu.JaxTestLoader())
