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
from absl.testing import parameterized

import jax
import jax.numpy as jnp

import numpy as onp

from jaxopt import BFGS
from jaxopt import BacktrackingLineSearch
from jaxopt import objective
from jaxopt._src import test_util


from sklearn import datasets


N_CALLS = 0

def _bfgs(fun, init, stepsize=1e-3, maxiter=500, tol=1e-3):
  value_and_grad_fun = jax.value_and_grad(fun)

  x = init
  I = jnp.eye(len(init))
  H = I
  value, grad = value_and_grad_fun(init)

  for it in range(maxiter):
    direction = -jnp.dot(H, grad)
    x_old, grad_old = x, grad
    x = x + stepsize * direction
    value, grad = value_and_grad_fun(x)

    s = x - x_old
    y = grad - grad_old
    rho = 1. / jnp.dot(s, y)

    sy = jnp.outer(s, y)
    ss = jnp.outer(s, s)
    new_H = (I - rho * sy).dot(H).dot(I - rho * sy.T) + rho * ss
    H = jnp.where(jnp.isfinite(rho), new_H, H)

    grad_norm = jnp.sqrt(jnp.sum(grad ** 2))

    if grad_norm <= tol:
      break

  return x


class BfgsTest(test_util.JaxoptTestCase):

  def test_correctness(self):
    def fun(x, *args, **kwargs):  # Rosenbrock function.
      return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

    x0 = jnp.zeros(2)
    bfgs = BFGS(fun=fun, tol=1e-3, stepsize=1e-3, maxiter=15)
    x1, _ = bfgs.run(x0)

    x2 = _bfgs(fun, x0, stepsize=1e-3, maxiter=15)

    self.assertArraysAllClose(x1, x2, atol=1e-5)

  @parameterized.product(linesearch=["backtracking", "zoom"])
  def test_binary_logreg(self, linesearch):
    X, y = datasets.make_classification(
        n_samples=10, n_features=5, n_classes=2, n_informative=3, random_state=0
    )
    data = (X, y)
    fun = objective.binary_logreg

    w_init = jnp.zeros(X.shape[1])
    bfgs = BFGS(fun=fun, tol=1e-3, maxiter=500, linesearch=linesearch)
    # Test with keyword argument.
    w_fit, info = bfgs.run(w_init, data=data)

    # Check optimality conditions.
    self.assertLessEqual(info.error, 5e-2)

    # Compare against sklearn.
    w_skl = test_util.logreg_skl(X, y, 1e-6, fit_intercept=False,
                                 multiclass=False)
    self.assertArraysAllClose(w_fit, w_skl, atol=5e-2)

  @parameterized.product(
      linesearch=[
          "backtracking",
          "zoom",
          BacktrackingLineSearch(objective.multiclass_logreg_with_intercept),
      ],
      linesearch_init=["max", "current", "increase"]
  )
  def test_multiclass_logreg(self, linesearch, linesearch_init):
    data = datasets.make_classification(
        n_samples=10, n_features=5, n_classes=3, n_informative=3, random_state=0
    )
    fun = objective.multiclass_logreg_with_intercept

    w_init = jnp.zeros((data[0].shape[1], 3))
    b_init = jnp.zeros(3)
    pytree_init = (w_init, b_init)

    bfgs = BFGS(
        fun=fun,
        tol=1e-3,
        maxiter=500,
        linesearch=linesearch,
        linesearch_init=linesearch_init,
    )
    # Test with positional argument.
    _, info = bfgs.run(pytree_init, data)

    # Check optimality conditions.
    self.assertLessEqual(info.error, 1e-2)

  @parameterized.product(n_iter=[10])
  def test_n_calls(self, n_iter):
    """Test whether the number of function calls
    is equal to the number of iterations + 1 in the
    no linesearch case, where the complexity is linear."""
    global N_CALLS
    N_CALLS = 0

    def fun(x, *args, **kwargs):  # Rosenbrock function.
      global N_CALLS
      N_CALLS += 1
      return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

    x0 = jnp.zeros(2)
    bfgs = BFGS(fun=fun, tol=1e-7, maxiter=n_iter, jit=False, stepsize=1e-3)
    x1, _ = bfgs.run(x0)

    self.assertEqual(N_CALLS, n_iter + 1)


if __name__ == '__main__':
  absltest.main()
