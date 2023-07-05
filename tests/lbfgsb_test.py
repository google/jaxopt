# Copyright 2023 Google LLC
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

"""Tests for L-BFGS-B."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jaxopt import LBFGSB
from jaxopt import BacktrackingLineSearch
from jaxopt import objective
from jaxopt import OptStep
from jaxopt import ScipyBoundedMinimize
from jaxopt._src import test_util
import numpy as onp

from sklearn import datasets


class LbfgsbTest(test_util.JaxoptTestCase):

  @parameterized.parameters(
      ((0.8, 1.2), (-1.0, -2.0), (2.0, 2.0), True),
      ((2.0, 0.0), (1.5, -2.0), (3.0, 0.5), False),
      ((2.0, 0.5), (0.0, -1.0), (3.0, 1.0), "function"),
  )
  def test_correctness(self, init_params, lower, upper, value_and_grad):
    def fun(x):  # Rosenbrock function.
      return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
    if value_and_grad is True:
      fun = jax.value_and_grad(fun)
    elif value_and_grad == "function":
      value_and_grad = jax.value_and_grad(fun)

    x0 = jnp.array(init_params)
    bounds = (jnp.array(lower), jnp.array(upper))
    lbfgsb = LBFGSB(
        fun=fun,
        tol=1e-5,
        stepsize=-1.0,
        maxiter=100,
        history_size=5,
        use_gamma=True,
        value_and_grad=value_and_grad,
    )
    x1, _ = lbfgsb.run(x0, bounds=bounds)

    scipy_lbfgsb = ScipyBoundedMinimize(
        fun=fun,
        tol=1e-5,
        maxiter=100,
        method="L-BFGS-B",
        options={"maxcor": 5},
        value_and_grad=value_and_grad,
    )
    x2, _ = scipy_lbfgsb.run(init_params=x0, bounds=bounds)

    self.assertArraysAllClose(x1, x2, atol=5e-5)

  @parameterized.product(
      implicit_diff=[True, False],
      linesearch=["backtracking", "zoom", "hager-zhang"],
  )
  def test_Rosenbrock(self, implicit_diff, linesearch):

    def fun(x):  # Rosenbrock function.
      return 100.0 * (x["b"] - x["a"]**2.0)**2.0 + (1 - x["a"])**2.0

    x0 = {"a": jnp.zeros([]), "b": jnp.zeros([])}
    upper = {"a": jnp.asarray(2.), "b": jnp.asarray(3.)}
    lower = {"a": jnp.asarray(-1), "b": jnp.asarray(-0.5)}
    lbfgsb = LBFGSB(
        fun=fun,
        tol=1e-3,
        stepsize=-1.,
        maxiter=500,
        linesearch=linesearch,
        implicit_diff=implicit_diff)
    x, _ = lbfgsb.run(x0, bounds=(lower, upper))

    # the Rosenbrock function is zero at its minimum
    self.assertLessEqual(fun(x), 1e-3)

  @parameterized.parameters(
      ((0., -5., 0), (2., 0., 1)),
      ((0., 0., 0.), (1.5, 1., 0.4)),
  )
  def test_quadratic_bowl(self, lower, upper):
    x0 = jnp.zeros(3)
    lower = jnp.array(lower)
    upper = jnp.array(upper)
    unconstrained_optimum = jnp.array([1., -2., 0.5])
    optimum = jnp.maximum(jnp.minimum(unconstrained_optimum, upper), lower)

    f = lambda x: jnp.sum((x - unconstrained_optimum)**2)

    lbfgsb = LBFGSB(
        fun=f,
        tol=1e-3,
        maxiter=20,
        stepsize=-1.,
        linesearch="zoom")
    x, state = lbfgsb.run(x0, bounds=(lower, upper))
    self.assertLess(onp.abs(x - optimum).max(), 1e-6)

    # quadratic optimum should be found almost immediately.
    self.assertLessEqual(state.iter_num, 2)

    # Check that bounds are satisfied.
    self.assertTrue(((x >= lower) & (x <= upper)).all())

  @parameterized.product(
      use_gamma=[True, False],
      linesearch=["backtracking", "zoom", "hager-zhang"],
  )
  def test_binary_logreg(self, use_gamma, linesearch):
    x, y = datasets.make_classification(
        n_samples=10, n_features=5, n_classes=2, n_informative=3, random_state=0
    )
    data = (x, y)
    fun = objective.binary_logreg

    w_init = jnp.zeros(x.shape[1])
    lbfgsb = LBFGSB(
        fun=fun,
        tol=1e-3,
        maxiter=500,
        use_gamma=use_gamma,
        linesearch=linesearch,
    )
    # Test with keyword argument.
    w_fit, info = lbfgsb.run(w_init, bounds=None, data=data)

    # Check optimality conditions.
    self.assertLessEqual(info.error, 5e-2)

    # Compare against sklearn.
    w_skl = test_util.logreg_skl(
        x, y, 1e-6, fit_intercept=False, multiclass=False
    )
    self.assertArraysAllClose(w_fit, w_skl, atol=5e-2)

  @parameterized.product(
      use_gamma=[True, False],
      linesearch=[
          "backtracking",
          "zoom",
          "hager-zhang",
          BacktrackingLineSearch(objective.multiclass_logreg_with_intercept),
      ],
  )
  def test_multiclass_logreg(self, use_gamma, linesearch):
    x, y = datasets.make_classification(
        n_samples=10, n_features=5, n_classes=3, n_informative=3, random_state=0
    )
    data = (x, y)
    fun = objective.multiclass_logreg_with_intercept

    w_init = jnp.zeros((x.shape[1], 3))
    b_init = jnp.zeros(3)
    pytree_init = (w_init, b_init)

    u = 1e2
    l = -1e2
    upper = (u * jnp.ones_like(w_init), u * jnp.ones_like(b_init))
    lower = (l * jnp.ones_like(w_init), l * jnp.ones_like(b_init))
    lbfgsb = LBFGSB(
        fun=fun,
        tol=1e-3,
        maxiter=500,
        use_gamma=use_gamma,
        linesearch=linesearch,
    )
    # Test with positional argument.
    _, info = lbfgsb.run(pytree_init, (lower, upper), data)

    # Check optimality conditions.
    self.assertLessEqual(info.error, 1e-2)

  def test_warm_start_hessian_approx(self):
    def fun(x):
      return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

    x0 = jnp.zeros(2)
    lbfgsb = LBFGSB(
        fun=fun,
        tol=1e-3,
        maxiter=2,
        stepsize=1e-3,
    )
    x1, lbfgs_state = lbfgsb.run(x0, bounds=None)

    # warm start with the previous solution
    lbfgsb.run(OptStep(x1, lbfgs_state), bounds=None)

  @parameterized.product(out_dtype=[jnp.float32, jnp.float64])
  def test_correct_dtypes(self, out_dtype):
    def f(x):
      return jnp.cos(jnp.sum(jnp.exp(-x)) ** 2).astype(out_dtype)

    with jax.experimental.enable_x64():
      x0 = jnp.ones([5, 5], dtype=jnp.float32)
      lbfgs = LBFGSB(fun=f, tol=1e-3, maxiter=500)
      x, state = lbfgs.run(x0, bounds=None)
      self.assertEqual(x.dtype, jnp.float32)
      for name in ("iter_num",):
        self.assertEqual(getattr(state, name).dtype, jnp.int64)
      for name in ("error", "s_history", "y_history", "grad"):
        self.assertEqual(getattr(state, name).dtype, jnp.float32, name)
      for name in ("value",):
        self.assertEqual(getattr(state, name).dtype, out_dtype, name)

  @parameterized.product(n_iter=[10])
  def test_n_calls(self, n_iter):
    # Test whether the number of function calls is equal to the number of
    # iterations + 1 in the no linesearch case, where the complexity is linear.
    # pylint: disable=global-variable-undefined
    global N_CALLS
    N_CALLS = 0

    def fun(x):
      global N_CALLS
      N_CALLS += 1
      return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

    x0 = jnp.zeros(2)
    lbfgsb = LBFGSB(
        fun=fun,
        tol=1e-7,
        maxiter=n_iter,
        stepsize=1e-3,
        jit=False,
    )
    lbfgsb.run(x0, bounds=None)

    self.assertEqual(N_CALLS, n_iter + 1)

  def test_grad_with_bounds(self):
    # Test that the gradient is correct when bounds are specified by keyword.
    # Pertinent to issue #463.
    def pipeline(x, init_pars, bounds, data):
      def fit_objective(pars, data, x):
        return -jax.scipy.stats.norm.logpdf(pars, loc=data*x, scale=1.0)
      solver = LBFGSB(fun=fit_objective, implicit_diff=True, maxiter=500, tol=1e-6)
      return solver.run(init_pars, bounds=bounds, data=data, x=x)[0]

    grad_fn = jax.grad(pipeline)
    data = jnp.array(1.5)
    res = grad_fn(0.5, jnp.array(0.0), (jnp.array(0.0), jnp.array(10.0)), data)
    self.assertEqual(res, data)
      


if __name__ == "__main__":
  absltest.main()
