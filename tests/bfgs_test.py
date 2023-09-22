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

import unittest
from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp

from jaxopt import BFGS
from jaxopt import BacktrackingLineSearch
from jaxopt import objective
from jaxopt._src import test_util
import numpy as onp
import scipy.optimize as scipy_opt
from sklearn import datasets



N_CALLS = 0

# Uncomment this line to test in x64 
# jax.config.update('jax_enable_x64', True)

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


def get_fun(name, np):

  def rosenbrock(x):
    return np.sum(100. * np.diff(x) ** 2 + (1. - x[:-1]) ** 2)

  def himmelblau(p):
    x, y = p
    return (x ** 2 + y - 11.) ** 2 + (x + y ** 2 - 7.) ** 2

  def matyas(p):
    x, y = p
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

  def eggholder(p):
    x, y = p
    return - (y + 47) * np.sin(np.sqrt(np.abs(x / 2. + y + 47.))) - x * np.sin(
      np.sqrt(np.abs(x - (y + 47.))))

  def zakharov(x):
    ii = np.arange(1, len(x) + 1, step=1, dtype=x.dtype)
    sum1 = (x**2).sum()
    sum2 = (0.5*ii*x).sum()
    answer = sum1+sum2**2+sum2**4
    return answer
  
  funs = dict(
    rosenbrock=rosenbrock,
    himmelblau=himmelblau,
    matyas=matyas,
    eggholder=eggholder,
    zakharov=zakharov
  )
  return funs[name]

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


  @parameterized.product(
    fun_init_and_opt=[
      ('rosenbrock', onp.zeros(2, dtype='float32'), 0.),
      ('himmelblau', onp.ones(2, dtype='float32'), 0.),
      ('matyas', onp.ones(2) * 6., 0.),
      ('eggholder', onp.ones(2) * 100., None),  
    ],
  )
  def test_against_scipy(self, fun_init_and_opt):
    # Taken from previous jax tests
    # NOTE(vroulet): Unclear whether lbfgs or bfgs can find true minimum for
    # eggholder, but they seem to converge to the same solution with current 
    # implementation and initialization, which is a good check.

    # high precision for faithful checks
    tol = 1e-15 if jax.config.jax_enable_x64 else 1e-6
    # jaxopt_opts = dict(maxls=100) if jax.config.jax_enable_x64 else {}
    fun_name, x0, opt = fun_init_and_opt
    jnp_fun, onp_fun = get_fun(fun_name, jnp), get_fun(fun_name, onp)
    jaxopt_res = BFGS(jnp_fun, tol=tol).run(x0).params
    scipy_res = scipy_opt.minimize(onp_fun, x0, method='BFGS').x
    if fun_name == 'matyas':
    # scipy not good for matyas function, compare to true minimum, zero
      self.assertAllClose(jaxopt_res, jnp.zeros_like(jaxopt_res), check_dtypes=False)
    elif fun_name == 'eggholder' and jax.config.jax_enable_x64:
      # NOTE(vroulet): slight issue at high precision here
      self.assertAllClose(jnp_fun(jaxopt_res), onp_fun(scipy_res),
                          atol=1e-11, rtol=1e-11, check_dtypes=False)
    else:
      self.assertAllClose(jaxopt_res, scipy_res, check_dtypes=False)

    if opt is not None:
      self.assertAllClose(jnp_fun(jaxopt_res), opt, check_dtypes=False)

  @unittest.skipIf(not jax.config.jax_enable_x64, 'test requires X64')
  def test_zakharov(self):
    # Taken from previous jax tests
    # Function to steep to work without high precision (curiously lbfgs works)
    x0 = jnp.array([600.0, 700.0, 200.0, 100.0, 90.0, 1e4])
    fun = get_fun('zakharov', jnp)
    jaxopt_res = BFGS(fun, tol=1e-16, maxls=50).run(x0).params
    self.assertAllClose(jaxopt_res, jnp.zeros_like(jaxopt_res))
    self.assertAllClose(fun(jaxopt_res), 0.)

  def test_minimize_bad_initial_values(self):
    # Taken from previous jax tests
    # This test runs deliberately "bad" initial values to test that handling
    # of failed line search, etc. is the same across implementations
    initial_value = onp.array([92, 0.001])
    tol = 1e-6 if jax.config.jax_enable_x64 else 1e-3
    jnp_fun, onp_fun = get_fun('himmelblau', jnp), get_fun('himmelblau', onp)
    # Here reuqires higher number of linesearch for jaxopt
    jaxopt_res = BFGS(jnp_fun, tol=tol).run(initial_value).params
    scipy_res = scipy_opt.minimize(
        fun=onp_fun,
        jac=jax.grad(onp_fun),
        method='BFGS',
        x0=initial_value
    ).x
    # Scipy and jaxopt converge to different minima so check in function values
    self.assertAllClose(onp_fun(scipy_res), jnp_fun(jaxopt_res))
    self.assertAllClose(jnp_fun(jaxopt_res), jnp.asarray(0.))

  def test_steep(self):
    # Taken from previous jax tests
    # See jax related issue https://github.com/google/jax/issues/4594
    n = 2
    A = jnp.eye(n) * 1e4
    def fun(x):
      return jnp.mean((A @ x) ** 2)
    results = BFGS(fun).run(init_params=jnp.ones(n)).params
    self.assertAllClose(results, jnp.zeros(n))


if __name__ == '__main__':
  absltest.main()
