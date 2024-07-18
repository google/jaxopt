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
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree
from jaxopt import BacktrackingLineSearch
from jaxopt import LBFGS
from jaxopt import GradientDescent
from jaxopt import objective
from jaxopt import OptStep
from jaxopt._src import test_util
from jaxopt._src.lbfgs import inv_hessian_product
from jaxopt._src.lbfgs import select_ith_tree
from jaxopt.tree_util import tree_sum
import numpy as onp
import scipy.optimize as scipy_opt
from sklearn import datasets


N_CALLS = 0

# Uncomment this line to test in x64 
# jax.config.update('jax_enable_x64', True)

def _materialize_inv_hessian(s_history, y_history, rho_history, start):
  history_size, n_dim = s_history.shape

  s_history = jnp.roll(s_history, -start, axis=0)
  y_history = jnp.roll(y_history, -start, axis=0)
  rho_history = jnp.roll(rho_history, -start, axis=0)

  I = jnp.eye(n_dim, n_dim)
  H = I

  for k in range(history_size):
    V = I - rho_history[k] * jnp.outer(y_history[k], s_history[k])
    H = (jnp.dot(V.T, jnp.dot(H, V)) +
         rho_history[k] * jnp.outer(s_history[k], s_history[k]))

  return H


def _inv_hessian_product(s, y, v, gamma=1.0):
  history_size = len(s)

  if history_size == 0:
    return v

  s = jnp.array(s)
  y = jnp.array(y)

  rho = jnp.zeros(history_size)
  alpha = jnp.zeros(history_size)

  # Compute right product.
  def body_fun(j, tup):
    rho, alpha, r = tup
    i = history_size - j - 1
    # rho[i] = 1. / jnp.sum(s[i] * y[i])
    rho = rho.at[i].set(1. / jnp.sum(s[i] * y[i]))
    # alpha[i] = rho[i] * jnp.sum(s[i] * r)
    alpha = alpha.at[i].set(rho[i] * jnp.sum(s[i] * r))
    r = r - alpha[i] * y[i]
    return rho, alpha, r

  # for i in reversed(range(history_size)):
  rho, alpha, r = jax.lax.fori_loop(0, history_size, body_fun, (rho, alpha, v))

  r = r * gamma

  # Compute left product.
  def body_fun2(i, r):
    beta = rho[i] * jnp.sum(y[i] * r)
    return r + s[i] * (alpha[i] - beta)

  # for i in range(history_size):
  r = jax.lax.fori_loop(0, history_size, body_fun2, r)

  return r


def _lbfgs(fun, init, stepsize=1e-3, maxiter=500, tol=1e-3,
           verbose=0, history_size=10, use_gamma=True):

  value_and_grad_fun = jax.value_and_grad(fun)

  x = init
  value, grad = value_and_grad_fun(init)
  s_history = []
  y_history = []

  for it in range(maxiter):
    if use_gamma and it > 0:
      gamma = jnp.vdot(y_history[-1], s_history[-1])
      gamma /= jnp.sum(y_history[-1] ** 2)
    else:
      gamma = 1.0

    direction = -_inv_hessian_product(s_history, y_history, grad, gamma)
    x_old, grad_old = x, grad
    x = x + stepsize * direction
    value, grad = value_and_grad_fun(x)

    s_history.append(x - x_old)
    y_history.append(grad - grad_old)

    if len(s_history) > history_size:
      s_history = s_history[1:]  # Pop left.
      y_history = y_history[1:]

    grad_norm = jnp.sqrt(jnp.sum(grad ** 2))

    if verbose:
      print("iter %d" % it, "f:", value, "grad:", grad_norm)

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


class LbfgsTest(test_util.JaxoptTestCase):

  def test_inv_hessian_product_array(self):
    rng = onp.random.RandomState(0)
    history_size = 4
    d = 3
    s_history = jnp.array(rng.randn(history_size, d))
    y_history = jnp.array(rng.randn(history_size, d))
    rho_history = 1.0 / jnp.sum(s_history * y_history, axis=1)
    v = jnp.array(rng.randn(d))

    Hv1 = _inv_hessian_product(s_history, y_history, v)

    H = _materialize_inv_hessian(s_history, y_history, rho_history, 0)
    Hv2 = H.dot(v)

    Hv3 = inv_hessian_product(v, s_history, y_history, rho_history, start=0)

    self.assertArraysAllClose(Hv1, Hv2, atol=1e-2)
    self.assertArraysAllClose(Hv1, Hv3, atol=1e-2)

  @parameterized.product(start=[0, 1, 2, 3])
  def test_inv_hessian_product_pytree(self, start):
    rng = onp.random.RandomState(0)
    history_size = 4
    shape1 = (3, 2)
    shape2 = (5,)

    s_history1 = jnp.array(rng.randn(history_size, *shape1))
    s_history2 = jnp.array(rng.randn(history_size, *shape2))

    y_history1 = jnp.array(rng.randn(history_size, *shape1))
    y_history2 = jnp.array(rng.randn(history_size, *shape2))

    v1 = jnp.array(rng.randn(*shape1))
    v2 = jnp.array(rng.randn(*shape2))

    s_history = (s_history1, s_history2)
    y_history = (y_history1, y_history2)
    pytree = (v1, v2)

    v = ravel_pytree(pytree)[0]
    s = [ravel_pytree(select_ith_tree(s_history, i))[0]
          for i in range(history_size)]
    y = [ravel_pytree(select_ith_tree(y_history, i))[0]
          for i in range(history_size)]
    s, y = jnp.stack(s), jnp.stack(y)

    inv_rho_history = [jnp.vdot(s[i], y[i])
                       for i in range(history_size)]
    rho_history = 1./ jnp.array(inv_rho_history)

    Hv = inv_hessian_product(pytree, s_history, y_history, rho_history,
                             start=start)
    Hv = ravel_pytree(Hv)[0]

    H = _materialize_inv_hessian(s, y, rho_history, start)
    Hv_mat = jnp.dot(H, v)

    self.assertArraysAllClose(Hv, Hv_mat, atol=1e-5, rtol=1e-5)

  @parameterized.product(use_gamma=[True, False])
  def test_correctness(self, use_gamma):
    def fun(x, *args, **kwargs):  # Rosenbrock function.
      return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

    x0 = jnp.zeros(2)
    lbfgs = LBFGS(fun=fun, tol=1e-3, stepsize=1e-3, maxiter=15, history_size=5,
                  use_gamma=use_gamma)
    x1, _ = lbfgs.run(x0)

    x2 = _lbfgs(fun, x0, stepsize=1e-3, maxiter=15, history_size=5,
                use_gamma=use_gamma)

    self.assertArraysAllClose(x1, x2, atol=1e-5)

  @parameterized.product(
      use_gamma=[True, False],
      linesearch=[
          "backtracking",
          "zoom",
          "hager-zhang",
          BacktrackingLineSearch(objective.binary_logreg, decrease_factor=0.5),
      ],
      linesearch_init=["max", "current", "increase"]
  )
  def test_binary_logreg(self, use_gamma, linesearch, linesearch_init):
    X, y = datasets.make_classification(
        n_samples=10, n_features=5, n_classes=2, n_informative=3, random_state=0
    )
    data = (X, y)
    fun = objective.binary_logreg

    w_init = jnp.zeros(X.shape[1])
    lbfgs = LBFGS(fun=fun, tol=1e-3, maxiter=500, use_gamma=use_gamma,
                  linesearch=linesearch, linesearch_init=linesearch_init)
    # Test with keyword argument.
    w_fit, info = lbfgs.run(w_init, data=data)

    # Check optimality conditions.
    self.assertLessEqual(info.error, 5e-2)

    # Compare against sklearn.
    w_skl = test_util.logreg_skl(X, y, 1e-6, fit_intercept=False,
                                 multiclass=False)
    self.assertArraysAllClose(w_fit, w_skl, atol=5e-2)

  @parameterized.product(
      use_gamma=[True, False],
      linesearch=[
          "backtracking",
          "zoom",
          "hager-zhang",
          BacktrackingLineSearch(
              objective.multiclass_logreg_with_intercept, decrease_factor=0.5
          ),
      ],
  )
  def test_multiclass_logreg(self, use_gamma, linesearch):
    data = datasets.make_classification(
        n_samples=10, n_features=5, n_classes=3, n_informative=3, random_state=0
    )
    fun = objective.multiclass_logreg_with_intercept

    w_init = jnp.zeros((data[0].shape[1], 3))
    b_init = jnp.zeros(3)
    pytree_init = (w_init, b_init)

    lbfgs = LBFGS(fun=fun, tol=1e-3, maxiter=500, use_gamma=use_gamma,
                  linesearch=linesearch)
    # Test with positional argument.
    _, info = lbfgs.run(pytree_init, data)

    # Check optimality conditions.
    self.assertLessEqual(info.error, 1e-2)

  @parameterized.product(implicit_diff=[True, False])
  def test_implicit_diff(self, implicit_diff): 
    fun = get_fun('rosenbrock', jnp)
    x0 = jnp.zeros(2)
    lbfgs = LBFGS(fun=fun, tol=1e-3, maxiter=500, implicit_diff=implicit_diff)
    x, _ = lbfgs.run(x0)

    # the Rosenbrock function is zero at its minimum
    self.assertLessEqual(fun(x), 1e-2)

  def test_warm_start_hessian_approx(self):
    def fun(x, *args, **kwargs):
      return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

    x0 = jnp.zeros(2)
    lbfgs = LBFGS(fun=fun, tol=1e-3, maxiter=2, stepsize=1e-3)
    x1, lbfgs_state = lbfgs.run(x0)

    # warm start with the previous solution
    x2, _ = lbfgs.run(OptStep(x1, lbfgs_state))

  @parameterized.product(out_dtype=[jnp.float32, jnp.float64])
  def test_correct_dtypes(self, out_dtype):
    def f(x):
      return  jnp.cos(jnp.sum(jnp.exp(-x)) ** 2).astype(out_dtype)

    with jax.experimental.enable_x64():
      x0 = jnp.ones([5, 5], dtype=jnp.float32)
      lbfgs = LBFGS(fun=f, tol=1e-3, maxiter=500)
      x, state = lbfgs.run(x0)
      self.assertEqual(x.dtype, jnp.float32)
      for name in ("iter_num",):
        self.assertEqual(getattr(state, name).dtype, jnp.int64)
      for name in ("error", "s_history", "y_history", "rho_history"):
        self.assertEqual(getattr(state, name).dtype, jnp.float32, name)
      for name in ("value",):
        self.assertEqual(getattr(state, name).dtype, out_dtype, name)

  @parameterized.product(n_iter=[10])
  def test_n_calls(self, n_iter):
    """Test whether the number of function calls
    is equal to the number of iterations + 1 in the
    no linesearch case, where the complexity is linear."""
    global N_CALLS
    N_CALLS = 0

    def fun(x, *args, **kwargs):
      global N_CALLS
      N_CALLS += 1
      return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

    x0 = jnp.zeros(2)
    lbfgs = LBFGS(fun=fun, tol=1e-7, maxiter=n_iter, stepsize=1e-3, jit=False)
    x, _ = lbfgs.run(x0)

    self.assertEqual(N_CALLS, n_iter + 1)

  def test_binary_logit_log_likelihood(self):
    # See issue #409
    rng = jax.random.PRNGKey(42)
    N = 1000
    beta = jnp.array([[0.5,0.5]]).T
    income = jax.random.normal(rng, shape=(N,1))
    x = jnp.hstack([jnp.ones((N,1)), income])

    def simulate_binary_logit(x, beta):
      beta = beta.reshape(-1,1)
      N = x.shape[0]
      J = beta.shape[0]

      epsilon = jax.random.gumbel(rng,shape =(N,J))
      Beta_augmented = jnp.hstack([beta, jnp.zeros_like(beta)])
      utility = x @ Beta_augmented + epsilon

      choice_idx = onp.argmax(utility, axis=1)
      return (choice_idx).reshape(-1,1)

    y = simulate_binary_logit(x, beta)
    y = jnp.ravel(y)

    # numpy version
    def binary_logit_log_likelihood(beta, y,x):
      lambda_xb = onp.exp(x@beta) / (1 + onp.exp(x@beta))
      ll_i = y * onp.log(lambda_xb) + (1-y) * onp.log(1-lambda_xb)
      ll = -onp.sum(ll_i)
      return ll

    # jax version
    def binary_logit_log_likelihood_jax(beta, y, x):
      lambda_xb = jnp.exp(x@beta) / (1 + jnp.exp(x@beta))
      ll_i = y * jnp.log(lambda_xb) + (1-y) * jnp.log(1-lambda_xb)
      ll = -jnp.sum(ll_i)
      return ll

    beta_init = jnp.array([0.01,0.01])

    # using scipy
    scipy_res = scipy_opt.minimize(
      fun=binary_logit_log_likelihood, 
      args=(onp.asarray(y),onp.asarray(x)), 
      x0 = (onp.asarray(beta_init)), method='BFGS'
    ).x

    # using jaxopt
    solver = LBFGS(fun=binary_logit_log_likelihood_jax, maxiter=100,
                   linesearch="zoom", maxls=10, tol=1e-12)
    jaxopt_res = solver.run(beta_init, y, x).params

    # comparison
    scipy_val = binary_logit_log_likelihood(scipy_res,
                                            onp.asarray(y),
                                            onp.asarray(x))
    jaxopt_val = binary_logit_log_likelihood(jaxopt_res, y, x)
    self.assertArraysAllClose(scipy_val, jaxopt_val, rtol=3e-5)
    

  @parameterized.product(linesearch=['zoom', 'backtracking', 'hager-zhang'])
  def test_complex(self, linesearch):
    """Test that optimization over complex variable z = x + jy matches equivalent real case"""
    # NOTE(vroulet): At high precision, the results differ slightly 
    # (tol=5*1e-15 instead of tol=1e-15 by default at double precision)
    tol = 5*1e-15 if jax.config.jax_enable_x64 else None

    W = jnp.array(
      [[1, - 2],
       [3, 4],
       [-4 + 2j, 5 - 3j],
       [-2 - 2j, 6]]
    )

    def C2R(z):
      return jnp.stack((z.real, z.imag))

    def R2C(x):
      return x[..., 0, :] + 1j * x[..., 1, :]  # if len(x)>0 else jnp.zeros_like(x)

    def f(z):
      return W @ z

    def loss_complex(z):
      return jnp.sum(jnp.abs(f(z)) ** 1.5)

    def loss_real(zR):
      return loss_complex(R2C(zR))

    z0 = jnp.array([1 - 1j, 0 + 1j])

    solver_options = dict(maxiter=5,
                          history_size=3,
                          stepsize=-1,
                          linesearch=linesearch,
                          min_stepsize=0.1,
                          max_stepsize=2,
                          )
    solver_C = LBFGS(fun=loss_complex, **solver_options)
    solver_R = LBFGS(fun=loss_real, **solver_options)
    sol_C, state_C = solver_C.run(z0)
    sol_R, state_R = solver_R.run(C2R(z0))

    self.assertArraysAllClose(sol_C, R2C(sol_R), atol=tol, rtol=tol)
    self.assertArraysAllClose(state_C.s_history, R2C(state_R.s_history), atol=tol, rtol=tol)

  def test_handling_pytrees(self):
    def fun_(x):
      return sum(100.0 * (x[..., 1:] - x[..., :-1] ** 2.0) ** 2.0 + (1 - x[..., :-1]) ** 2.0)

    def fun(x):
      return tree_sum(tree_map(fun_, x))

    key = random.PRNGKey(0)
    x_arr0 = random.normal(key, (2, 4))
    x_tree0 = (x_arr0[0], x_arr0[1])

    lbfgs = LBFGS(fun=fun, maxiter=5)
    x_arr, _ = lbfgs.run(x_arr0)
    x_tree, _ = lbfgs.run(x_tree0)
    x_tree = jnp.stack((x_tree[0], x_tree[1]))
    self.assertArraysAllClose(x_arr, x_tree)

  def test_zero_history(self):
    # Should match a gradient descent if history_size = 0
    def fun(x):
      return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    
    x0 = jnp.zeros(2)
    lbfgs = LBFGS(fun=fun, tol=1e-3, maxiter=20, history_size=0, stepsize=0.01)
    x_lbfgs = lbfgs.run(x0).params
    gd = GradientDescent(fun=fun, tol=1e-3, maxiter=20, stepsize=0.01, acceleration=False)
    x_gd = gd.run(x0).params
    self.assertArraysAllClose(x_lbfgs, x_gd)

  @parameterized.product(
    fun_init_and_opt=[
      ('rosenbrock', onp.zeros(2), 0.),
      ('himmelblau', onp.ones(2), 0.),
      ('matyas', onp.ones(2) * 6., 0.),
      ('eggholder', onp.ones(2) * 100., None),  
      ('zakharov', onp.array([600.0, 700.0, 200.0, 100.0, 90.0, 1e4]), 0.),
    ],
  )
  def test_against_scipy(self, fun_init_and_opt):
    # Taken from previous jax tests
    # NOTE(vroulet): Unclear whether lbfgs or bfgs can find true minimum for
    # eggholder, but they seem to converge to the same solution with current 
    # implementation and initialization, which is a good check.
    tol = 1e-15 if jax.config.jax_enable_x64 else 1e-7
    fun_name, x0, opt = fun_init_and_opt
    jnp_fun, onp_fun = get_fun(fun_name, jnp), get_fun(fun_name, onp)
    jaxopt_res = LBFGS(jnp_fun, tol=tol).run(x0).params
    scipy_res = scipy_opt.minimize(onp_fun, x0, method='BFGS').x
    # scipy not good for matyas and zakharov functions, 
    # compare to true minimum, zero
    if fun_name in ['matyas', 'zakharov']:
      self.assertAllClose(jaxopt_res, jnp.zeros_like(jaxopt_res))
    elif fun_name == 'eggholder' and jax.config.jax_enable_x64:
      # FIXME(vroulet): at high precision, eggholder poses issues,
      # even in function values
      self.assertAllClose(jnp_fun(jaxopt_res), onp_fun(scipy_res), rtol=1e-14)
    else:
      self.assertAllClose(jaxopt_res, scipy_res, check_dtypes=False)
    if opt is not None:
      self.assertAllClose(jnp_fun(jaxopt_res), opt, check_dtypes=False)

  def test_minimize_bad_initial_values(self):
    # Taken from previous jax tests
    # This test runs deliberately "bad" initial values to test that handling
    # of failed line search, etc. is the same across implementations
    tol = 1e-6 if jax.config.jax_enable_x64 else 1e-3
    initial_value = onp.array([92, 0.001])
    jnp_fun, onp_fun = get_fun('himmelblau', jnp), get_fun('himmelblau', onp)
    # Here reuqires higher number of linesearch for jaxopt
    jaxopt_res = LBFGS(jnp_fun, tol=tol).run(initial_value).params
    scipy_res = scipy_opt.minimize(
        fun=onp_fun,
        jac=jax.grad(onp_fun),
        method='BFGS',
        x0=initial_value,
    ).x
    # Scipy and jaxopt converge to different minima so check in function values
    self.assertAllClose(onp_fun(scipy_res), jnp_fun(jaxopt_res))
    self.assertAllClose(jnp_fun(jaxopt_res), 0.)

  def test_steep(self):
    # Taken from previous jax tests
    # See jax related issue https://github.com/google/jax/issues/4594
    n = 2
    A = jnp.eye(n) * 1e4
    def fun(x):
      return jnp.mean((A @ x) ** 2)
    results = LBFGS(fun).run(init_params=jnp.ones(n)).params
    self.assertAllClose(results, jnp.zeros(n))

  def test_minimize_complex_sphere(self):
    # Taken from previous jax tests
    z0 = jnp.array([1., 2. - 3.j, 4., -5.j])

    def fun(z):
      return jnp.real(jnp.dot(jnp.conj(z - z0), z - z0))

    jaxopt_res = LBFGS(fun, tol=1e-6).run(jnp.zeros_like(z0)).params

    self.assertAllClose(jaxopt_res, z0)

  def test_complex_rosenbrock(self):
    # Taken from previous jax tests
    tol = 1e-12 if jax.config.jax_enable_x64 else 1e-6

    complex_dim = 5

    fun_real = get_fun('rosenbrock', jnp)
    init_re = jnp.zeros((2 * complex_dim,), dtype=complex)
    expect_re = jnp.ones((2 * complex_dim,), dtype=complex)

    def fun(z):
      x_re = jnp.concatenate([jnp.real(z), jnp.imag(z)])
      return fun_real(x_re)

    init = init_re[:complex_dim] + 1.j * init_re[complex_dim:]
    expect = expect_re[:complex_dim] + 1.j * expect_re[complex_dim:]

    jaxopt_res = LBFGS(fun, tol=tol).run(init).params
    self.assertAllClose(jaxopt_res, expect)

if __name__ == '__main__':
  absltest.main()
