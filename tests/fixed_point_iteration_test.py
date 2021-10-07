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
from jax import test_util as jtu
from jax.tree_util import tree_map, tree_all
from jax.test_util import check_grads

from jaxopt.tree_util import tree_l2_norm, tree_scalar_mul, tree_sub
from jaxopt import objective
from jaxopt import FixedPointIteration

import numpy as onp
from sklearn import datasets


class FixedPointIterationTest(jtu.JaxTestCase):

  def test_geometric_decay(self):
    """Test convergence for geometric progression with common ratio r < 1."""
    def f(x):
      return jnp.array([0.9, 0.99, 0.999]) * x
    x0 = jnp.array([1., 0.4, -0.2])
    tol = 1e-5
    sol, state = FixedPointIteration(f, maxiter=8000, tol=tol).run(x0)
    self.assertLess(state.error, tol)
    sol_norm = tree_l2_norm(sol)
    big_tol = 1e-2  # we are forced to take high tol since convergence is very slow for 0.999
    self.assertLess(sol_norm, big_tol)

  @parameterized.product(jit=[False,True])
  def test_sin_fixed_point(self, jit):
    """Test convergence for simple polynomials and sin.

    Also test the support of pytree in input/output.
    """
    def f(x):  # Another fixed point exists for x[0] : ~1.11
      return jnp.sin(x[0]) * (x[0] ** 2), x[1] ** 3
    x0 = jnp.array([0.6, 0., -0.1]), jnp.array([[0.7], [0.5]])
    tol = 1e-6
    sol, state = FixedPointIteration(f, maxiter=100, tol=tol, jit=jit).run(x0)
    self.assertLess(state.error, tol)
    sol_norm = tree_l2_norm(sol)
    self.assertLess(sol_norm, tol)
    f_fixed_point = jnp.zeros(3), jnp.zeros((2,1))
    self.assertAllClose(sol, f_fixed_point)

  def test_cos_fixed_point(self):
    """Test convergence for cos fixed point (non-zero fixed point).

    Also test support for additional parameters.
    """
    def f(x, theta):
      return jnp.cos(x) + theta
    x0 = jnp.array([0.3, jnp.pi / 4])
    theta = jnp.array([0., jnp.pi / 2])
    f_fixed_point = jnp.array([0.73908614842288, jnp.pi / 2])
    tol = 1e-2
    sol, state = FixedPointIteration(f, maxiter=100000, tol=tol).run(x0, theta)
    self.assertLess(state.error, tol)
    self.assertArraysAllClose(sol, f_fixed_point, atol=1e-2)

  def test_has_aux(self):
    """Test support of ``has_aux`` functionnality."""
    def f(x, r, y):
      fx = jnp.cos(x)  # fixed point 0.739085
      fy = r * y * (1 - y)  # logistic map: chaotic behavior
      return fx, fy
    r = jnp.array([3.95])  # for this value there is chaotic behavior
    x0, y0 = jnp.array([1.]), jnp.array([0.6])
    tol = 2e-2
    fp = FixedPointIteration(f, maxiter=400*1000, tol=tol, has_aux=True)
    sol, state = fp.init(x0, r, y0)
    y = y0
    sols = []
    for i in range(10):
      sol, state = fp.update(sol, state, r, y)
      sols.append(sol)
      y = state.aux
    self.assertLess(state.error, tol)
    self.assertArraysAllClose(sol, jnp.array([0.739085]), atol=1e2)

  @parameterized.product(implicit_diff=[False, True])
  def test_simple_grads(self, implicit_diff):
    """Test correctness of gradients on a simple function."""
    def f(x, theta):
      return jnp.cos(x) + theta
    x0 = jnp.array([0., jnp.pi / 4])
    theta = jnp.array([0., jnp.pi / 2])
    fp = FixedPointIteration(f, maxiter=4*1000, tol=1e-6, implicit_diff=implicit_diff)
    def solve_run(args, kwargs):
      return fp.run(x0, *args, **kwargs)[0]
    check_grads(solve_run, args=([theta], {}), order=1, modes=['rev'], eps=None)

  def test_grads_flat_landscape(self):
    """Test correctness of gradients."""

    def f(x, theta):
      return theta * x, x

    x0 = jnp.array([1.,-0.5])
    theta = jnp.array([0.05, 0.2])
    fp = FixedPointIteration(f, maxiter=1000, tol=1e-6, has_aux=True)

    def solve_run(theta):
      return fp.run(x0, theta).params[0]

    check_grads(solve_run, args=(theta,), order=1, modes=['rev'], eps=1e-4)

  def test_affine_contractive_mapping(self):
    """Test correctness on big affine contractive mapping."""
    n = 200
    key, subkey = jax.random.split(jax.random.PRNGKey(0))
    M = jax.random.uniform(subkey, shape=(n,n), minval=-1., maxval=1.)
    M = M + M.T  # random symmetric matrix
    eigv = jnp.linalg.eigvalsh(M)
    spectral_radius = jnp.abs(eigv[-1])
    eps = 1e-4
    M = M / (2*spectral_radius + eps)  # contractive mapping: lbda_max < 0.5
    key, subkey = jax.random.split(key)
    b = jax.random.uniform(subkey, shape=(n,))
    def f(x, M, b):
      return M @ x + b
    tol = 1e-6
    fp = FixedPointIteration(f, maxiter=100, tol=tol)
    x0 = jnp.zeros_like(b)
    sol, state = fp.run(x0, M, b)
    self.assertLess(state.error, tol)
    def solve_run(args, kwargs):
      return fp.run(x0, *args, **kwargs)[0]
    check_grads(solve_run, args=([M, b], {}), order=1, modes=['rev'], eps=1e-4)


if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  # jax.config.update("jax_enable_x64", True)
  absltest.main(testLoader=jtu.JaxTestLoader())

