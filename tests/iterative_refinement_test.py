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

from functools import partial

from absl.testing import absltest

import jax
import jax.numpy as jnp
from jax.test_util import check_grads

from jaxopt import linear_solve
from jaxopt import IterativeRefinement
from jaxopt._src import test_util

import numpy as onp


class IterativeRefinementTest(test_util.JaxoptTestCase):

  def test_simple_system(self):
    onp.random.seed(0)
    n = 20
    A = onp.random.rand(n, n)
    b = onp.random.randn(n)

    low_acc = 1e-1
    high_acc = 1e-5

    # Heavily regularized low acuracy solver.
    inner_solver = partial(linear_solve.solve_gmres, tol=low_acc, ridge=1e-3)

    solver = IterativeRefinement(solve=inner_solver, tol=high_acc, maxiter=10)
    x, state = solver.run(None, A, b)
    self.assertLess(state.error, high_acc)

    x_approx = inner_solver(lambda x: jnp.dot(A, x), b)
    error_inner_solver = solver.l2_optimality_error(x_approx, A, b)
    # High accuracy solution obtained from low accuracy solver.
    self.assertLess(state.error, error_inner_solver)

  def test_ill_posed_problem(self):
    onp.random.seed(0)
    n = 10
    e = 5

    # duplicated rows.
    A = onp.random.rand(e, n)
    A = jnp.concatenate([A, A], axis=0)
    b = onp.random.randn(e)
    b = jnp.concatenate([b, b], axis=0)

    low_acc = 1e-1
    high_acc = 1e-3

    # Heavily regularized low acuracy solver.
    inner_solver = partial(linear_solve.solve_gmres, tol=low_acc, ridge=5e-2)

    solver = IterativeRefinement(solve=inner_solver, tol=high_acc, maxiter=30)
    x, state = solver.run(init_params=None, A=A, b=b)
    self.assertLess(state.error, high_acc)

    x_approx = inner_solver(lambda x: jnp.dot(A, x), b)
    error_inner_solver = solver.l2_optimality_error(x_approx, A, b)
    # High accuracy solution obtained from low accuracy solver.
    self.assertLess(state.error, error_inner_solver)

  def test_perturbed_system(self):
    onp.random.seed(0)
    n = 20

    A = onp.random.rand(n, n)  # invertible matrix (with high probability).

    noise = onp.random.randn(n, n)
    sigma = 0.05
    A_bar = A + sigma * noise  # perturbed system.

    expected = onp.random.randn(n)
    b = A @ expected  # unperturbed target.

    high_acc = 1e-3
    solver = IterativeRefinement(matvec_A=None, matvec_A_bar=jnp.dot,
                                 tol=high_acc, maxiter=100)
    x, state = solver.run(init_params=None, A=A, b=b, A_bar=A_bar)
    self.assertLess(state.error, high_acc)
    self.assertArraysAllClose(x, expected, rtol=5e-2)

  def test_implicit_diff(self):
    onp.random.seed(17)
    n = 20
    A = onp.random.rand(n, n)
    b = onp.random.randn(n)

    low_acc = 1e-1
    high_acc = 1e-5

    # Heavily regularized low acuracy solver.
    inner_solver = partial(linear_solve.solve_gmres, tol=low_acc, ridge=1e-3)
    solver = IterativeRefinement(solve=inner_solver, tol=high_acc, maxiter=10)

    def solve_run(A, b):
      x, state = solver.run(init_params=None, A=A, b=b)
      return x

    check_grads(solve_run, args=(A, b), order=1, modes=['rev'], eps=1e-3)

  def test_warm_start(self):
    onp.random.seed(0)
    n = 20
    A = onp.random.rand(n, n)
    b = onp.random.randn(n)

    init_x = onp.random.randn(n)

    high_acc = 1e-5

    solver = IterativeRefinement(tol=high_acc, maxiter=10)
    x, state = solver.run(init_x, A, b)
    self.assertLess(state.error, high_acc)


if __name__ == "__main__":
  jax.config.update("jax_enable_x64", False)  # low precision environment.
  absltest.main()
