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

import jax
import jax.numpy as jnp

from jaxopt._src import linear_solve as _linear_solve
from jaxopt import linear_solve
from jaxopt._src import test_util

import numpy as onp

class LinearSolveTest(test_util.JaxoptTestCase):

  def test_materialize_array(self):
    rng = onp.random.RandomState(0)

    # Matrix case.
    A = rng.randn(5, 5)
    matvec = lambda x: jnp.dot(A, x)
    A2 = _linear_solve._materialize_array(matvec, (5,))
    self.assertArraysAllClose(A, A2)

    # Tensor case.
    A = rng.randn(5, 3, 5, 3)
    A_mat = A.reshape(15, 15)
    matvec = lambda x: jnp.dot(A_mat, x.ravel()).reshape(5, 3)
    A2 = _linear_solve._materialize_array(matvec, (5, 3))
    self.assertArraysAllClose(A, A2, atol=1e-3)

  def test_rmatvec(self):
    rng = onp.random.RandomState(0)
    A = rng.randn(5, 5)
    matvec = lambda x: jnp.dot(A, x)
    x = rng.randn(5)
    self.assertArraysAllClose(_linear_solve._rmatvec(matvec, x),
                              jnp.dot(A.T, x))

  def test_normal_matvec(self):
    rng = onp.random.RandomState(0)
    A = rng.randn(5, 5)
    matvec = lambda x: jnp.dot(A, x)
    x = rng.randn(5)
    self.assertArraysAllClose(_linear_solve._normal_matvec(matvec, x),
                              jnp.dot(A.T, jnp.dot(A, x)))

  def test_solve_dense(self):
    rng = onp.random.RandomState(0)

    # Matrix case.
    A = rng.randn(5, 5)
    b = rng.randn(5)
    matvec = lambda x: jnp.dot(A, x)
    x = linear_solve.solve_lu(matvec, b)
    x2 = jax.numpy.linalg.solve(A, b)
    x3 = linear_solve.solve_iterative_refinement(matvec, b)
    self.assertArraysAllClose(x, x2)
    self.assertArraysAllClose(x, x3)

    # Tensor case.
    A = rng.randn(5, 3, 5, 3)
    A_mat = A.reshape(15, 15)
    b = rng.randn(5, 3)

    def matvec(x):
      return jnp.dot(A_mat, x.ravel()).reshape(5, 3)

    x = linear_solve.solve_lu(matvec, b)
    x2 = linear_solve.solve_gmres(matvec, b)
    x3 = linear_solve.solve_iterative_refinement(matvec, b)
    self.assertArraysAllClose(x, x2, atol=1e-4)
    self.assertArraysAllClose(x, x3, atol=1e-4)

  def test_solve_dense_psd(self):
    rng = onp.random.RandomState(0)
    A = rng.randn(5, 5)
    A = jnp.dot(A.T, A)
    b = rng.randn(5)
    matvec = lambda x: jnp.dot(A, x)
    x = linear_solve.solve_cholesky(matvec, b)
    x2 = jax.numpy.linalg.solve(A, b)
    self.assertArraysAllClose(x, x2, atol=1e-2)

  def test_solve_sparse(self):
    rng = onp.random.RandomState(0)

    # Matrix case.
    A = rng.randn(5, 5)
    b = rng.randn(5)

    def matvec(x):
      return jnp.dot(A, x)

    x = linear_solve.solve_lu(matvec, b)
    x2 = linear_solve.solve_normal_cg(matvec, b)
    x3 = linear_solve.solve_gmres(matvec, b)
    x4 = linear_solve.solve_bicgstab(matvec, b)
    x5 = linear_solve.solve_iterative_refinement(matvec, b)
    self.assertArraysAllClose(x, x2, atol=1e-4)
    self.assertArraysAllClose(x, x3, atol=1e-4)
    self.assertArraysAllClose(x, x4, atol=1e-4)
    self.assertArraysAllClose(x, x5, atol=1e-4)

  def test_solve_sparse_ridge(self):
    rng = onp.random.RandomState(0)

    # Value of the ridge regularizaer.
    ridge = 1.0

    # Matrix case.
    A = rng.randn(5, 5)
    b = rng.randn(5)

    def matvec(x):
      return jnp.dot(A, x)

    def matvec_with_ridge(x):
      return jnp.dot(A + ridge * onp.eye(A.shape[0]), x)

    x = linear_solve.solve_lu(matvec_with_ridge, b)
    x2 = linear_solve.solve_gmres(matvec, b, ridge=ridge)
    x3 = linear_solve.solve_bicgstab(matvec, b, ridge=ridge)
    self.assertArraysAllClose(x, x2, atol=1e-4)
    self.assertArraysAllClose(x, x3, atol=1e-4)

    K = onp.dot(A.T, A)
    Ab = onp.dot(A.T, b)

    def gram_matvec(x):
      return jnp.dot(K, x)

    def gram_matvec_with_ridge(x):
      return jnp.dot(K + ridge * onp.eye(A.shape[0]), x)

    x = linear_solve.solve_lu(gram_matvec_with_ridge, b)
    x4 = linear_solve.solve_cg(gram_matvec, b, ridge=ridge)
    self.assertArraysAllClose(x, x4, atol=1e-4)

    x = linear_solve.solve_lu(gram_matvec_with_ridge, Ab)
    x5 = linear_solve.solve_normal_cg(matvec, b, ridge=ridge)
    self.assertArraysAllClose(x, x5, atol=1e-4)

  def test_solve_1d(self):
    rng = onp.random.RandomState(0)

    # Matrix case.
    A = rng.randn(1, 1)
    b = rng.randn(1)

    def matvec(x):
      return jnp.dot(A, x)

    x = linear_solve.solve_lu(matvec, b)
    self.assertArraysAllClose(x, b / A[0, 0])


if __name__ == '__main__':
  absltest.main()
