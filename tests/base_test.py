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
from jax import test_util as jtu
import jax.numpy as jnp

from jaxopt import base

import numpy as onp

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional

import dataclasses

import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt.tree_util import tree_add
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_l2_norm
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_sub
from jaxopt.tree_util import tree_zeros_like


class DummySolverState(NamedTuple):
  iter_num: int
  error: float
  value: float
  aux: Any


@dataclasses.dataclass(eq=False)
class DummySolver(base.IterativeSolver):
  """Dummy solver."""

  fun: Callable
  maxiter: int = 500
  tol: float = 1e-3
  implicit_diff: bool = False

  def init_state(self, init_params: Any, *args, **kwargs) -> DummySolverState:
    return DummySolverState(iter_num=0, error=jnp.inf, value=jnp.inf, aux=None)

  def update(self,
             params: Any,
             state: DummySolverState,
             *args,
             **kwargs) -> base.OptStep:
    return base.OptStep(params=params, state=state)

  def dummy_method(self):
    return self

  def __post_init__(self):
    self.dummy_attr = True


class BaseTest(jtu.JaxTestCase):

  def test_linear_operator(self):
    rng = onp.random.RandomState(0)
    A = rng.randn(5, 3)
    x = rng.randn(3)
    y = rng.randn(5)
    I_x = jnp.eye(3)
    I_y = jnp.eye(5)
    delta_x = rng.randn(1)[0]
    delta_y = rng.randn(1)[0]
    X = rng.randn(3, 2)
    delta_X = rng.randn(2)
    Y = rng.randn(5, 2)
    delta_Y = rng.randn(5)
    linop = base.LinearOperator(A)

    # Check matrix-vector operations.
    Ax = jnp.dot(A, x)
    self.assertArraysAllClose(linop.matvec(x), Ax)
    ATy = jnp.dot(A.T, y)
    self.assertArraysAllClose(linop.rmatvec(y), ATy)

    for i in range(A.shape[0]):
      self.assertAllClose(linop.matvec_element(x, i), Ax[i])
      self.assertArraysAllClose(linop.update_rmatvec(ATy, delta_y, i),
                                jnp.dot(A.T, y + delta_y * I_y[i]))

    for j in range(A.shape[1]):
      self.assertAllClose(linop.rmatvec_element(y, j), ATy[j])
      self.assertArraysAllClose(linop.update_matvec(Ax, delta_x, j),
                                jnp.dot(A, x + delta_x * I_x[j]))

    # Check matrix-matrix operations.
    def E(i, shape):
      ret = onp.zeros(shape)
      ret[i] = 1
      return ret

    AX = jnp.dot(A, X)
    self.assertArraysAllClose(linop.matvec(X), AX)
    ATY = jnp.dot(A.T, Y)

    self.assertArraysAllClose(linop.rmatvec(Y), ATY)
    for i in range(A.shape[0]):
      self.assertAllClose(linop.matvec_element(X, i), AX[i])
      # todo: implement this
      # self.assertArraysAllClose(linop.update_rmatvec(ATY, delta_Y, i),
      #                    jnp.dot(A.T, Y + delta_Y[:, None] * E(i, Y.shape)))

    for j in range(A.shape[1]):
      self.assertAllClose(linop.rmatvec_element(Y, j), ATY[j])
      self.assertArraysAllClose(linop.update_matvec(AX, delta_X, j),
                                jnp.dot(A, X + delta_X * E(j, X.shape)))

    # Check that flatten and unflatten work.
    leaf_values, treedef = jax.tree_util.tree_flatten(linop)
    linop2 = jax.tree_util.tree_unflatten(treedef, leaf_values)
    self.assertArraysAllClose(linop2.matvec(x), Ax)

  def test_solver_attributes(self):
    fun = lambda x: x
    solver = DummySolver(fun=fun, maxiter=10, tol=1.0, implicit_diff=True)
    self.assertEqual(solver.attribute_names(),
                     ("fun", "maxiter", "tol", "implicit_diff"))
    self.assertEqual(solver.attribute_values(), (fun, 10, 1.0, True))

  def test_solver_hash(self):
    fun = lambda x: x
    solver = DummySolver(fun=fun, maxiter=10, tol=1.0, implicit_diff=True)
    hash(solver)

  def test_solver_equality(self):
    fun = lambda x: x
    solver = DummySolver(fun=fun, maxiter=10, tol=1.0, implicit_diff=True)
    self.assertTrue(solver == solver)

  def test_jit_update(self):
    fun = lambda x: x
    solver = DummySolver(fun=fun, maxiter=10, tol=1.0, implicit_diff=True)
    update = jax.jit(solver.update)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
