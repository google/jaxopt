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

from jaxopt import objective
from jaxopt._src import test_util
from jaxopt import BacktrackingLineSearch
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_vdot

import numpy as onp

from sklearn import datasets


class BacktrackingLinesearchTest(test_util.JaxoptTestCase):

  def _check_conditions_satisfied(
      self,
      condition_name,
      c1,
      c2,
      stepsize,
      initial_value,
      initial_grad,
      final_state):
    self.assertTrue(jnp.all(final_state.done))
    self.assertFalse(jnp.any(final_state.failed))

    descent_direction = tree_scalar_mul(-1, initial_grad)
    # Check sufficient decrease for all line search methods.
    sufficient_decrease = jnp.all(
        final_state.value <= initial_value +
        c1 * stepsize * tree_vdot(final_state.grad, descent_direction))
    self.assertTrue(sufficient_decrease)

    if condition_name == "goldstein":
      goldstein = jnp.all(
          final_state.value >= initial_value +
          (1 - c1) * stepsize * tree_vdot(
              initial_grad, descent_direction))
      self.assertTrue(goldstein)
    elif condition_name == "strong-wolfe":
      new_gd_vdot = tree_vdot(final_state.grad, descent_direction)
      gd_vdot = tree_vdot(initial_grad, descent_direction)
      curvature = jnp.all(jnp.abs(new_gd_vdot) <= c2 * jnp.abs(gd_vdot))
      self.assertTrue(curvature)

    elif condition_name == "wolfe":
      new_gd_vdot = tree_vdot(final_state.grad, descent_direction)
      gd_vdot = tree_vdot(initial_grad, descent_direction)
      curvature = jnp.all(new_gd_vdot >= c2 * gd_vdot)
      self.assertTrue(curvature)

  def test_aux_value(self):
    X, y = datasets.make_classification(n_samples=10, n_features=5, n_classes=2,
                                        n_informative=3, random_state=0)
    data = (X, y)
    true_fun = objective.binary_logreg

    def augmented_fun(w, data):
      value = true_fun(w, data)
      aux = w[0:1]
      return value, aux

    rng = onp.random.RandomState(0)
    w_init = rng.randn(X.shape[1])
    ls = BacktrackingLineSearch(fun=augmented_fun, maxiter=20, condition='armijo', has_aux=True)
    stepsize, state = ls.run(init_stepsize=1.0, params=w_init, data=data)
    _, aux_at_params = augmented_fun(state.params, data)
    self.assertArraysEqual(aux_at_params, state.aux)

  @parameterized.product(cond=[
      "armijo", "goldstein", "strong-wolfe", "wolfe"])
  def test_backtracking_linesearch(self, cond):
    X, y = datasets.make_classification(n_samples=10, n_features=5, n_classes=2,
                                        n_informative=3, random_state=0)
    data = (X, y)
    fun = objective.binary_logreg

    rng = onp.random.RandomState(0)
    w_init = rng.randn(X.shape[1])
    initial_grad = jax.grad(fun)(w_init, data=data)
    initial_value = fun(w_init, data=data)

    # Manual loop.
    ls = BacktrackingLineSearch(fun=fun, condition=cond)
    stepsize = 1.0
    state = ls.init_state(init_stepsize=1.0, params=w_init, data=data)
    stepsize, state = ls.update(stepsize=stepsize, state=state, params=w_init,
                                data=data)
    error1 = state.error
    stepsize, state = ls.update(stepsize=stepsize, state=state, params=w_init,
                                data=data)
    error2 = state.error
    self.assertLessEqual(error2, error1)

    # Call to run.
    ls = BacktrackingLineSearch(fun=fun, maxiter=20, condition=cond)
    stepsize, state = ls.run(init_stepsize=1.0, params=w_init, data=data)
    self.assertLessEqual(state.error, 1e-5)
    self._check_conditions_satisfied(
        cond, ls.c1, ls.c2, stepsize, initial_value, initial_grad, state)

    # Call to run with value_and_grad=True.
    ls = BacktrackingLineSearch(fun=jax.value_and_grad(fun),
                                condition=cond,
                                maxiter=20,
                                value_and_grad=True)
    stepsize, state = ls.run(init_stepsize=1.0, params=w_init, data=data)
    self.assertLessEqual(state.error, 1e-5)
    self._check_conditions_satisfied(
        cond, ls.c1, ls.c2, stepsize, initial_value, initial_grad, state)

    # Failed linesearch (high c1 ensures convergence condition is not met).
    ls = BacktrackingLineSearch(fun=fun, maxiter=20, condition=cond, c1=2.)
    _, state = ls.run(init_stepsize=1.0, params=w_init, data=data)
    self.assertTrue(jnp.all(state.failed))
    self.assertFalse(jnp.any(state.done))


if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  #jax.config.update("jax_enable_x64", True)
  absltest.main()
