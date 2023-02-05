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

from jaxopt import HagerZhangLineSearch
from jaxopt import objective
from jaxopt._src import test_util
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_vdot

import numpy as onp

from sklearn import datasets


class HagerZhangLinesearchTest(test_util.JaxoptTestCase):

  def _check_conditions_satisfied(
      self,
      c1,
      c2,
      stepsize,
      initial_value,
      initial_grad,
      final_state):
    self.assertTrue(jnp.all(final_state.done))

    descent_direction = tree_scalar_mul(-1, initial_grad)
    sufficient_decrease = jnp.all(
        final_state.value <= initial_value +
        c1 * stepsize * tree_vdot(final_state.grad, descent_direction))
    self.assertTrue(sufficient_decrease)

    new_gd_vdot = tree_vdot(final_state.grad, descent_direction)
    gd_vdot = tree_vdot(initial_grad, descent_direction)
    curvature = jnp.all(new_gd_vdot >= c2 * gd_vdot)
    self.assertTrue(curvature)

  def test_hager_zhang_linesearch(self):
    x, y = datasets.make_classification(
        n_samples=10, n_features=5, n_classes=2,
        n_informative=3, random_state=0)
    data = (x, y)
    fun = objective.binary_logreg

    rng = onp.random.RandomState(0)
    w_init = rng.randn(x.shape[1])
    initial_grad = jax.grad(fun)(w_init, data=data)
    initial_value = fun(w_init, data=data)

    # Manual loop.
    ls = HagerZhangLineSearch(fun=fun)
    stepsize = 1.0
    state = ls.init_state(init_stepsize=1., params=w_init, data=data)
    stepsize, state = ls.update(stepsize=stepsize, state=state, params=w_init,
                                data=data)

    # Call to run.
    ls = HagerZhangLineSearch(fun=fun, maxiter=20)
    stepsize, state = ls.run(init_stepsize=1., params=w_init, data=data)
    self._check_conditions_satisfied(
        ls.c1, ls.c2, stepsize, initial_value, initial_grad, state)

    # Call to run with value_and_grad=True.
    ls = HagerZhangLineSearch(fun=jax.value_and_grad(fun),
                              maxiter=20,
                              value_and_grad=True)
    stepsize, state = ls.run(init_stepsize=1., params=w_init, data=data)
    self._check_conditions_satisfied(
        ls.c1, ls.c2, stepsize, initial_value, initial_grad, state)


if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  # jax.config.update("jax_enable_x64", True)
  absltest.main()
