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
from jax import test_util as jtu
import jax.numpy as jnp

from jaxopt import objective
from jaxopt._src import test_util
from jaxopt import BacktrackingLineSearch

import numpy as onp

from sklearn import datasets


class BacktrackingLinesearchTest(jtu.JaxTestCase):

  @parameterized.product(cond=["strong-wolfe", "wolfe"])
  def test_backtracking_linesearch(self, cond):
    X, y = datasets.make_classification(n_samples=10, n_features=5, n_classes=2,
                                        n_informative=3, random_state=0)
    data = (X, y)
    fun = objective.binary_logreg

    rng = onp.random.RandomState(0)
    w_init = rng.randn(X.shape[1])
    grad = jax.grad(fun)(w_init, data=data)


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

    # Call to run with value_and_grad=True.
    ls = BacktrackingLineSearch(fun=jax.value_and_grad(fun),
                                            condition=cond,
                                            maxiter=20,
                                            value_and_grad=True)
    stepsize, state = ls.run(init_stepsize=1.0, params=w_init, data=data)
    self.assertLessEqual(state.error, 1e-5)


if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  jax.config.update("jax_enable_x64", True)
  absltest.main(testLoader=jtu.JaxTestLoader())
