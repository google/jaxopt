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

import jax.random
import jax.numpy as jnp

import numpy as onp

import jaxopt
from jaxopt import NonlinearCG
from jaxopt import objective
from jaxopt._src import test_util
from sklearn import datasets


def get_random_pytree():
    key = jax.random.PRNGKey(1213)

    def rn(key, l=3):
      return 0.05 * jnp.array(onp.random.normal(size=(10,)))

    def _get_random_pytree(curr_depth=0, max_depth=3):
        r = onp.random.uniform()
        if curr_depth == max_depth or r <= 0.2:  # leaf
            return rn(key)
        elif curr_depth <= 1 or r <= 0.7:  # list
            return [
                _get_random_pytree(curr_depth=curr_depth + 1, max_depth=max_depth)
                for _ in range(2)
            ]
        else:  # dict
            return {
                str(_): _get_random_pytree(
                    curr_depth=curr_depth + 1, max_depth=max_depth
                )
                for _ in range(2)
            }
    return [rn(key), {'a': rn(key), 'b': rn(key)}, _get_random_pytree()]


class NonlinearCGTest(test_util.JaxoptTestCase):

  def test_arbitrary_pytree(self):
    def loss(w, data):
      X, y = data
      _w = jnp.concatenate(jax.tree_util.tree_leaves(w))
      return ((jnp.dot(X, _w) - y) ** 2).mean()

    w = get_random_pytree()
    f_w = jnp.concatenate(jax.tree_util.tree_leaves(w))
    X, y = datasets.make_classification(n_samples=15, n_features=f_w.shape[-1],
                                        n_classes=2, n_informative=3,
                                        random_state=0)
    data = (X, y)
    cg_model = NonlinearCG(fun=loss, tol=1e-2, maxiter=300,
                           method="polak-ribiere")
    w_fit, info = cg_model.run(w, data=data)
    self.assertLessEqual(info.error, 5e-2)

  @parameterized.product(
      method=["fletcher-reeves", "polak-ribiere", "hestenes-stiefel"],
      linesearch=[
          "backtracking",
          "zoom",
          jaxopt.BacktrackingLineSearch(
              objective.binary_logreg, decrease_factor=0.5
          ),
      ],
      linesearch_init=["max", "current", "increase"],
  )
  def test_binary_logreg(self, method, linesearch, linesearch_init):
    X, y = datasets.make_classification(
        n_samples=10, n_features=5, n_classes=2, n_informative=3, random_state=0
    )
    data = (X, y)
    fun = objective.binary_logreg

    w_init = jnp.zeros(X.shape[1])
    cg_model = NonlinearCG(
        fun=fun,
        tol=1e-3,
        maxiter=100,
        method=method,
        linesearch=linesearch,
        linesearch_init=linesearch_init,
    )

    # Test with positional argument.
    w_fit, info = cg_model.run(w_init, data)

    # Check optimality conditions.
    self.assertLessEqual(info.error, 5e-2)

    # Compare against sklearn.
    w_skl = test_util.logreg_skl(X, y, 1e-6, fit_intercept=False,
                                 multiclass=False)
    self.assertArraysAllClose(w_fit, w_skl, atol=5e-2)


if __name__ == '__main__':
  absltest.main()
