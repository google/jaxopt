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
from jax.nn import softplus
import jax.numpy as jnp
from jaxopt import loss
import optax

from sklearn import datasets

try:
  from jaxopt.examples.implicit_diff import sparse_coding
except ImportError:
  import sparse_coding


class SparseCodingTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.N = 74
    self.k = 7
    self.d = 13
    # X is N x d
    # dic is k x d
    self.X, self.dictionary_0, self.codes_0 = datasets.make_sparse_coded_signal(
        n_samples=self.N,
        n_components=self.k,
        n_features=self.d,
        n_nonzero_coefs=self.k//2,
        random_state=0,
    )
    self.X = self.X.T # bug in https://github.com/scikit-learn/scikit-learn/issues/19894
    self.X = .1 * self.X + .0001 * jax.random.normal(
        jax.random.PRNGKey(0), (self.N, self.d))

  @parameterized.parameters([
      None,
      lambda x, y: jnp.sum(jnp.abs(x - y)**2.1),
      lambda x, y: jnp.sum(loss.huber_loss(x, y, .01))
  ])
  def test_task_driven_sparse_coding(self, reconstruction_loss_fun):
    elastic_penalty = 0.01
    regularization = 0.01

    # slightly complicated Vanilla dictionary learning when no task.
    # complicated in the sense that Danskin is not used. Here using prox from
    # jaxopt.
    solver = jax.jit(
        sparse_coding.make_task_driven_dictionary_learner(
            reconstruction_loss_fun=reconstruction_loss_fun, maxiter=10,
            sparse_coding_kw={'maxiter': 100}),
        static_argnums=(1, 8))  # n_components & reconstruction_loss_fun

    # Compute dictionary
    dic_jop_0 = solver(
        self.X,
        n_components=self.k,
        regularization=regularization,
        elastic_penalty=elastic_penalty)
    self.assertEqual(dic_jop_0.shape, (self.k, self.d))

    # Test now task driven dictionary learning using *arbitrary* labels computed
    # from initial codes. This is a binary logistic regression problem.
    label = jnp.sum(self.codes_0[0:3, :], axis=0) > 0
    def task_loss_fun(codes, dic, task_vars, task_params):
      del dic
      w, b = task_vars
      logit = jnp.dot(codes, w) + b
      return jnp.sum(
          jnp.sum(softplus(logit) - label * logit) + 0.5 * task_params *
          (jnp.dot(w, w) + b * b))

    # Create a solver that will now use optax's Adam to learn both dic and
    # logistic regression parameters.
    solver = jax.jit(
        sparse_coding.make_task_driven_dictionary_learner(
            task_loss_fun=task_loss_fun,
            reconstruction_loss_fun=reconstruction_loss_fun, maxiter=10,
            sparse_coding_kw={'maxiter': 100},
            optimizer=optax.adam(1e-3)),
        static_argnums=(1, 8))  # n_components & reconstruction_loss_fun

    dic_jop_task, w_and_b = solver(
        self.X,
        n_components=self.k,
        regularization=regularization,
        elastic_penalty=elastic_penalty,
        task_vars_init=(jnp.zeros(self.k), jnp.zeros(1)),
        task_params=0.001)

    # Check we have at least improved results using the very same w_and_b
    losses = []
    for dic in [dic_jop_0, dic_jop_task]:
      losses.append(
          task_loss_fun(
              sparse_coding.sparse_coding(
                  dic, (self.X, regularization, elastic_penalty)), dic, w_and_b,
              0.0))
    self.assertGreater(losses[0], losses[1])

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
