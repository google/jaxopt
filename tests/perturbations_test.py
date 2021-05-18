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
"""Tests for jax_perturbations.perturbations."""

from absl.testing import absltest

import distrax
import jax
from jax import test_util as jtu
import jax.numpy as jnp

from jaxopt import perturbations


def one_hot_argmax(inputs: jnp.array) -> jnp.array:
  """An argmax one-hot function for arbitrary shapes."""
  inputs_flat = jnp.reshape(inputs, (-1))
  flat_one_hot = jax.nn.one_hot(jnp.argmax(inputs_flat), inputs_flat.shape[0])
  return jnp.reshape(flat_one_hot, inputs.shape)


def ranks(inputs: jnp.array) -> jnp.array:
  return jnp.argsort(jnp.argsort(inputs, axis=-1), axis=-1)


class PerturbationsArgmaxTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.num_samples = 1000
    self.sigma = 0.5
    self.rng = jax.random.PRNGKey(0)

  def test_argmax(self):
    """Checks that the perturbed argmax approximates well the softmax."""

    theta_matrix = jnp.array([[-0.5, 0.2, 0.1],
                              [-0.2, 0.1, 0.4]])
    single_argmax = one_hot_argmax(theta_matrix[0])
    test_single_argmax = jnp.array([0., 1., 0.])
    self.assertArraysEqual(single_argmax, test_single_argmax)
    matrix_argmax = one_hot_argmax(theta_matrix)
    test_matrix_argmax = jnp.array([[0., 0., 0.],
                                    [0., 0., 1.]])
    self.assertArraysEqual(matrix_argmax, test_matrix_argmax)
    batch_argmax = jax.vmap(one_hot_argmax)(theta_matrix)
    test_batch_argmax = jnp.array([[0., 1., 0.],
                                   [0., 0., 1.]])
    self.assertArraysEqual(batch_argmax, test_batch_argmax)

    square_norm_fun = lambda x: jnp.sum(x * x, axis=-1)

    theta_batch = jnp.array([[-0.6, 1.9, -0.2, 1.1, -1.0],
                             [-0.6, 1.0, -0.2, 1.8, -1.0]], dtype=jnp.float32)
    one_hot_argmax_batch = jax.jit(jax.vmap(one_hot_argmax))(theta_batch)
    one_hot_test = jnp.array([[0.0, 1.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 1.0, 0.0]], dtype=jnp.float32)
    self.assertArraysEqual(one_hot_argmax_batch, one_hot_test)

    pert_argmax_fun = jax.jit(perturbations.make_perturbed_argmax(
        argmax_fun=one_hot_argmax,
        num_samples=self.num_samples,
        sigma=self.sigma,
        noise=perturbations.Gumbel()))
    rngs_batch = jax.random.split(self.rng, 2)
    pert_argmax = jax.vmap(pert_argmax_fun)(theta_batch, rngs_batch)
    soft_argmax = jax.nn.softmax(theta_batch/self.sigma)
    self.assertArraysAllClose(pert_argmax, soft_argmax, atol=3e-2)

    theta_batch_repeat = jnp.array([[-0.6, 1.9, -0.2, 1.1, -1.0],
                                    [-0.6, 1.9, -0.2, 1.1, -1.0]],
                                   dtype=jnp.float32)
    pert_argmax_fun = jax.jit(perturbations.make_perturbed_argmax(
        argmax_fun=one_hot_argmax,
        num_samples=self.num_samples,
        sigma=self.sigma,
        noise=perturbations.Gumbel()))
    rngs_batch = jax.random.split(self.rng, 2)
    pert_argmax_repeat = jax.vmap(pert_argmax_fun)(theta_batch_repeat,
                                                   rngs_batch)
    self.assertArraysAllClose(pert_argmax_repeat[0], pert_argmax_repeat[1],
                              atol=2e-2)
    delta_noise = pert_argmax_repeat[0] - pert_argmax_repeat[1]
    self.assertNotAlmostEqual(jnp.linalg.norm(delta_noise), 0)

    def square_loss(theta_batch, rng):
      batch_size = theta_batch.shape[0]
      rngs_batch = jax.random.split(rng, batch_size)
      pert_argmax_batch = jax.vmap(pert_argmax_fun)(theta_batch, rngs_batch)
      return jnp.mean(square_norm_fun(pert_argmax_batch))

    grad_pert = jax.grad(square_loss)(theta_batch, self.rng)

    def square_loss_soft(theta):
      soft_max = jax.nn.softmax(theta/self.sigma)
      return jnp.mean(square_norm_fun(soft_max))

    grad_soft = jax.grad(square_loss_soft)(theta_batch)
    self.assertArraysAllClose(grad_pert, grad_soft, atol=1e-1)

  def test_distrax(self):
    """Checks that the function is compatible with distrax distributions."""

    theta_batch = jnp.array([[-0.5, 0.2, 0.1],
                             [-0.2, 0.1, 0.4]])
    pert_argmax_fun = jax.jit(perturbations.make_perturbed_argmax(
        argmax_fun=one_hot_argmax,
        num_samples=self.num_samples,
        sigma=self.sigma,
        noise=perturbations.Normal()))
    rngs_batch = jax.random.split(self.rng, 2)
    pert_argmax = jax.vmap(pert_argmax_fun)(theta_batch, rngs_batch)

    dist_argmax_fun = jax.jit(perturbations.make_perturbed_argmax(
        argmax_fun=one_hot_argmax,
        num_samples=self.num_samples,
        sigma=self.sigma,
        noise=distrax.Normal(loc=0., scale=1.)))
    dist_argmax = jax.vmap(dist_argmax_fun)(theta_batch, rngs_batch)

    self.assertArraysAllClose(pert_argmax, dist_argmax, atol=1e-6)

  def test_or(self):
    """Checks that the perturbed or returns the correct value."""
    # High value of num_samples for this specific test. Not required in normal
    # usecases, as in learning tasks.
    pert_sign = jax.jit(jax.vmap(perturbations.make_perturbed_argmax(
        argmax_fun=jnp.sign,
        num_samples=10000,
        sigma=self.sigma,
        noise=perturbations.Normal())))

    theta_batch = jnp.array([[-0.5, 1.2],
                             [-0.4, -1.2],
                             [-0.2, 0.2],
                             [0.1, 0.05]])
    rngs_batch = jax.random.split(self.rng, 4)
    soft_sign = pert_sign(theta_batch, rngs_batch)
    test_sign = 2 * jax.scipy.stats.norm.cdf(theta_batch/self.sigma) - 1
    self.assertArraysAllClose(soft_sign, test_sign, atol=1e-1)

    def pert_sum_or(x_batch, rng):
      batch_size = x_batch.shape[0]
      rngs_batch = jax.random.split(rng, batch_size)
      pert_sign_batch = pert_sign(x_batch, rngs_batch)
      return jnp.sum(jnp.max(pert_sign_batch, axis=-1))

    def explicit_sum_or(x):
      cdf_value = jax.scipy.stats.norm.cdf(x / self.sigma)
      return jnp.sum(jnp.max(2 * cdf_value - 1, axis=-1))

    grad_pert_sum = jax.grad(pert_sum_or)(theta_batch, self.rng)

    grad_test = jax.grad(explicit_sum_or)(theta_batch)

    self.assertArraysAllClose(1 + grad_pert_sum, 1 + grad_test, atol=1e-1)

  def test_rank_small_sigma(self):

    theta = jnp.array([-0.8, 0.6, 1.2, -1.0, -0.7, 0.6, 1.1, -1.0, 0.4])
    pert_ranks_fn_small_sigma = jax.jit(perturbations.make_perturbed_argmax(
        argmax_fun=ranks,
        num_samples=self.num_samples,
        sigma=1e-9,
        noise=perturbations.Normal()))
    pert_ranks_small_sigma = pert_ranks_fn_small_sigma(theta, self.rng)
    test_ranks_no_sigma = jnp.array(ranks(theta), dtype='float32')
    self.assertArraysAllClose(pert_ranks_small_sigma,
                              test_ranks_no_sigma, atol=1e-3)

  def test_rank_finite_diff(self):
    theta = jnp.array([-0.8, 0.6, 1.2, -1.0, -0.7, 0.6, 1.1, -1.0, 0.4])
    # High value of num_samples for this specific test. Not required in normal
    # usecases, as in learning tasks.
    pert_ranks_fun = jax.jit(perturbations.make_perturbed_argmax(
        argmax_fun=ranks,
        num_samples=100000,
        sigma=self.sigma,
        noise=perturbations.Normal()))

    square_loss_rank = lambda x, rng: jnp.mean(pert_ranks_fun(x, rng) ** 2)

    gradient_square_rank = jax.grad(square_loss_rank)(theta, self.rng)
    eps = 1e-2
    h = jax.random.uniform(self.rng, shape=theta.shape)
    rngs = jax.random.split(self.rng, 2)
    sq_loss_plus_h = square_loss_rank(theta + eps * h, rngs[0])
    sq_loss_minus_h = square_loss_rank(theta - eps * h, rngs[1])
    delta_num = (sq_loss_plus_h - sq_loss_minus_h) / (2 * eps)
    delta_lin = jnp.sum(gradient_square_rank * h)

    self.assertArraysAllClose(delta_num, delta_lin, atol=5e-2)


class PerturbationsMaxTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.num_samples = 1000
    self.sigma = 0.5
    self.rng = jax.random.PRNGKey(0)
    self.theta_batch = jnp.array([[-0.6, 1.9, -0.2, 1.7, -1.0],
                                  [-0.6, 1.0, -0.2, 1.8, -1.0]],
                                 dtype=jnp.float32)

  def test_max_small_sigma(self):
    """Checks that the perturbed argmax is close to the max for small sigma."""
    pert_max_small_sigma_fun = jax.jit(perturbations.make_perturbed_max(
        argmax_fun=one_hot_argmax,
        num_samples=self.num_samples,
        sigma=1e-7,
        noise=perturbations.Gumbel()))
    rngs_batch = jax.random.split(self.rng, 2)
    pert_max_small_sigma = jax.vmap(pert_max_small_sigma_fun)(self.theta_batch,
                                                              rngs_batch)

    self.assertArraysAllClose(pert_max_small_sigma,
                              jnp.array([1.9, 1.8]), atol=1e-6)

  def test_max_jensen(self):
    """Checks that the noise increases the value of the expected max."""
    rngs_batch = jax.random.split(self.rng, 2)
    pert_max_fun = jax.jit(perturbations.make_perturbed_max(
        argmax_fun=one_hot_argmax,
        num_samples=10000,
        sigma=self.sigma,
        noise=perturbations.Gumbel()))
    pert_max = jax.vmap(pert_max_fun)(self.theta_batch, rngs_batch)
    self.assertGreater(pert_max[0], 1.9)
    self.assertGreater(pert_max[1], 1.8)

  def test_grads(self):
    """Tests that the perturbed and regularized maxes match."""
    pert_max_fun = jax.jit(perturbations.make_perturbed_max(
        argmax_fun=one_hot_argmax,
        num_samples=10000,
        sigma=self.sigma,
        noise=perturbations.Gumbel()))

    def sum_max(theta_batch, rng):
      batch_size = theta_batch.shape[0]
      rngs_batch = jax.random.split(rng, batch_size)
      pert_max_batch = jax.vmap(pert_max_fun)(theta_batch, rngs_batch)
      return jnp.sum(pert_max_batch)

    soft_argmax = jax.nn.softmax(self.theta_batch/self.sigma)

    grad_pert = jax.grad(sum_max)(self.theta_batch, self.rng)

    self.assertArraysAllClose(soft_argmax, grad_pert, atol=1e-2)

  def test_noise_iid(self):
    """Checks that different elements of the batch have different noises."""
    pert_max_fun = jax.jit(perturbations.make_perturbed_max(
        argmax_fun=one_hot_argmax,
        num_samples=10000,
        sigma=self.sigma,
        noise=perturbations.Gumbel()))
    theta_batch_repeat = jnp.array([[-0.6, 1.9, -0.2, 1.1, -1.0],
                                    [-0.6, 1.9, -0.2, 1.1, -1.0]],
                                   dtype=jnp.float32)
    rngs_batch = jax.random.split(self.rng, 2)
    pert_max_repeat = jax.vmap(pert_max_fun)(theta_batch_repeat, rngs_batch)
    self.assertArraysAllClose(pert_max_repeat[0], pert_max_repeat[1],
                              atol=1e-2)
    delta_noise = pert_max_repeat[0] - pert_max_repeat[1]
    self.assertNotAlmostEqual(jnp.linalg.norm(delta_noise), 0)

  def test_distrax(self):
    """Checks that the function is compatible with distrax distributions."""

    theta_batch = jnp.array([[-0.5, 0.2, 0.1],
                             [-0.2, 0.1, 0.4]])
    pert_max_fun = jax.jit(perturbations.make_perturbed_max(
        argmax_fun=one_hot_argmax,
        num_samples=self.num_samples,
        sigma=self.sigma,
        noise=perturbations.Normal()))
    rngs_batch = jax.random.split(self.rng, 2)
    pert_max = jax.vmap(pert_max_fun)(theta_batch, rngs_batch)

    dist_max_fun = jax.jit(perturbations.make_perturbed_max(
        argmax_fun=one_hot_argmax,
        num_samples=self.num_samples,
        sigma=self.sigma,
        noise=distrax.Normal(loc=0., scale=1.)))
    dist_max = jax.vmap(dist_max_fun)(theta_batch, rngs_batch)

    self.assertArraysAllClose(pert_max, dist_max, atol=1e-6)

  def test_rank_finite_diff(self):
    theta = jnp.array([-0.8, 0.6, 1.2, -1.0, -0.7, 0.6, 1.1, -1.0, 0.4])
    # High value of num_samples for this specific test. Not required in normal
    # usecases, as in learning tasks.
    pert_ranks_fun = jax.jit(perturbations.make_perturbed_max(
        argmax_fun=ranks,
        num_samples=100000,
        sigma=self.sigma,
        noise=perturbations.Normal()))

    gradient_ranks_value = jax.grad(pert_ranks_fun)(theta, self.rng)
    eps = 1e-2
    h = jax.random.uniform(self.rng, shape=theta.shape)
    rngs = jax.random.split(self.rng, 2)
    delta_num = (pert_ranks_fun(theta + eps * h, rngs[0]) -
                 pert_ranks_fun(theta - eps * h, rngs[0])) / (2 * eps)
    delta_lin = jnp.sum(gradient_ranks_value * h)

    self.assertArraysAllClose(delta_num, delta_lin, rtol=1e-2)

  def test_rank_autodiff(self):
    rnd_b_s = 10
    theta_random_batch = jax.random.uniform(self.rng, (rnd_b_s, 5))
    pert_ranks_max_fun = jax.jit(perturbations.make_perturbed_max(
        argmax_fun=ranks,
        num_samples=self.num_samples,
        sigma=self.sigma,
        noise=perturbations.Normal()))
    pert_ranks_argmax_fun = jax.jit(perturbations.make_perturbed_argmax(
        argmax_fun=ranks,
        num_samples=self.num_samples,
        sigma=self.sigma,
        noise=perturbations.Normal()))
    rngs = jax.random.split(self.rng, rnd_b_s)
    autodiff_grad = jax.vmap(jax.grad(pert_ranks_max_fun))(theta_random_batch,
                                                           rngs)
    expected_grad = jax.vmap(pert_ranks_argmax_fun)(theta_random_batch, rngs)
    self.assertArraysAllClose(autodiff_grad, expected_grad, rtol=1e-6)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
