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

"""Tests for jaxopt.perturbations."""

from absl.testing import absltest

import jax
import jax.numpy as jnp

from jaxopt import perturbations
from jaxopt._src import test_util
from jaxopt import loss


def one_hot_argmax(inputs: jnp.ndarray) -> jnp.ndarray:
  """An argmax one-hot function for arbitrary shapes."""
  inputs_flat = jnp.reshape(inputs, (-1))
  flat_one_hot = jax.nn.one_hot(jnp.argmax(inputs_flat), inputs_flat.shape[0])
  return jnp.reshape(flat_one_hot, inputs.shape)


def ranks(inputs: jnp.ndarray) -> jnp.ndarray:
  return jnp.argsort(jnp.argsort(inputs, axis=-1), axis=-1)


def top_k_hots(values, k):
  n = values.shape[0]
  return jax.nn.one_hot(jnp.argsort(values)[-k:], n)


def scalar_function(inputs):
  """Example function with discrete operation and scalar output."""
  inputs_ranks = ranks(inputs)
  n = inputs.shape[-1]
  true_ranks = ranks(jnp.arange(n))
  return jnp.mean((inputs_ranks - true_ranks) ** 2)


def quad_function(inputs_ranks):
  """Example function with scalar output."""
  n = inputs_ranks.shape[-1]
  true_ranks = ranks(jnp.arange(n))
  return jnp.mean((inputs_ranks - true_ranks) ** 2)


class PerturbationsArgmaxTest(test_util.JaxoptTestCase):

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

    # same as pert_argmax_fun with control variate 
    pert_argmax_fun_rv = jax.jit(perturbations.make_perturbed_argmax(
        argmax_fun=one_hot_argmax,
        num_samples=self.num_samples,
        sigma=self.sigma,
        noise=perturbations.Gumbel(),
        control_variate=True))


    rngs_batch = jax.random.split(self.rng, 2)
    pert_argmax = jax.vmap(pert_argmax_fun)(theta_batch, rngs_batch)
    pert_argmax_rv = jax.vmap(pert_argmax_fun_rv)(theta_batch, rngs_batch)
    soft_argmax = jax.nn.softmax(theta_batch/self.sigma)
    self.assertArraysAllClose(pert_argmax, soft_argmax, atol=3e-2)
    self.assertArraysAllClose(pert_argmax_rv, soft_argmax, atol=3e-2)

    theta_batch_repeat = jnp.array([[-0.6, 1.9, -0.2, 1.1, -1.0],
                                    [-0.6, 1.9, -0.2, 1.1, -1.0]],
                                   dtype=jnp.float32)
    pert_argmax_fun = jax.jit(perturbations.make_perturbed_argmax(
        argmax_fun=one_hot_argmax,
        num_samples=self.num_samples,
        sigma=self.sigma,
        noise=perturbations.Gumbel()))

    # same as pert_argmax_fun but with control variate 
    pert_argmax_fun_rv = jax.jit(perturbations.make_perturbed_argmax(
        argmax_fun=one_hot_argmax,
        num_samples=self.num_samples,
        sigma=self.sigma,
        noise=perturbations.Gumbel(),
        control_variate=True))

    rngs_batch = jax.random.split(self.rng, 2)
    pert_argmax_repeat = jax.vmap(pert_argmax_fun)(theta_batch_repeat,
                                                   rngs_batch)
    pert_argmax_repeat_rv = jax.vmap(pert_argmax_fun_rv)(theta_batch_repeat,
                                                         rngs_batch)

    self.assertArraysAllClose(pert_argmax_repeat[0], pert_argmax_repeat[1],
                              atol=2e-2)
    self.assertArraysAllClose(pert_argmax_repeat_rv[0], pert_argmax_repeat_rv[1],
                              atol=2e-2)

    delta_noise = pert_argmax_repeat[0] - pert_argmax_repeat[1]
    delta_noise_rv = pert_argmax_repeat_rv[0] - pert_argmax_repeat_rv[1]
    self.assertNotAlmostEqual(jnp.linalg.norm(delta_noise), 0)
    self.assertNotAlmostEqual(jnp.linalg.norm(delta_noise_rv), 0)

    def square_loss(theta_batch, rng):
      batch_size = theta_batch.shape[0]
      rngs_batch = jax.random.split(rng, batch_size)
      pert_argmax_batch = jax.vmap(pert_argmax_fun)(theta_batch, rngs_batch)
      return jnp.mean(square_norm_fun(pert_argmax_batch))

    # with control variate 
    def square_loss_rv(theta_batch, rng):
      batch_size = theta_batch.shape[0]
      rngs_batch = jax.random.split(rng, batch_size)
      pert_argmax_batch_rv = jax.vmap(pert_argmax_fun_rv)(theta_batch, rngs_batch)
      return jnp.mean(square_norm_fun(pert_argmax_batch_rv))


    grad_pert = jax.grad(square_loss)(theta_batch, self.rng)
    grad_pert_rv = jax.grad(square_loss_rv)(theta_batch, self.rng)

    def square_loss_soft(theta):
      soft_max = jax.nn.softmax(theta/self.sigma)
      return jnp.mean(square_norm_fun(soft_max))

    grad_soft = jax.grad(square_loss_soft)(theta_batch)
    self.assertArraysAllClose(grad_pert, grad_soft, atol=1e-1)
    self.assertArraysAllClose(grad_pert_rv, grad_soft, atol=1e-1)

    """
    Test ensuring that for small sigma, control variate indeed leads
    to a Jacobian that is closter to that of the softmax.
    """
    sigma = 0.01

    pert_argmax_fun = jax.jit(perturbations.make_perturbed_argmax(
        argmax_fun=one_hot_argmax,
        num_samples=self.num_samples,
        sigma=0.01,
        noise=perturbations.Gumbel()))

    # same as pert_argmax_fun but with control variate 
    pert_argmax_fun_rv = jax.jit(perturbations.make_perturbed_argmax(
        argmax_fun=one_hot_argmax,
        num_samples=self.num_samples,
        sigma=0.01,
        noise=perturbations.Gumbel(),
        control_variate=True))

    rng, _  = jax.random.split(self.rng, 2)
    jac_softmax = jax.jacfwd(jax.nn.softmax)
    jac_pert_argmax_fun = jax.jacfwd(pert_argmax_fun)
    jac_pert_argmax_fun_rv = jax.jacfwd(pert_argmax_fun_rv)

    # calculate the jacobians
    js = jac_softmax(theta_matrix[0])
    jp = jac_pert_argmax_fun(theta_matrix[0], rng)
    jp_rv = jac_pert_argmax_fun_rv(theta_matrix[0], rng)

    # distance of pert argmax jacobians from softmax jacobian
    jac_dist= jnp.linalg.norm((js - jp) ** 2)
    jac_dist_rv = jnp.linalg.norm((js - jp_rv) ** 2)
    self.assertLessEqual(jac_dist_rv, jac_dist)


  def test_distrax(self):
    """Checks that the function is compatible with distrax distributions."""
    try:
      import distrax
    except ImportError:
      return

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

    pert_sign_rv = jax.jit(jax.vmap(perturbations.make_perturbed_argmax(
        argmax_fun=jnp.sign,
        num_samples=10000,
        sigma=self.sigma,
        noise=perturbations.Normal(),
        control_variate=True)))



    theta_batch = jnp.array([[-0.5, 1.2],
                             [-0.4, -1.2],
                             [-0.2, 0.2],
                             [0.1, 0.05]])
    rngs_batch = jax.random.split(self.rng, 4)
    soft_sign = pert_sign(theta_batch, rngs_batch)
    soft_sign_rv = pert_sign_rv(theta_batch, rngs_batch)
    test_sign = 2 * jax.scipy.stats.norm.cdf(theta_batch/self.sigma) - 1
    self.assertArraysAllClose(soft_sign, test_sign, atol=1e-1)
    self.assertArraysAllClose(soft_sign_rv, test_sign, atol=1e-1)

    def pert_sum_or(x_batch, rng):
      batch_size = x_batch.shape[0]
      rngs_batch = jax.random.split(rng, batch_size)
      pert_sign_batch = pert_sign(x_batch, rngs_batch)
      return jnp.sum(jnp.max(pert_sign_batch, axis=-1))

    def pert_sum_or_rv(x_batch, rng):
      batch_size = x_batch.shape[0]
      rngs_batch = jax.random.split(rng, batch_size)
      pert_sign_batch_rv = pert_sign_rv(x_batch, rngs_batch)
      return jnp.sum(jnp.max(pert_sign_batch_rv, axis=-1))


    def explicit_sum_or(x):
      cdf_value = jax.scipy.stats.norm.cdf(x / self.sigma)
      return jnp.sum(jnp.max(2 * cdf_value - 1, axis=-1))

    grad_pert_sum = jax.grad(pert_sum_or)(theta_batch, self.rng)
    grad_pert_sum_rv = jax.grad(pert_sum_or_rv)(theta_batch, self.rng)

    grad_test = jax.grad(explicit_sum_or)(theta_batch)

    self.assertArraysAllClose(1 + grad_pert_sum, 1 + grad_test, atol=1e-1)
    self.assertArraysAllClose(1 + grad_pert_sum_rv, 1 + grad_test, atol=1e-1)

  def test_rank_small_sigma(self):

    theta = jnp.array([-0.8, 0.6, 1.2, -1.0, -0.7, 0.6, 1.1, -1.0, 0.4])
    pert_ranks_fn_small_sigma = jax.jit(perturbations.make_perturbed_argmax(
        argmax_fun=ranks,
        num_samples=self.num_samples,
        sigma=1e-9,
        noise=perturbations.Normal()))

    pert_ranks_fn_small_sigma_rv = jax.jit(perturbations.make_perturbed_argmax(
        argmax_fun=ranks,
        num_samples=self.num_samples,
        sigma=1e-9,
        noise=perturbations.Normal(),
        control_variate=True))

    pert_ranks_small_sigma = pert_ranks_fn_small_sigma(theta, self.rng)
    pert_ranks_small_sigma_rv = pert_ranks_fn_small_sigma_rv(theta, self.rng)
    test_ranks_no_sigma = jnp.array(ranks(theta), dtype='float32')
    self.assertArraysAllClose(pert_ranks_small_sigma,
                              test_ranks_no_sigma, atol=1e-3)
    self.assertArraysAllClose(pert_ranks_small_sigma_rv,
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


class PerturbationsMaxTest(test_util.JaxoptTestCase):

  def setUp(self):
    super().setUp()
    self.num_samples = 1000
    self.sigma = 0.5
    self.rng = jax.random.PRNGKey(0)
    self.theta_batch = jnp.array([[-0.6, 1.9, -0.2, 1.7, -1.0],
                                  [-0.6, 1.0, -0.2, 1.8, -1.0]],
                                 dtype=jnp.float32)

  def test_max_small_sigma(self):
    """Checks that the perturbed max is close to the max for small sigma."""
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
    try:
      import distrax
    except ImportError:
      return

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

  def test_shape(self):
    rngs = jax.random.split(self.rng)
    values = jax.random.normal(rngs[0], (6,))

    def top_2_hots(values_k):
      values = values_k[0]
      return top_k_hots(values, 2)

    diff_top_2_hots = perturbations.make_perturbed_argmax(top_2_hots,
                                                          num_samples=100,
                                                          sigma=0.25)

    def loss_example(values, rng):
      y_true = top_k_hots(jnp.arange(6), 2)
      values_k = jnp.tile(values[jnp.newaxis, :], (2, 1))
      y_pred = diff_top_2_hots(values_k, rng)
      return jnp.sum((y_true - y_pred) ** 2)

    gradient = jax.grad(loss_example)(values, rngs[1])
    self.assertEqual(gradient.shape, values.shape)

class PerturbationsScalarTest(test_util.JaxoptTestCase):

  def setUp(self):
    super().setUp()
    self.num_samples = 1000
    self.sigma = 0.5
    self.rng = jax.random.PRNGKey(0)
    self.theta_batch = jnp.array(
        [[-0.6, 1.9, -0.2, 1.7, -1.0], [-0.6, 1.0, -0.2, 1.8, -1.0]],
        dtype=jnp.float32,
    )

  def test_scalar_small_sigma(self):
    """Checks that the perturbed scalar is close to the max for small sigma."""
    pert_scalar_small_sigma_fun = jax.jit(
        perturbations.make_perturbed_fun(
            fun=scalar_function,
            num_samples=self.num_samples,
            sigma=1e-7,
            noise=perturbations.Gumbel(),
        )
    )
    rngs_batch = jax.random.split(self.rng, 2)
    pert_scalar_small_sigma = jax.vmap(pert_scalar_small_sigma_fun)(
        self.theta_batch, rngs_batch
    )

    self.assertArraysAllClose(
        pert_scalar_small_sigma, jnp.array([5.2, 4.4]), atol=1e-6
    )

  def test_grads(self):
    """Tests that the gradients have the correct shape."""
    pert_scalar_fun = jax.jit(
        perturbations.make_perturbed_fun(
            fun=scalar_function,
            num_samples=self.num_samples,
            sigma=self.sigma,
            noise=perturbations.Gumbel(),
        )
    )

    grad_pert = jax.grad(pert_scalar_fun)(self.theta_batch, self.rng)

    self.assertArraysEqual(grad_pert.shape, self.theta_batch.shape)

  def test_noise_iid(self):
    """Checks that different elements of the batch have different noises."""
    pert_scalar_fun = jax.jit(
        perturbations.make_perturbed_fun(
            fun=scalar_function,
            num_samples=self.num_samples,
            sigma=self.sigma,
            noise=perturbations.Gumbel(),
        )
    )
    theta_batch_repeat = jnp.array([[-0.6, 1.9, -0.2, 1.1, -1.0],
                                    [-0.6, 1.9, -0.2, 1.1, -1.0]],
                                   dtype=jnp.float32)
    rngs_batch = jax.random.split(self.rng, 2)
    pert_scalar_repeat = jax.vmap(pert_scalar_fun)(theta_batch_repeat, 
                                                   rngs_batch)
    self.assertArraysAllClose(pert_scalar_repeat[0], pert_scalar_repeat[1],
                              atol=2e-2)
    delta_noise = pert_scalar_repeat[0] - pert_scalar_repeat[1]
    self.assertNotAlmostEqual(jnp.linalg.norm(delta_noise), 0)

  def test_distrax(self):
    """Checks that the function is compatible with distrax distributions."""
    try:
      import distrax
    except ImportError:
      return

    theta_batch = jnp.array([[-0.5, 0.2, 0.1],
                             [-0.2, 0.1, 0.4]])
    pert_scalar_fun = jax.jit(perturbations.make_perturbed_fun(
        fun=scalar_function,
        num_samples=self.num_samples,
        sigma=self.sigma,
        noise=perturbations.Normal()))
    rngs_batch = jax.random.split(self.rng, 2)
    pert_scalar = jax.vmap(pert_scalar_fun)(theta_batch, rngs_batch)

    dist_scalar_fun = jax.jit(perturbations.make_perturbed_fun(
        fun=scalar_function,
        num_samples=self.num_samples,
        sigma=self.sigma,
        noise=distrax.Normal(loc=0., scale=1.)))
    dist_scalar = jax.vmap(dist_scalar_fun)(theta_batch, rngs_batch)

    self.assertArraysAllClose(pert_scalar, dist_scalar, atol=1e-6)

  def test_grad_finite_diff(self):
    theta = jnp.array([-0.8, 0.6, 1.2, -1.0, -0.7, 0.6, 1.1, -1.0, 0.4])
    # High value of num_samples for this specific test. Not required in normal
    # usecases, as in learning tasks.
    pert_scalar_fun = jax.jit(perturbations.make_perturbed_fun(
        fun=scalar_function,
        num_samples=100_000,
        sigma=self.sigma,
        noise=perturbations.Normal()))

    gradient_pert = jax.grad(pert_scalar_fun)(theta, self.rng)
    eps = 1e-2
    h = jax.random.uniform(self.rng, shape=theta.shape)
    rngs = jax.random.split(self.rng, 2)
    delta_num = (pert_scalar_fun(theta + eps * h, rngs[0]) -
                 pert_scalar_fun(theta - eps * h, rngs[0])) / (2 * eps)
    delta_lin = jnp.sum(gradient_pert * h)

    self.assertArraysAllClose(delta_num, delta_lin, rtol=3e-2)

  def compare_grads(self):
    """Checks composition of gradients with only one sample."""
    pert_ranks_fun = jax.jit(perturbations.make_perturbed_argmax(
        argmax_fun=ranks,
        num_sample=1,
        sigma=self.sigma))
    pert_scalar_fun = jax.jit(perturbations.make_perturbed_fun(
        fun=scalar_function,
        num_samples=1,
        sigma=self.sigma))
    rngs = jax.random.split(self.rng, 2)
    theta = jax.random(rngs[0], shape=(5,))
    pert_ranks = pert_ranks_fun(theta, rngs[1])
    grad_quad = jax.grad(quad_function)
    grad_pert_ranks = grad_quad(pert_ranks)
    jac_fun = jax.jacobian(fun=pert_ranks_fun)
    jac_pert_ranks = jac_fun(theta, rngs[1])
    grad_scalar_fun = jax.grad(pert_scalar_fun)
    grad_scalar = grad_scalar_fun(pert_ranks, rngs[1])
    grad_compose = grad_pert_ranks @ jac_pert_ranks
    self.assertArraysAllClose(grad_scalar, grad_compose)

  def test_fenchel_young_pert(self):
    # Checks the behavior of the FL loss with perturbed optimizers.
    pert_max_one_hot = perturbations.make_perturbed_max(one_hot_argmax,
                                                        num_samples=50000,
                                                        sigma=1.)
    # Artificially high values of num_samples to test asymptotic behavior.
    fy_loss_pert = loss.make_fenchel_young_loss(pert_max_one_hot)
    rngs = jax.random.split(self.rng, 2)
    rngs_batch = jax.random.split(self.rng, 8)
    theta_true = jax.random.uniform(rngs[0], (8, 5))
    pert_argmax = perturbations.make_perturbed_argmax(argmax_fun=one_hot_argmax,
                                                      num_samples=50000,
                                                      sigma=1.)
    y_true = jax.vmap(pert_argmax)(theta_true, rngs_batch)
    theta_random = jax.random.uniform(rngs[1], (8, 5))
    y_random = jax.vmap(pert_argmax)(theta_random, rngs_batch)
    fy_min = jax.vmap(fy_loss_pert)(y_true, theta_true, rngs_batch)
    fy_random = jax.vmap(fy_loss_pert)(y_true, theta_random, rngs_batch)
    # Checks that the loss is minimized for true value of the parameters.
    self.assertGreater(fy_random[0], fy_min[0])
    self.assertGreater(jnp.mean(fy_random), jnp.mean(fy_min))
    grad_random = jax.vmap(jax.grad(fy_loss_pert, argnums=1))(y_true,
                                                              theta_random,
                                                              rngs_batch)
    # Checks that the gradient of the loss takes the correct form.
    self.assertArraysAllClose(grad_random, y_random - y_true)
    y_one_hot = jax.vmap(one_hot_argmax)(theta_true)
    int_one_hot = jnp.where(y_one_hot == 1.)[1]
    loss_one_hot = jax.vmap(fy_loss_pert)(y_one_hot, theta_random, rngs_batch)
    log_loss = jax.vmap(loss.multiclass_logistic_loss)(int_one_hot,
                                                       theta_random)
    # Checks that the FY loss associated to simple perturbed argmax is correct.
    self.assertArraysAllClose(loss_one_hot, log_loss + jnp.euler_gamma,
                              rtol=1e-2)


class PerturbationsAnyDimTest(test_util.JaxoptTestCase):

  def setUp(self):
    super().setUp()
    self.num_samples = 1000
    self.sigma = 0.5
    self.rng = jax.random.PRNGKey(0)
    self.theta_batch = jnp.array(
        [[-0.6, 1.9, -0.2, 1.7, -1.0], [-0.6, 1.0, -0.2, 1.8, -1.0]],
        dtype=jnp.float32,
    )

  def test_scalar_small_sigma(self):
    """Checks that the perturbed scalar is close to the max for small sigma."""
    pert_fun_small_sigma = jax.jit(
        perturbations.make_perturbed_fun(
            fun=ranks,
            num_samples=self.num_samples,
            sigma=1e-7,
            noise=perturbations.Gumbel(),
        )
    )
    rngs_batch = jax.random.split(self.rng, 2)
    pert_out_small_sigma = jax.vmap(pert_fun_small_sigma)(
        self.theta_batch, rngs_batch
    )
    out_no_sigma = 1. * jax.vmap(ranks)(self.theta_batch)

    self.assertArraysAllClose(
        pert_out_small_sigma, out_no_sigma, atol=1e-6
    )

  def test_grads(self):
    """Tests that the gradients have the correct shape."""
    pert_fun = jax.jit(
        perturbations.make_perturbed_fun(
            fun=ranks,
            num_samples=self.num_samples,
            sigma=self.sigma,
            noise=perturbations.Gumbel(),
        )
    )

    grad_pert = jax.jacobian(pert_fun)(self.theta_batch[0], self.rng)
    dim = self.theta_batch.shape[1]
    self.assertArraysEqual(grad_pert.shape, (dim, dim))

  def test_noise_iid(self):
    """Checks that different elements of the batch have different noises."""
    pert_fun = jax.jit(
        perturbations.make_perturbed_fun(
            fun=ranks,
            num_samples=self.num_samples,
            sigma=self.sigma,
            noise=perturbations.Gumbel(),
        )
    )
    theta_batch_repeat = jnp.array([[-0.6, 1.9, -0.2, 1.1, -1.0],
                                    [-0.6, 1.9, -0.2, 1.1, -1.0]],
                                   dtype=jnp.float32)
    rngs_batch = jax.random.split(self.rng, 2)
    pert_repeat = jax.vmap(pert_fun)(theta_batch_repeat,
                                     rngs_batch)
    self.assertArraysAllClose(pert_repeat[0], pert_repeat[1],
                              atol=2e-2)
    delta_noise = pert_repeat[0] - pert_repeat[1]
    self.assertNotAlmostEqual(jnp.linalg.norm(delta_noise), 0)

  def test_distrax(self):
    """Checks that the function is compatible with distrax distributions."""
    try:
      import distrax
    except ImportError:
      return

    theta_batch = jnp.array([[-0.5, 0.2, 0.1],
                             [-0.2, 0.1, 0.4]])
    pert_fun = jax.jit(perturbations.make_perturbed_fun(
        fun=ranks,
        num_samples=self.num_samples,
        sigma=self.sigma,
        noise=perturbations.Normal()))
    rngs_batch = jax.random.split(self.rng, 2)
    pert_scalar = jax.vmap(pert_fun)(theta_batch, rngs_batch)

    dist_scalar_fun = jax.jit(perturbations.make_perturbed_fun(
        fun=ranks,
        num_samples=self.num_samples,
        sigma=self.sigma,
        noise=distrax.Normal(loc=0., scale=1.)))
    dist_scalar = jax.vmap(dist_scalar_fun)(theta_batch, rngs_batch)

    self.assertArraysAllClose(pert_scalar, dist_scalar, atol=1e-6)

  def compose_grads(self):
    """Checks composition of gradients."""
    pert_ranks_fun = jax.jit(perturbations.make_perturbed_argmax(
        argmax_fun=ranks,
        num_sample=1,
        sigma=self.sigma))
    def scalar_pert_inside(inputs, rng):
      pert_ranks_inputs = pert_ranks_fun(inputs, rng)
      return jnp.sum(pert_ranks_inputs ** 2)
    def quad_local(inputs):
      return jnp.sum(inputs ** 2)
    rngs = jax.random.split(self.rng, 2)
    theta = jax.random(rngs[0], shape=(5,))
    pert_ranks = pert_ranks_fun(theta, rngs[1])
    grad_quad = jax.grad(quad_local)
    grad_pert_ranks = grad_quad(pert_ranks)
    jac_fun = jax.jacobian(fun=pert_ranks_fun)
    jac_pert_ranks = jac_fun(theta, rngs[1])
    grad_scalar_fun = jax.grad(scalar_pert_inside)
    grad_scalar = grad_scalar_fun(pert_ranks, rngs[1])
    grad_compose = grad_pert_ranks @ jac_pert_ranks
    self.assertArraysAllClose(grad_scalar, grad_compose)


if __name__ == '__main__':
  absltest.main()
