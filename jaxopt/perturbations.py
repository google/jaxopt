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
"""Introduces differentiation via perturbations."""

from typing import Tuple, Callable
import jax
import jax.numpy as jnp


class Normal:
  """Normal distribution."""

  def sample(self,
             seed: jax.random.PRNGKey,
             sample_shape: Tuple[int]) -> jnp.array:
    return jax.random.normal(seed, sample_shape)

  def log_prob(self, inputs: jnp.array) -> jnp.array:
    return -0.5 * inputs ** 2


class Gumbel:
  """Gumbel distribution."""

  def sample(self,
             seed: jax.random.PRNGKey,
             sample_shape: Tuple[int]) -> jnp.array:
    return jax.random.gumbel(seed, sample_shape)

  def log_prob(self, inputs: jnp.array) -> jnp.array:
    return -inputs - jnp.exp(-inputs)


def make_perturbed_argmax(argmax_fun: Callable[[jnp.array], jnp.array],
                          num_samples: int = 1000,
                          sigma: float = 0.1,
                          noise=Gumbel()):
  """Transforms a function into a differentiable version with perturbations.

  Args:
    argmax_fun: the argmax function to transform into a differentiable version.
      The signature for argmax_fn currently supported for custom jvp and jit is:
      + input [D1, ..., Dk], output [D1, ..., Dk], k >= 1
    num_samples: an int, the number of perturbed outputs to average over.
    sigma: a float, the scale of the random perturbation.
    noise: a distribution object that must implement a sample function and a
      log-pdf of the desired distribution, similar to the examples above.
      Default is Gumbel distribution.

  Returns:
    A function with the same signature (and an rng) that can be differentiated.

  Example:
    Given an argmax function, e.g.

      def argmax_fun(x):
        return jax.nn.one_hot(jnp.argmax(x), x.shape[0])

      pert_argmax_fun = make_perturbed_argmax(argmax_fun,
                                              num_samples=200,
                                              sigma=0.01)

    Then pert_argmax_fun is differentiable, a perturbed version of argmax_fun.
    Since it relies on randomness, it requires an rng key

      pert_argmax = pert_argmax_fun(x, rng)

    When handling a batched input, vmap can be applied to this function, with
    some care in splitting the rng key.

      batch_size = x_batch.shape[0]
      rngs_batch = jax.random.split(rng, batch_size)
      pert_argmax = jax.vmap(pert_argmax_fun)(x_batch, rngs_batch)

    Further, if the argmax_fun is jittable, then so is pert_argmax_fun.
  """

  @jax.custom_jvp
  def forward_pert(inputs, rng):
    samples = noise.sample(seed=rng,
                           sample_shape=(num_samples,) + inputs.shape)
    output_pert = jax.vmap(argmax_fun)(inputs + sigma * samples)
    return jnp.mean(output_pert, axis=0)

  def pert_jvp(tangent, _, inputs, rng):
    samples = noise.sample(seed=rng,
                           sample_shape=(num_samples,) + inputs.shape)
    output_pert = jax.vmap(argmax_fun)(inputs + sigma * samples)
    # noise.log_prob corresponds to -nu in the paper.
    nabla_z_flat = -jax.vmap(jax.grad(noise.log_prob))(samples.reshape([-1]))
    tangent_flat = 1.0 / (num_samples * sigma) * jnp.einsum(
        'nd,ne,e->d',
        jnp.reshape(output_pert, (num_samples, -1)),
        jnp.reshape(nabla_z_flat, (num_samples, -1)),
        jnp.reshape(tangent, (-1,)))
    return jnp.reshape(tangent_flat, inputs.shape)

  forward_pert.defjvps(pert_jvp, None)

  return forward_pert


def make_perturbed_max(argmax_fun: Callable[[jnp.array], jnp.array],
                       num_samples: int = 1000,
                       sigma: float = 0.1,
                       noise=Gumbel()):
  """Turns an argmax in a differentiable version of the max with perturbations.

  Args:
    argmax_fun: the argmax function to transform into a differentiable version.
      The signature for argmax_fn currently supported for custom jvp and jit is:
      + input [D1, ..., Dk], output [D1, ..., Dk], k >= 1
    num_samples: an int, the number of perturbed outputs to average over.
    sigma: a float, the scale of the random perturbation.
    noise: a distribution object that must implement a sample function and a
      log-pdf of the desired distribution, similar to the examples above.
      Default is Gumbel distribution.

  Returns:
    A function with the same inputs (and an rng) that can be differentiated.

  Example:
    Given an argmax function, e.g.

      def argmax_fun(x):
        return jax.nn.one_hot(jnp.argmax(x), x.shape[0])

      pert_max_fun = make_perturbed_max(argmax_fun,
                                        num_samples=200,
                                        sigma=0.01)

    Then pert_max_fun is differentiable, a perturbed version of the associated
    max to argmax_fun. Since it relies on randomness, it requires an rng key

      pert_max = pert_max_fun(x, rng)

    When handling a batched input, vmap can be applied to this function, with
    some care in splitting the rng key.

      batch_size = x_batch.shape[0]
      rngs_batch = jax.random.split(rng, batch_size)
      pert_max = jax.vmap(pert_max_fun)(x_batch, rngs_batch)

    Further, if the argmax_fun is jittable, then so is pert_max_fun.
  """

  @jax.custom_jvp
  def forward_pert(inputs, rng):
    samples = noise.sample(seed=rng,
                           sample_shape=(num_samples,) + inputs.shape)
    output_pert = jax.vmap(argmax_fun)(inputs + sigma * samples)
    max_values = jnp.einsum('nd,nd->n',
                            jnp.reshape(inputs + sigma * samples,
                                        (num_samples, -1)),
                            jnp.reshape(output_pert, (num_samples, -1)))
    return jnp.mean(max_values)

  def pert_jvp(tangent, _, inputs, rng):
    pert_argmax_fun = make_perturbed_argmax(argmax_fun,
                                            num_samples,
                                            sigma,
                                            noise)
    pert_argmax = pert_argmax_fun(inputs, rng)
    return jnp.sum(pert_argmax * tangent)

  forward_pert.defjvps(pert_jvp, None)

  return forward_pert
