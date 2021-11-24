# Copyright 2020 DeepMind Technologies Limited
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

"""
VAE example with Haiku and JAXopt.
==================================
"""

from typing import NamedTuple

from absl import app
from absl import flags
import haiku as hk
import jax
import jax.numpy as jnp
from jaxopt import OptaxSolver
import numpy as onp
import optax
import tensorflow_datasets as tfds
import tensorflow as tf

MNIST_IMAGE_SHAPE = (28, 28, 1)

flags.DEFINE_integer("batch_size", 128, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 1000, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 100, "How often to evaluate the model.")
flags.DEFINE_integer("random_seed", 42, "Random seed.")
FLAGS = flags.FLAGS


def load_dataset(split, batch_size):
  ds = tfds.load("binarized_mnist", split=split, shuffle_files=True,
                 read_config=tfds.ReadConfig(shuffle_seed=FLAGS.random_seed))
  ds = ds.shuffle(buffer_size=10 * batch_size, seed=FLAGS.random_seed)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=5)
  ds = ds.repeat()
  return iter(tfds.as_numpy(ds))


class Encoder(hk.Module):
  """Encoder model."""

  def __init__(self, hidden_size=512, latent_size=10):
    super().__init__()
    self._hidden_size = hidden_size
    self._latent_size = latent_size

  def __call__(self, x):
    x = hk.Flatten()(x)
    x = hk.Linear(self._hidden_size)(x)
    x = jax.nn.relu(x)

    mean = hk.Linear(self._latent_size)(x)
    log_stddev = hk.Linear(self._latent_size)(x)
    stddev = jnp.exp(log_stddev)

    return mean, stddev


class Decoder(hk.Module):
  """Decoder model."""

  def __init__(self, hidden_size=512, output_shape=MNIST_IMAGE_SHAPE):
    super().__init__()
    self._hidden_size = hidden_size
    self._output_shape = output_shape

  def __call__(self, z):
    z = hk.Linear(self._hidden_size)(z)
    z = jax.nn.relu(z)

    logits = hk.Linear(onp.prod(self._output_shape))(z)
    logits = jnp.reshape(logits, (-1, *self._output_shape))

    return logits


class VAEOutput(NamedTuple):
  image: jnp.ndarray
  mean: jnp.ndarray
  stddev: jnp.ndarray
  logits: jnp.ndarray


class VariationalAutoEncoder(hk.Module):
  """Main VAE model class, uses Encoder & Decoder under the hood."""

  def __init__(self, hidden_size=512, latent_size=10,
               output_shape=MNIST_IMAGE_SHAPE):
    super().__init__()
    self._hidden_size = hidden_size
    self._latent_size = latent_size
    self._output_shape = output_shape

  def __call__(self, x):
    x = x.astype(jnp.float32)
    mean, stddev = Encoder(self._hidden_size, self._latent_size)(x)
    z = mean + stddev * jax.random.normal(hk.next_rng_key(), mean.shape)
    logits = Decoder(self._hidden_size, self._output_shape)(z)

    p = jax.nn.sigmoid(logits)
    image = jax.random.bernoulli(hk.next_rng_key(), p)

    return VAEOutput(image, mean, stddev, logits)


def binary_cross_entropy(x: jnp.ndarray, logits: jnp.ndarray) -> jnp.ndarray:
  if x.shape != logits.shape:
    raise ValueError("inputs x and logits must be of the same shape")

  x = jnp.reshape(x, (x.shape[0], -1))
  logits = jnp.reshape(logits, (logits.shape[0], -1))

  return -jnp.sum(x * logits - jnp.logaddexp(0.0, logits), axis=-1)


def kl_gaussian(mean: jnp.ndarray, var: jnp.ndarray) -> jnp.ndarray:
  r"""Calculate KL divergence between given and standard gaussian distributions.
  KL(p, q) = H(p, q) - H(p) = -\int p(x)log(q(x))dx - -\int p(x)log(p(x))dx
           = 0.5 * [log(|s2|/|s1|) - 1 + tr(s1/s2) + (m1-m2)^2/s2]
           = 0.5 * [-log(|s1|) - 1 + tr(s1) + m1^2] (if m2 = 0, s2 = 1)
  Args:
    mean: mean vector of the first distribution
    var: diagonal vector of covariance matrix of the first distribution
  Returns:
    A scalar representing KL divergence of the two Gaussian distributions.
  """
  return 0.5 * jnp.sum(-jnp.log(var) - 1.0 + var + jnp.square(mean), axis=-1)


# pylint: disable=unnecessary-lambda
model = hk.transform(lambda x: VariationalAutoEncoder()(x))


@jax.jit
def loss_fun(params, rng_key, batch):
  """ELBO loss: E_p[log(x)] - KL(d||q), where p ~ Be(0.5) and q ~ N(0,1)."""
  outputs = model.apply(params, rng_key, batch["image"])
  log_likelihood = -binary_cross_entropy(batch["image"], outputs.logits)
  kl = kl_gaussian(outputs.mean, jnp.square(outputs.stddev))
  elbo = log_likelihood - kl
  return -jnp.mean(elbo)


def main(argv):
  del argv

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  # Initialize solver.
  solver = OptaxSolver(opt=optax.adam(FLAGS.learning_rate), fun=loss_fun)

  # Set up data iterators.
  train_ds = load_dataset(tfds.Split.TRAIN, FLAGS.batch_size)
  test_ds = load_dataset(tfds.Split.TEST, FLAGS.batch_size)

  # Initialize parameters.
  rng_seq = hk.PRNGSequence(FLAGS.random_seed)
  params = model.init(next(rng_seq), onp.zeros((1, *MNIST_IMAGE_SHAPE)))
  state = solver.init_state(params)

  # Run training loop.
  for step in range(FLAGS.training_steps):
    params, state = solver.update(params=params, state=state,
                                  rng_key=next(rng_seq),
                                  batch=next(train_ds))

    if step % FLAGS.eval_frequency == 0:
      val_loss = loss_fun(params, next(rng_seq), next(test_ds))
      print(f"STEP: {step}; Validation ELBO: {val_loss:.3f}")

if __name__ == "__main__":
  app.run(main)
