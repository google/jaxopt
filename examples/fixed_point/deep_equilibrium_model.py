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
Deep Equilibrium (DEQ) model in Flax with Anderson acceleration.
================================================================

This implementation is strongly inspired by the Pytorch code snippets in [3].

A similar model called "implicit deep learning" is also proposed in [2].

In practice BatchNormalization and initialization of weights in convolutions are
important to ensure convergence.

[1] Bai, S., Kolter, J.Z. and Koltun, V., 2019. Deep Equilibrium Models.
Advances in Neural Information Processing Systems, 32, pp.690-701.

[2] El Ghaoui, L., Gu, F., Travacca, B., Askari, A. and Tsai, A., 2021.
Implicit deep learning. SIAM Journal on Mathematics of Data Science, 3(3), pp.930-958.

[3] http://implicit-layers-tutorial.org/deep_equilibrium_models/
"""

from functools import partial
from typing import Any, Mapping, Tuple, Callable

from absl import app
from absl import flags

import flax
from flax import linen as nn

import jax
import jax.numpy as jnp
from jax.tree_util import tree_structure

import jaxopt
from jaxopt import loss
from jaxopt import OptaxSolver
from jaxopt import FixedPointIteration
from jaxopt import AndersonAcceleration
from jaxopt.linear_solve import solve_gmres, solve_normal_cg
from jaxopt.tree_util import tree_add, tree_sub, tree_l2_norm

import optax

import tensorflow_datasets as tfds
from collections import namedtuple


dataset_names = [
    "mnist", "kmnist", "emnist", "fashion_mnist", "cifar10", "cifar100"
]

# training hyper-parameters
flags.DEFINE_float("l2reg", 0., "L2 regularization.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_integer("maxiter", 1000, "Maximum number of iterations.")
flags.DEFINE_enum("dataset", "cifar10", dataset_names, "Dataset to train on.")
flags.DEFINE_integer("net_width", 1, "Multiplicator of neural network width.")
flags.DEFINE_integer("evaluation_frequency", 1, "Number of iterations between two evaluation measures.")
flags.DEFINE_integer("train_batch_size", 256, "Batch size at train time.")
flags.DEFINE_integer("test_batch_size", 1024, "Batch size at test time.")


solvers = ["normal_cg", "gmres", "anderson"]
flags.DEFINE_enum("backward_solver", "normal_cg", solvers, "Solver of linear sytem in implicit differentiation.")

# anderson acceleration parameters
flags.DEFINE_enum("forward_solver", "anderson", ["anderson", "fixed_point"], "Whether to use Anderson acceleration.")
flags.DEFINE_integer("forward_maxiter", 20, "Number of fixed point iterations.")
flags.DEFINE_float("forward_tol", 1e-2, "Tolerance in fixed point iterations.")
flags.DEFINE_integer("anderson_history_size", 5, "Size of history in Anderson updates.")
flags.DEFINE_float("anderson_ridge", 1e-4, "Ridge regularization in Anderson updates.")

FLAGS = flags.FLAGS


def load_dataset(split, *, is_training, batch_size):
  """Loads the dataset as a generator of batches."""
  ds, ds_info = tfds.load(f"{FLAGS.dataset}:3.*.*", split=split, as_supervised=True, with_info=True)
  ds = ds.cache().repeat()
  if is_training:
    ds = ds.shuffle(10 * batch_size, seed=0)
  ds = ds.batch(batch_size)
  return iter(tfds.as_numpy(ds)), ds_info


class ResNetBlock(nn.Module):
  """ResNet block."""
  channels: int
  channels_bottleneck: int
  num_groups: int = 8
  kernel_size: Tuple[int, int] = (3, 3)
  use_bias: bool = False
  act: Callable = nn.relu

  @nn.compact
  def __call__(self, z, x):
    # Note that stddev=0.01 is important to avoid divergence.
    # Empirically it ensures that fixed point iterations converge.
    y = z
    y = nn.Conv(features=self.channels_bottleneck, kernel_size=self.kernel_size,
                padding="SAME", use_bias=self.use_bias,
                kernel_init=jax.nn.initializers.normal(stddev=0.01))(y)
    y = self.act(y)
    y = nn.GroupNorm(num_groups=self.num_groups)(y)
    y = nn.Conv(features=self.channels, kernel_size=self.kernel_size,
                padding="SAME", use_bias=self.use_bias,
                kernel_init=jax.nn.initializers.normal(stddev=0.01))(y)
    y = y + x
    y = nn.GroupNorm(num_groups=self.num_groups)(y)
    y = y + z
    y = self.act(y)
    y = nn.GroupNorm(num_groups=self.num_groups)(y)
    return y


class DEQFixedPoint(nn.Module):
    """Batched computation of the fixed point of a Module ``block`` using ``fixed_point_solver``."""
    block: Any  # nn.Module
    fixed_point_solver: Any  # jaxopt.AndersonAcceleration or jaxopt.FixedPointIteration

    @nn.compact
    def __call__(self, x):
        init = lambda rng, x: self.block.init(rng, x[0], x[0])  # shape of a single example
        block_params = self.param("block_params", init, x)

        def block_apply(z, x, block_params):
          return self.block.apply(block_params, z, x)

        solver = self.fixed_point_solver(fixed_point_fun=block_apply)
        def batch_run(x, block_params):
          return solver.run(x, x, block_params)[0]

        # We use vmap since we want to compute the fixed point separately for each example in the batch.
        return jax.vmap(batch_run, in_axes=(0,None), out_axes=0)(x, block_params)


class FullDEQ(nn.Module):
    """DEQ model."""
    num_classes: int
    channels: int
    channels_bottleneck: int
    fixed_point_solver: Callable

    @nn.compact
    def __call__(self, x, train):
        # Note that x is a batch of examples:
        # because of BatchNorm we cannot define the forward pass in the network for a single example.
        x = nn.Conv(features=self.channels, kernel_size=(3,3), use_bias=True, padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-5)(x)
        block = ResNetBlock(self.channels, self.channels_bottleneck)
        deq_fixed_point = DEQFixedPoint(block, self.fixed_point_solver)
        x = deq_fixed_point(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-5)(x)
        x = nn.avg_pool(x, window_shape=(8,8), padding="SAME")
        x = x.reshape(x.shape[:-3]+(-1,))  # flatten
        x = nn.Dense(self.num_classes)(x)
        return x

# For completeness, we also allow Anderson acceleration for solving
# the implicit differentiation linear system occurring in the backward pass.
def solve_linear_system_fixed_point(matvec, v):
  """Solve linear system matvec(u) = v.

  The solution u* of the system is the fixed point of:
    T(u) = matvec(u) + u - v
  """
  def fixed_point_fun(u):
    d_1_T_transpose_u = tree_add(matvec(u), u)
    return tree_sub(d_1_T_transpose_u, v)

  aa = AndersonAcceleration(fixed_point_fun,
                            history_size=FLAGS.anderson_history_size, tol=1e-2,
                            ridge=FLAGS.anderson_ridge, maxiter=20)
  return aa.run(v)[0]


def main(argv):
    del argv

    # Solver used for implicit differentiation (backward pass).
    if FLAGS.backward_solver == "normal_cg":
        implicit_solver = partial(solve_normal_cg, tol=1e-2, maxiter=20)
    elif FLAGS.backward_solver == "gmres":
        implicit_solver = partial(solve_gmres, tol=1e-2, maxiter=20)
    elif FLAGS.backward_solver == "anderson":
        implicit_solver = solve_linear_system_fixed_point

    # Solver used for fixed point resolution (forward pass).
    if FLAGS.forward_solver == "anderson":
        fixed_point_solver = partial(AndersonAcceleration,
                                     history_size=FLAGS.anderson_history_size, ridge=FLAGS.anderson_ridge,
                                     maxiter=FLAGS.forward_maxiter, tol=FLAGS.forward_tol,
                                     implicit_diff=True, implicit_diff_solve=implicit_solver)
    else:
        fixed_point_solver = partial(FixedPointIteration,
                                     maxiter=FLAGS.forward_maxiter, tol=FLAGS.forward_tol,
                                     implicit_diff=True, implicit_diff_solve=implicit_solver)

    train_ds, ds_info = load_dataset("train", is_training=True, batch_size=FLAGS.train_batch_size)
    test_ds, _ = load_dataset("test", is_training=False, batch_size=FLAGS.test_batch_size)
    input_shape = (1,) + ds_info.features["image"].shape
    num_classes = ds_info.features["label"].num_classes

    net = FullDEQ(num_classes, 3*8*FLAGS.net_width, 4*8*FLAGS.net_width, fixed_point_solver)

    def predict(all_params, images, train):
        """Forward pass in the network on the images."""
        x = images.astype(jnp.float32) / 255.
        mutable = ["batch_stats"] if train else False
        return net.apply(all_params, x, train=train, mutable=mutable)

    logistic_loss = jax.vmap(loss.multiclass_logistic_loss)
    def loss_from_logits(logits, labels, l2reg):
        """Compute the loss of the network."""
        sqnorm = tree_l2_norm(params, squared=True)
        loss_value = jnp.mean(logistic_loss(labels, logits))
        loss = loss_value + 0.5 * l2reg * sqnorm
        return loss

    @jax.jit
    def test_loss_and_accuracy(params, batch_stats, l2reg, data):
        """Return testing loss and accuracy."""
        all_vars = {"params":params, "batch_stats":batch_stats}
        images, labels = data
        logits = predict(all_vars, images, train=False)
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
        loss = loss_from_logits(logits, labels, l2reg)
        return loss, accuracy

    @jax.jit
    def train_loss_fun(params, batch_stats, l2reg, data):
        """Return training loss and new network state (i.e batch_stats)."""
        all_vars = {"params":params, "batch_stats":batch_stats}
        images, labels = data
        logits, net_state = predict(all_vars, images, train=True)
        loss = loss_from_logits(logits, labels, l2reg)
        return loss, net_state


    def print_accuracy(params, batch_stats, state):
        if state.iter_num % FLAGS.evaluation_frequency == 0:
            # Periodically evaluate classification accuracy on test set.
            data = next(test_ds)
            loss, accuracy = test_loss_and_accuracy(params, batch_stats, FLAGS.l2reg, data)
            print(f"[Step {state.iter_num}]\tTest accuracy: {accuracy:.3f}\tTest loss: {loss:.3f}.")
        return params, state


    # Initialize solver and parameters.
    solver = OptaxSolver(opt=optax.adam(FLAGS.learning_rate), fun=train_loss_fun,
                         maxiter=FLAGS.maxiter, pre_update=None,
                         has_aux=True)

    rng = jax.random.PRNGKey(0)
    init_vars = net.init(rng, jnp.ones(input_shape), train=True)
    params = init_vars["params"]
    batch_stats = init_vars["batch_stats"]
    state = solver.init_state(params)

    @jax.jit
    def jitted_update(params, state, batch_stats, data):
        return solver.update(params, state, batch_stats, FLAGS.l2reg, data)

    for iternum in range(solver.maxiter):
        print_accuracy(params, batch_stats, state)
        params, state = jitted_update(params, state, batch_stats, next(train_ds))
        net_state = state.aux
        batch_stats = net_state["batch_stats"]

    print_accuracy(params, batch_stats, state)


if __name__ == "__main__":
    app.run(main)
