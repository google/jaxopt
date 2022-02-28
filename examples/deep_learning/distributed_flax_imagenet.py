# Copyright 2022 Google LLC.
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
====================================
SPMD ResNet example with Flax and JAXopt.
====================================
The purpose of this example is to illustrate how JAXopt solvers can be easily
used for distributed training thanks to `jax.pjit`. In this case, we begin by
implementing data parallel training of a ResNet50 model on the ImageNet dataset
as a fork of Flax's official ImageNet example. General aspects to pay attention
to include:
 + How auxiliary information (e.g. Flax mutables, model outputs from train
     metrics, etc) can be extracted from `loss_fun` using the `state.aux` field
     of JAXopt's optimizer state.
 + How `jax.pjit` can be used to easily port single-device training loops to
     distributed training loops.

Running on Google Cloud TPU:

1. Follow the instructions in Flax's official ImageNet example to set a single
   VM with 8 TPUs (`--accelerator_type v3-8`).
2. Likewise, follow the instructions in Flax's official ImageNet example to
   prepare the ImageNet dataset and ensure the `TFDS_DATA_DIR` environment
   variable has been set appropriately.

You may finally run the example as
```
python3 distributed_flax_imagenet.py --workdir=$HOME/spmd_flax_imagenet
```

NOTES: this example supports TPU pod slices (e.g. `--accelerator_type v3-32`) as
well as hosts with one or more GPUs attached. However, CPU-only execution is not
yet supported.
"""


import functools
import itertools
import os
import time
from typing import Any, Callable, Iterator, Mapping, NamedTuple, Optional, Sequence, Type, Tuple, Union

from absl import app
from absl import flags
from absl import logging

from chex import Array, ArrayTree, Numeric, PRNGKey

from clu import checkpoint
from clu import metric_writers
from clu import metrics as clu_metrics
from clu import periodic_actions

from flax import linen as nn
from flax import struct

import jax
from jax import numpy as jnp
from jax import random
from jax.experimental.maps import Mesh
from jax.experimental.pjit import PartitionSpec
from jax.experimental.pjit import pjit

import jaxopt
from jaxopt import tree_util

import numpy as np
import optax

import tensorflow as tf
import tensorflow_datasets as tfds


### Constants.

# Input pipeline-related constants.
IMAGE_SIZE = 224
CROP_PADDING = 32
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]
# Model-related constants.
SUPPORTED_MODELS = ['Resnet1', 'Resnet18', 'ResNet34', 'ResNet50', 'ResNet101',
                    'ResNet152', 'ResNet200']
NUM_CLASSES = 1000


### Type aliases.
ArrayDType = jnp.lax_numpy._ScalarMeta  # pylint: disable=protected-access
Batch = Mapping[str, Any]
DataIter = Iterator[Batch]
LearningRateFn = Callable[[int], Numeric]
Metrics = Mapping[str, Numeric]
ModuleDef = Any


### Input flags.

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')

flags.DEFINE_enum('model', 'ResNet50', SUPPORTED_MODELS, 'Model to use.')
flags.DEFINE_string('dataset', 'imagenet2012:5.*.*', 'TFDS builder name.')

flags.DEFINE_float('learning_rate', 0.1, 'Learning rate.')
flags.DEFINE_integer('warmup_epochs', 5, 'Number of warmup epochs.')
flags.DEFINE_float('momentum', 0.9, 'Momentum.')
flags.DEFINE_integer('batch_size', 1024, 'Global batch size.')

flags.DEFINE_integer('num_epochs', 100, 'Number of training epochs.')
flags.DEFINE_integer('log_every_steps', 100, 'Number of steps between logging.')

flags.DEFINE_bool('cache', True, 'Whether to cache the dataset.')
flags.DEFINE_bool('half_precision', True, 'Whether to use FP16.')

flags.DEFINE_integer('num_train_steps', -1, 'Number of training steps.')
flags.DEFINE_integer('steps_per_eval', -1, 'Number of steps between logging.')

flags.DEFINE_integer('seed', 0, 'Seed for PRNG.')


### Input pipeline (adapted from `flax/examples/imagenet/input_pipeline.py`).


def distorted_bounding_box_crop(
    image_bytes: tf.Tensor,
    bbox: tf.Tensor,
    min_object_covered: Optional[Union[float, tf.Tensor]] = 0.1,
    aspect_ratio_range: Optional[Sequence[float]] = (0.75, 1.33),
    area_range: Optional[Sequence[float]] = (0.05, 1.0),
    max_attempts: Optional[int] = 100,
) -> tf.Tensor:
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image_bytes: `Tensor` of binary image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
  Returns:
    cropped image `Tensor`
  """
  shape = tf.io.extract_jpeg_shape(image_bytes)
  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      shape,
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=max_attempts,
      use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, _ = sample_distorted_bounding_box

  # Crop the image to the specified bounding box.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
  image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

  return image


def _resize(image, image_size):
  return tf.image.resize([image], [image_size, image_size],
                         method=tf.image.ResizeMethod.BICUBIC)[0]


def _at_least_x_are_equal(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes, image_size):
  """Make a random crop of image_size."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = distorted_bounding_box_crop(
      image_bytes,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=10)
  original_shape = tf.io.extract_jpeg_shape(image_bytes)
  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

  image = tf.cond(
      bad,
      lambda: _decode_and_center_crop(image_bytes, image_size),
      lambda: _resize(image, image_size))

  return image


def _decode_and_center_crop(image_bytes, image_size):
  """Crops to center of image with padding then scales image_size."""
  shape = tf.io.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + CROP_PADDING)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)),
      tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([offset_height, offset_width,
                          padded_center_crop_size, padded_center_crop_size])
  image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  image = _resize(image, image_size)

  return image


def normalize_image(image: tf.Tensor) -> tf.Tensor:
  image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
  return image


def create_split(
    dataset_builder: tfds.core.DatasetBuilder,
    batch_size: int,
    train: bool,
    dtype: tf.DType = tf.float32,
    image_size: int = IMAGE_SIZE,
    cache: bool = False,
) -> tf.data.Dataset:
  """Creates a split from the ImageNet dataset using TensorFlow Datasets.

  Args:
    dataset_builder: TFDS dataset builder for ImageNet.
    batch_size: the batch size returned by the data pipeline.
    train: whether to load the train or evaluation split.
    dtype: data type of the image.
    image_size: the target size of the images.
    cache: whether to cache the dataset.
  Returns:
    A `tf.data.Dataset`.
  """
  split = 'train' if train else 'validation'
  num_examples = dataset_builder.info.splits[split].num_examples
  split_size = num_examples // jax.process_count()
  start = jax.process_index() * split_size
  split = f'{split}[{start}:{start + split_size}]'

  def decode_example(example):
    decode_fn = _decode_and_random_crop if train else _decode_and_center_crop
    image = decode_fn(example['image'], image_size)
    image = tf.reshape(image, [image_size, image_size, 3])
    if train:
      image = tf.image.random_flip_left_right(image)
    image = normalize_image(image)
    image = tf.image.convert_image_dtype(image, dtype=dtype)
    return {'image': image, 'label': example['label']}

  ds = dataset_builder.as_dataset(
      split=split,
      decoders={'image': tfds.decode.SkipDecoding()},
  )
  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  if cache:
    ds = ds.cache()

  if train:
    ds = ds.repeat()
    ds = ds.shuffle(16 * batch_size, seed=0)

  ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size, drop_remainder=True)

  if not train:
    ds = ds.repeat()

  return ds.prefetch(10)


def create_input_iterators(
    dataset_builder: tfds.core.DatasetBuilder,
    batch_size: int,
    half_precision: bool = True,
    image_size: int = IMAGE_SIZE,
    cache: bool = False,
) -> Tuple[DataIter, DataIter]:
  """Returns train and evaluation data iterators.

  Args:
    dataset_builder: TFDS dataset builder for ImageNet.
    batch_size: the (local) batch size returned by the data pipeline.
    half_precision: whether to use FP16..
    image_size: the target size of the images.
    cache: whether to cache the dataset.

  Returns:
    A tuple of `tf.data.Dataset` iterators over minibatches.
  """
  input_dtype = tf.float32
  if half_precision:
    platform = jax.local_devices()[0].platform
    input_dtype = tf.bfloat16 if platform == 'tpu' else tf.float16

  train_ds = create_split(
      dataset_builder, batch_size, True, input_dtype, image_size, cache)
  eval_ds = create_split(
      dataset_builder, batch_size, False, input_dtype, image_size, cache)

  train_ds, eval_ds = train_ds.as_numpy_iterator(), eval_ds.as_numpy_iterator()
  return iter(train_ds), iter(eval_ds)


### Model (adapted from `flax/examples/imagenet/models.py`).


class ResNetBlock(nn.Module):
  """ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x,):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.filters, (1, 1))(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), self.strides)(y)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters * 4, (1, 1))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters * 4, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class ResNet(nn.Module):
  """ResNetV1."""
  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  num_classes: int
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu

  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)
    norm = functools.partial(nn.BatchNorm,
                             use_running_average=not train,
                             momentum=0.9,
                             epsilon=1e-5,
                             dtype=self.dtype)

    x = conv(self.num_filters, (7, 7), (2, 2),
             padding=[(3, 3), (3, 3)],
             name='conv_init')(x)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(self.num_filters * 2 ** i,
                           strides=strides,
                           conv=conv,
                           norm=norm,
                           act=self.act)(x)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    return x


MODELS = {
    'ResNet1': functools.partial(
        ResNet, stage_sizes=[1], block_cls=ResNetBlock),
    'ResNet18': functools.partial(
        ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock),
    'ResNet34': functools.partial(
        ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock),
    'ResNet50': functools.partial(
        ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock),
    'ResNet101': functools.partial(
        ResNet, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock),
    'ResNet152': functools.partial(
        ResNet, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock),
    'ResNet200': functools.partial(
        ResNet, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock),
}


### SPMD utilities.


def setup_data_parallel_mesh():
  global_mesh = Mesh(np.asarray(jax.devices(), dtype=object), ['data'])
  jax.experimental.maps.thread_resources.env = (
      jax.experimental.maps.ResourceEnv(physical_mesh=global_mesh, loops=()))


### Training loop utilities.


class LossFnAux(NamedTuple):
  """Stores auxiliary values computed during loss function eval for reuse."""
  batch_stats: ArrayTree
  logits: Array


def cross_entropy_loss(labels, logits, **_):
  xentropy = jax.vmap(jaxopt.loss.multiclass_logistic_loss)(labels, logits)
  return jnp.mean(xentropy)


def loss_fun(
    params: ArrayTree,
    batch_stats: ArrayTree,
    batch: Batch,
    model: nn.Module,
    weight_decay: float = 1e-4,
) -> Tuple[Numeric, LossFnAux]:
  """Loss function used for training."""
  logits, new_mutable_variables = model.apply(
      {'params': params, 'batch_stats': batch_stats},
      batch['image'],
      mutable=['batch_stats'])

  xentropy = cross_entropy_loss(labels=batch['label'], logits=logits)
  weight_penalty_params = [x for x in jax.tree_leaves(params) if x.ndim > 1]
  weight_l2 = tree_util.tree_l2_norm(weight_penalty_params, squared=True)
  loss = xentropy + weight_decay * 0.5 * weight_l2

  new_batch_stats = new_mutable_variables['batch_stats']
  aux = LossFnAux(batch_stats=new_batch_stats, logits=logits)

  return loss, aux


@struct.dataclass
class TrainMetrics(clu_metrics.Collection):

  accuracy: clu_metrics.Accuracy
  learning_rate: clu_metrics.Average.from_output('learning_rate')
  loss: clu_metrics.Average.from_output('loss')
  xent: clu_metrics.Average.from_fun(cross_entropy_loss)


@struct.dataclass
class EvalMetrics(clu_metrics.Collection):

  accuracy: clu_metrics.Accuracy
  xent: clu_metrics.Average.from_fun(cross_entropy_loss)


def train_step(
    params: ArrayTree,
    state: ArrayTree,
    batch: Batch,
    metrics: Optional[TrainMetrics],
    learning_rate_fn: LearningRateFn,
    solver: jaxopt.OptaxSolver,
) -> Tuple[ArrayTree, ArrayTree, TrainMetrics]:
  """Performs a single training step."""
  # Retrieves Flax mutables from previous step, stored in `state.aux`.
  batch_stats = state.aux.batch_stats
  # Computes updated model parameters and optimizer state.
  params, state = solver.update(
      params=params,
      state=state,
      batch_stats=batch_stats,
      batch=batch,
  )
  # Computes train metrics for `batch`, re-using the auxiliary outputs from
  # `loss_fun` (e.g. logits) that are stored in `state.aux``.
  new_metrics = TrainMetrics.single_from_model_output(
      logits=state.aux.logits,
      labels=batch['label'],
      learning_rate=learning_rate_fn(state.iter_num),
      loss=state.value,  # xentropy + L2 regularization.
  )
  # Accumulates train metrics for current batch into history.
  if metrics is None:
    metrics = new_metrics
  else:
    metrics = metrics.merge(new_metrics)

  return params, state, metrics


def eval_step(
    params: ArrayTree,
    state: ArrayTree,
    batch: Batch,
    metrics: Optional[EvalMetrics],
    model: nn.Module,
) -> EvalMetrics:
  """Performs a single evaluation step."""
  # Retrieves Flax mutables from last train step, stored in `state.aux`.
  batch_stats = state.aux.batch_stats
  # Computes model outputs in inference-mode.
  variables = {'params': params, 'batch_stats': batch_stats}
  logits = model.apply(
      variables, batch['image'], train=False, mutable=False)
  # Computes eval metrics for `batch`.
  new_metrics = EvalMetrics.single_from_model_output(
      logits=logits, labels=batch['label'])
  # Accumulates eval metrics for current batch into history.
  if metrics is None:
    metrics = new_metrics
  else:
    metrics = metrics.merge(new_metrics)

  return metrics


def create_model(
    model_cls: Type[nn.Module],
    num_classes: int = NUM_CLASSES,
    half_precision: bool = True,
    **kwargs,
) -> nn.Module:
  """Creates FLAX model."""
  model_dtype = jnp.float32
  if half_precision:
    platform = jax.local_devices()[0].platform
    model_dtype = jnp.bfloat16 if platform == 'tpu' else jnp.float16
  return model_cls(num_classes=num_classes, dtype=model_dtype, **kwargs)


def initialize_model(
    key: PRNGKey,
    model: nn.Module,
    image_size: int = IMAGE_SIZE,
) -> Tuple[ArrayTree, ArrayTree]:
  """Initializes FLAX model, returning params and mutable variables."""
  input_shape = (1, image_size, image_size, 3)
  @jax.jit
  def init(*args):
    return model.init(*args)
  variables = init({'params': key}, jnp.ones(input_shape, model.dtype))

  return variables['params'], variables['batch_stats']


def create_learning_rate_fn(
    learning_rate: float,
    batch_size: int,
    steps_per_epoch: int,
    warmup_epochs: int,
    num_epochs: int,
) -> LearningRateFn:
  """Creates learning rate schedule."""
  base_learning_rate = learning_rate * batch_size / 256.
  warmup_fn = optax.linear_schedule(
      init_value=0., end_value=base_learning_rate,
      transition_steps=warmup_epochs * steps_per_epoch)
  cosine_epochs = max(num_epochs - warmup_epochs, 1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=base_learning_rate,
      decay_steps=cosine_epochs * steps_per_epoch)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[warmup_epochs * steps_per_epoch])
  return schedule_fn


def create_solver(
    learning_rate_fn: LearningRateFn,
    momentum: float,
    model: nn.Module,
) -> jaxopt.OptaxSolver:
  """Creates JAXopt solver."""
  opt = optax.sgd(learning_rate=learning_rate_fn,
                  momentum=momentum,
                  nesterov=True)
  fun = functools.partial(loss_fun, model=model)
  return jaxopt.OptaxSolver(opt=opt, fun=fun, has_aux=True)


def zeros_like_fun_output(
    fun: Callable,
    index: Optional[int] = None,
) -> Callable:
  """Replaces fun, outputting a pytree of zeroes with the original structure."""
  def wrapper(*args, **kwargs):
    pytree = jax.eval_shape(fun, *args, **kwargs)
    leaves, treedef = jax.tree_flatten(pytree)
    leaves = [jnp.zeros(shape=leaf.shape, dtype=leaf.dtype) for leaf in leaves]
    zeros_like_pytree = jax.tree_unflatten(treedef, leaves)
    return zeros_like_pytree if index is None else zeros_like_pytree[index]
  return wrapper


def initialize_solver(
    solver: jaxopt.OptaxSolver,
    init_params: ArrayTree,
    init_batch_stats: ArrayTree,
    first_batch: Batch,
    model: nn.Module,
) -> ArrayTree:
  """Initializes the state of jaxopt.OptaxSolver."""
  # "Default" JAXopt initial optimizer state.
  state = solver.init_state(init_params)

  # To prevent `train_step` from being compiled twice, we must ensure all its
  # input arguments have the same shape and dtype in all calls. To this end,
  # we will
  #   1) Initialize `state.aux` with a Pytree of the right shape and dtype.
  #   2) Ensure that `state.value` and `state.error` are strongly typed.
  zeros_like_loss_fun = zeros_like_fun_output(
      functools.partial(loss_fun, model=model))
  init_loss, init_aux = zeros_like_loss_fun(
      init_params, init_batch_stats, first_batch)
  init_aux = init_aux._replace(batch_stats=init_batch_stats)
  loss_dtype = init_loss.dtype
  return state._replace(
      value=jnp.asarray(jnp.inf, dtype=loss_dtype),
      error=jnp.asarray(jnp.inf, dtype=loss_dtype),
      aux=init_aux)


def initialize_metrics(
    init_params: ArrayTree,
    init_state: ArrayTree,
    first_batch: Batch,
    learning_rate_fn: LearningRateFn,
    solver: jaxopt.OptaxSolver,
    model: nn.Module,
) -> Tuple[TrainMetrics, EvalMetrics]:
  """Initializes train and eval metric accumulators."""
  # To prevent `train_step` and `eval_step` from being compiled twice, we must
  # ensure all its input arguments have the same shape and dtype in all calls.
  # To this end, we will initialize the `train_metrics` and `eval_metrics`
  # accumulators with Pytrees of the right shape and dtype but containing all
  # zeroes (including for the `count` field).
  zeros_like_train_step_fun = zeros_like_fun_output(
      functools.partial(
          train_step, learning_rate_fn=learning_rate_fn, solver=solver),
      index=-1)
  zeros_like_eval_step_fun = zeros_like_fun_output(
      functools.partial(eval_step, model=model))

  train_metrics_init = zeros_like_train_step_fun(
      init_params, init_state, first_batch, None)
  eval_metrics_init = zeros_like_eval_step_fun(
      init_params, init_state, first_batch, None)

  return train_metrics_init, eval_metrics_init


### Training loop.


def train_and_evaluate(workdir: str, seed: int = 0):
  """Execute model training and evaluation loop.

  Args:
    workdir: Directory where the tensorboard summaries are written to.
    seed: Initial seed for the PRNG.

  Returns:
    Final OptState.
  """
  # Sets PRNG seed (same in all hosts and devices).
  rng = random.PRNGKey(seed)

  # Computes local (i.e. per-device) batch size.
  if FLAGS.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = FLAGS.batch_size // jax.process_count()

  # Obtains iterators for training and evaluation datasets.
  ds_builder = tfds.builder(FLAGS.dataset)
  train_iter, eval_iter = create_input_iterators(ds_builder,
                                                 local_batch_size,
                                                 FLAGS.half_precision,
                                                 IMAGE_SIZE,
                                                 FLAGS.cache)

  # Computes period, in number of steps, for logging, evaluation and
  # checkpointing.
  num_train_examples = ds_builder.info.splits['train'].num_examples
  num_validation_examples = ds_builder.info.splits['validation'].num_examples
  steps_per_epoch = num_train_examples // FLAGS.batch_size
  num_train_steps = FLAGS.num_train_steps
  if num_train_steps == -1:
    num_train_steps = int(steps_per_epoch * FLAGS.num_epochs)
  steps_per_eval = FLAGS.steps_per_eval
  if steps_per_eval == -1:
    steps_per_eval = num_validation_examples // FLAGS.batch_size
  steps_per_checkpoint = steps_per_epoch * 10

  # Retrieves the first batch from the training iterator, to be used for
  # initialization purposes, and puts it back into the iterator.
  first_batch = next(train_iter)
  train_iter = itertools.chain([first_batch], train_iter)

  # Creates Flax model and initializes its parameters and mutable variables.
  model = create_model(MODELS[FLAGS.model], NUM_CLASSES, FLAGS.half_precision)
  params, batch_stats = initialize_model(rng, model, IMAGE_SIZE)

  # Creates learning rate schedule.
  learning_rate_fn = create_learning_rate_fn(FLAGS.learning_rate,
                                             FLAGS.batch_size,
                                             steps_per_epoch,
                                             FLAGS.warmup_epochs,
                                             FLAGS.num_epochs)
  # Creates a JAXopt optimizer and initializes its state with special care to
  # avoid recompilations.
  solver = create_solver(learning_rate_fn, FLAGS.momentum, model)
  state = initialize_solver(solver, params, batch_stats, first_batch, model)

  # Initializes accumulators for train and eval metrics and defines inline util
  # for replicating them across devices, also to prevent recompilations.
  train_metrics_init, eval_metrics_init = initialize_metrics(
      params, state, first_batch, learning_rate_fn, solver, model)
  replicate_metrics_init = pjit(
      lambda t: t, in_axis_resources=None, out_axis_resources=None)

  # Compiles data parallel train and eval steps using `jax.pjit`.
  p_train_step = pjit(
      functools.partial(
          train_step, learning_rate_fn=learning_rate_fn, solver=solver),
      in_axis_resources=(None, None, PartitionSpec('data'), None),
      out_axis_resources=None)
  p_eval_step = pjit(
      functools.partial(eval_step, model=model),
      in_axis_resources=(None, None, PartitionSpec('data'), None),
      out_axis_resources=None)

  # Instantiates metrics writer for logging.
  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() != 0)

  # Instantiates checkpointer manager and tries to restore state from `workdir`.
  ckpt = checkpoint.MultihostCheckpoint(
      os.path.join(workdir, 'checkpoints'), max_to_keep=3)
  params, state = ckpt.restore_or_initialize((params, state))
  step_offset = int(state.iter_num)

  # Sets up callbacks.
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer)
  hooks = [report_progress]
  if jax.process_index() == 0:
    hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]

  # Runs training loop.
  train_metrics = replicate_metrics_init(train_metrics_init)
  tic = time.time()
  for step, batch in zip(range(step_offset, num_train_steps), train_iter):
    params, state, train_metrics = p_train_step(
        params, state, batch, train_metrics)

    if step == step_offset:
      time_elapsed = time.time() - tic
      logging.info('p_train_step compilation done in %.2f s.', time_elapsed)

    for h in hooks:
      h(step)

    if (step + 1) % FLAGS.log_every_steps == 0 or step + 1 == num_train_steps:
      summary = train_metrics.compute()
      writer.write_scalars(
          step + 1, {f'train_{key}': val for key, val in summary.items()})
      # Resets accumulator for train metrics.
      train_metrics = replicate_metrics_init(train_metrics_init)

    if (step + 1) % steps_per_epoch == 0:
      with report_progress.timed('evaluation'):
        eval_metrics = replicate_metrics_init(eval_metrics_init)
        for _, eval_batch in zip(range(steps_per_eval), eval_iter):
          eval_metrics = p_eval_step(params, state, eval_batch, eval_metrics)

        summary = eval_metrics.compute()
        writer.write_scalars(
            step + 1, {f'eval_{key}': val for key, val in summary.items()})
        writer.flush()

    if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_train_steps:
      with report_progress.timed('checkpointing'):
        ckpt.save((params, state))

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  return params, state


### Entry point.


def main(_):
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  setup_data_parallel_mesh()
  logging.info('JAX PJIT mesh: %s', jax.experimental.maps.thread_resources.env)

  return train_and_evaluate(FLAGS.workdir)


if __name__ == '__main__':
  app.run(main)
