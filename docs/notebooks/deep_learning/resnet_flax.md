---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.1
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "KjYlO80JL3j1"}

Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

+++ {"id": "uJHywE_oL3j2"}

# ResNet on CIFAR with Flax and JAXopt.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/jaxopt/blob/main/docs/notebooks/deep_learning/resnet_flax.ipynb)

In this notebook, we'll go through training a deep residual network with jaxopt.

```{code-cell} ipython3
:id: gzQc20SyL3j2

%%capture
%pip install -U jaxopt flax tqdm
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: VaYIiCnjL3j3
outputId: 89f40496-5760-4d57-bc35-faec50f37934
---
from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Sequence, Dict

from flax import linen as nn

import jax
import jax.numpy as jnp
from tqdm.notebook import trange
import numpy as np

import optax
import tensorflow_datasets as tfds
import tensorflow as tf

from matplotlib import pyplot as plt

import jaxopt
from jaxopt import OptaxSolver

# hide the GPU from tensorflow, otherwise it might
# reserve memory on it
tf.config.experimental.set_visible_devices([], "GPU")

# Show on which platform JAX is running.
print("JAX running on", jax.devices()[0].platform.upper())
```

```{code-cell} ipython3
:id: jve2h810L3j3

# @markdown Total number of epochs to train for:
MAX_EPOCHS = 50  # @param{type:"integer"}
# @markdown Number of samples in each batch:
BATCH_SIZE = 128  # @param{type:"integer"}
# @markdown The initial learning rate for the optimizer:
PEAK_LR = 0.12  # @param{type:"number"}
# @markdown The model architecture for the neural network. Can be one of `'resnet1'`, `'resnet18'`, `'resnet34'`, `'resnet50'`, `'resnet101'`, `'resnet152'` and `'resnet200'`:
MODEL = "resnet18"  # @param{type:"string"}
# @markdown The dataset to use. Could be either `'cifar10'` or `'cifar100'`:
DATASET = "cifar10"  # @param{type:"string"}
# @markdown The amount of L2 regularization (aka weight decay) to use:
L2_REG = 1e-4  # @param{type:"number"}
```

+++ {"id": "iLGeV4y4DBkL"}

CIFAR10 and CIFAR100 are composed of 32x32 images with 3 channels (RGB). We'll now load the dataset using `tensorflow_datasets` and display a few of the first samples.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 488
  referenced_widgets: [09956b1819974645ba506df14386dd04, 96a39150b0ae4a86b5aedef5761b5ee8,
    80be9ad750f54c2486a9760ed2d1d4f3, 986285aef29e47c99b0643891727921b, b5f9df25a2f34e8a85dabfb1b0b30183,
    a474be4a23b44ddcadd047da85577a2d, af5c95fb0629428c964ce4a40b876ed6, 090eb94704454037a59a4cdd721889d8,
    6e0d75ecd88c42048079ef5723456e4c, ef1e6b8bf90649358527b10e3ba9ca3a, 00201685e7f541918c051d76f40964f4,
    1791cd84bfc44aa3a8d943337278d4bd, ddabf989cbc94d5dafc777c8f952c6a4, ceb4a19fce6a4528990b42ebc474f791,
    58ae836d9cac4d51896e28151278a9e5, 93304448db0f4fcc8abdfd0cfda31fa5, 0f8e5a923446428ba008973135b6df55,
    1401a343a76a470795611a0c6f18a40f, 702b6b798b9749cabaaeb6202499b534, b5768c3f473b4ae69d749ede43334378,
    b8e2e9a2723c491099a601fa674241f9, dc68565e10bc46fcbc3dfadf81cd4c75, 23335679ab4347f3b693d6662e5fb708,
    9b6387eb0671431693e07e297370dd00, b67f95a24baf42dbb5e51716b937e9ec, 74fb8d844bc64f0397d2db8ee2075b19,
    c5ff563b02c042fa955b41ce346792ab, aac305a3fca5417fab820e9b93cbf445, 0d1530ee39e6413bbf5f47a3b6c5278b,
    39b324873d45443e8a724eb3176165e9, 08531346ae104cd0aa15b8ed6a654efa, 4dc9b8944aec471da9231f84b3bb366b,
    6752246dcf4e4fe9816b9f4548ae79a2, 4de23fa22dac4a5eb729f6915bc0cd69, aae5733cd67b45bc9239db1f7dfed0bb,
    423d5802148f4aa2b0760cbfb04789c0, 8741dee0ef3742bca3e2bd66cde98f09, 5a72fcbe581747c6a121a1dbf2af62bc,
    7f2b49cfc63c4cc19405382a4b256b55, 968010f7077249a7a85e92d343573d06, 9c23a4ca891f4d08a0a083b19ba7776e,
    612e6481165043cab4f629edf3c3a4cb, 8b5c230a90ab48638cf913a140522d91, 390210b0ef144384a2b94f133a1b6687,
    05d093a1cbfa4eb999c5d23885b2d615, fa10986a171849c0b8cb74f1ccbfa68a, 792b932d7b04463e84d0aee1c99e603a,
    bd9ffd863c23413ab351ed6174242eae, 5b45e57a0c8e4e35b935351bc3a0c5dc, d037a4261eaf4542ac49302428ce0c6c,
    5128cb1c04f14a8e9e2db0a410c1878f, ae1af2c8c47341ea8fc6f3ed9f2d6d2b, a77cd9db9f99490b8a9a69c716b8ca39,
    3e684fa705374ddead7b607bc66d3814, b8a217623b49405eb13db064185a0c9b, 845eeb9bed57435f98ae0df95a434da9,
    ad3fbebde4a74a3d963115c2d0d5bccb, 8e6d47f7f5cc4908b40d7b5352d0354e, d255b8f8ed1d429381f5a4c9b4a5c700,
    5fe05a36f0164c71a61edeb240ac3799, 2eec029ddc0843aebeb853c84ed2ab1c, c3c0aa276c4446c9a1e36a9c546e489b,
    0e73c219dd05456c93739ea8ba905107, ede2a5135ff441d0bb38d6932e64f3ab, 1769899af8ae4d90ab1199a95cb76876,
    ef126d511f4f4b7d99a3bc8b604e305b, bbe2d0e28d08428e8e9c70e43fe7f511, 24d834ddc4014f84897d53e5dda5e56e,
    a6ca2f14d2d84bfab33d9dd90d4b023b, ef730f3f501445f38648d83b4040799f, 9e58921b1aa74765a947b9f4c6183c81,
    b8962abeb4c34fabbd2eb86aa1ed9946, f3256cf26c9b4e4ba766b50c013feea6, b0c050a3155243a9ba804892fd50f7d4,
    3a0a635c37954be0b7cff67f5e947dad, 2b81bccaf7d14bbf9e7b0179effbf92f, 3995fc3fd3dc453fbd8e99ae2a188137,
    58405c29521f4715b06d01c7949060cb, ca7a731a803f4316b55ad18b1fb83aa1, dc9561af72514d809abdb4e52d31256e,
    b42fb7e79f8c48dfbf6e4ec0bd884500, 113da5ee49474c0fb6c9312f17fb6bf0, 40b9b97b36a94adab8906403fd69ab04,
    53a34443359344189ac599e1e89307eb, 06e26266f3204d53a369623e07bfd640, f9da3e6d4d674fd68df2f8289984f928,
    555aa84109104a2eba70aa2a31ed3306, ca8f12d5e35f4ffbba02375a836d5cec]
id: zynvtk4wDBkL
outputId: e7ee8094-cc59-4c79-a99e-9270748647e8
---
(train_loader, test_loader), info = tfds.load(
    DATASET, split=["train", "test"], as_supervised=True, with_info=True
)
NUM_CLASSES = info.features["label"].num_classes
IMG_SIZE = info.features["image"].shape


def plot_sample_images(loader):
  loader_iter = iter(loader)
  _, axes = plt.subplots(nrows=4, ncols=5, figsize=(6, 4))
  for i in range(4):
    for j in range(5):
      k = i * 4 + j
      image, label = next(loader_iter)
      axes[i, j].imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
      axes[i, j].set_axis_off()
      axes[i, j].set_title(info.features["label"].names[label], fontsize=10, y=0.9)


plot_sample_images(train_loader)
```

+++ {"id": "XlSCdO8yDBkM"}

The accuracy of the model can be improved significantly through data augmentation. That is, instead of training on the above images, we'll generate random modifications of the images and train on those. This is done by using the `transform` argument of `tfds.load` to apply a random crop, random horizontal flip, and random color jittering.

In the next cell we show an instance of these transformations on the above images.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 357
id: dVkDlShoDBkM
outputId: e4036fb8-2260-4ee9-e4b4-bfdc369667e6
---
def augment(image, label):
  """Data augmentation for CIFAR10."""
  image = tf.image.resize_with_crop_or_pad(image, 40, 40)
  image = tf.image.random_crop(image, [32, 32, 3])
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_brightness(image, max_delta=0.2)
  image = tf.image.random_contrast(image, 0.8, 1.2)
  image = tf.image.random_saturation(image, 0.8, 1.2)
  return image, label


train_loader_augmented = train_loader.map(augment)
plot_sample_images(train_loader_augmented)
```

+++ {"id": "ITnYDuiIDBkM"}

We now shuffle the data in the train set and create batches of size `'BATCH_SIZE'` for both train and test set

```{code-cell} ipython3
:id: TSjMCbukDBkN

train_loader_batched = train_loader_augmented.shuffle(
    buffer_size=10_000, reshuffle_each_iteration=True
).batch(BATCH_SIZE, drop_remainder=True)

test_loader_batched = test_loader.batch(BATCH_SIZE, drop_remainder=True)
```

+++ {"id": "CrSSsvpUDBkN"}

With the data ready, we can now define the model. Below we define the ResNet architecture that we'll later instantiate. We define different variants of the architecture with different sizes and depths (`'ResNet1'`, `'ResNet18'`, `'ResNet34'`, `'ResNet50'` and `'ResNet101'`).

The following code is based on the [Flax imagenet example](https://github.com/google/flax/blob/main/examples/imagenet/models.py). The only difference with that code is the addition of the keyword argument `initial_conv_config` to the `ResNet` class, which allows to change the configuration of the initial convolutional layer of the network. This is important to get state of the art accuracy on CIFAR10, as the default kernel size (7, 7) is too big for the small 32x32 images of CIFAR10.

```{code-cell} ipython3
:id: P7Z3Vex8QuGz

ModuleDef = Any


class ResNetBlock(nn.Module):
  """ResNet block."""

  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(
      self,
      x,
  ):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)
    y = self.norm(scale_init=nn.initializers.zeros_init())(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(
          residual
      )
      residual = self.norm(name="norm_proj")(residual)

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
    y = self.norm(scale_init=nn.initializers.zeros_init())(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters * 4, (1, 1), self.strides, name="conv_proj")(
          residual
      )
      residual = self.norm(name="norm_proj")(residual)

    return self.act(residual + y)


class ResNet(nn.Module):
  """ResNetV1."""

  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  num_classes: int
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  conv: ModuleDef = nn.Conv
  initial_conv_config: Optional[Dict[str, Any]] = None

  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype,
    )

    initial_conv_config = dict(self.initial_conv_config)
    initial_conv_config.setdefault("kernel_size", 7)
    initial_conv_config.setdefault("stride", 2)
    initial_conv_config.setdefault("with_bias", False)
    initial_conv_config.setdefault("padding", "SAME")
    initial_conv_config.setdefault("name", "initial_conv")

    x = conv(self.num_filters, **self.initial_conv_config)(x)
    x = norm(name="bn_init")(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(
            self.num_filters * 2**i,
            strides=strides,
            conv=conv,
            norm=norm,
            act=self.act,
        )(x)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    return x


ResNet1 = partial(ResNet, stage_sizes=[1], block_cls=ResNetBlock)
ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock)
ResNet101 = partial(ResNet, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock)
ResNet152 = partial(ResNet, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock)
ResNet200 = partial(ResNet, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock)
```

+++ {"id": "Rkxuc-fhQuGz"}

As mentioned before, the default kernel size of (7, 7) in the first convolutional layer is too big for the small image sizes of CIFAR10. Because of this, we will instantiate the network with a custom kernel size of (3, 3) and strides of 1. Note that this is specific to CIFAR10, and datasets with bigger images such as ImageNet or CelebA should instead use the default values.

```{code-cell} ipython3
:id: d69RQ4UtQuGz

initial_conv_config = {
    "kernel_size": (3, 3),
    "strides": 1,
    "padding": "SAME",
}
# Set up model.
if MODEL == "resnet1":
  net = ResNet1(num_classes=NUM_CLASSES, initial_conv_config=initial_conv_config)
elif MODEL == "resnet18":
  net = ResNet18(num_classes=NUM_CLASSES, initial_conv_config=initial_conv_config)
elif MODEL == "resnet34":
  net = ResNet34(num_classes=NUM_CLASSES, initial_conv_config=initial_conv_config)
elif MODEL == "resnet50":
  net = ResNet50(num_classes=NUM_CLASSES, initial_conv_config=initial_conv_config)
elif MODEL == "resnet101":
  net = ResNet101(num_classes=NUM_CLASSES, initial_conv_config=initial_conv_config)
elif MODEL == "resnet152":
  net = ResNet152(num_classes=NUM_CLASSES, initial_conv_config=initial_conv_config)
elif MODEL == "resnet200":
  net = ResNet200(num_classes=NUM_CLASSES, initial_conv_config=initial_conv_config)
else:
  raise ValueError(f"Unknown model {MODEL}.")
```

+++ {"id": "BZPW1DnOL3j4"}

We'll now load our train and test dataset and plot a few of the training images.

```{code-cell} ipython3
:id: _kbXJT07L3j5

def _predict(params, bn_params, inputs, train=False):
  # Predict logits from inputs and parameters
  all_params = {"params": params, "batch_stats": bn_params}

  def train_fn(inputs):
    logits, net_state = net.apply(
        all_params, inputs, train=True, mutable=["batch_stats"]
    )
    return logits, net_state

  def eval_fn(inputs):
    logits = net.apply(all_params, inputs, train=False, mutable=False)
    net_state = {"batch_stats": bn_params}
    return logits, net_state

  logits, net_state = jax.lax.cond(
      train, lambda x: train_fn(x), lambda x: eval_fn(x), inputs
  )
  return logits, net_state


logistic_loss = jax.vmap(jaxopt.loss.multiclass_logistic_loss)


@jax.jit
def loss_accuracy(params, bn_params, data, train=True):
  """Compute loss and accuracy over a mini-batch.

  Args:
    params: parameters of the model.
    bn_params: batch normalization parameters.
    data: tuple of (inputs, labels).
    train: whether to use train mode or eval mode.
  """
  inputs, labels = data
  logits, net_state = _predict(params, bn_params, inputs, train=train)
  mean_loss = jnp.mean(logistic_loss(labels, logits))
  accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
  l2_params = jax.tree_util.tree_leaves(params)
  # make sure batchnorm parameters and biases are not regularized
  weight_l2 = sum(jnp.sum(x**2) for x in l2_params if x.ndim > 1)
  loss = mean_loss + 0.5 * L2_REG * weight_l2
  return loss, {"accuracy": accuracy, "batch_stats": net_state["batch_stats"]}
```

+++ {"id": "5T1-w-FhQuG0"}

We'll use a learning rate that initially increases up to the peak value of `'PEAK_LR'` and then decreases. Such learning rate is implemented (for instance) in optax's [linear_onecycle_schedule](https://optax.readthedocs.io/en/latest/api.html#optax.linear_onecycle_schedule).

Below we create the learning rate and plot its value across iterations.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 490
id: hPOD3TxWQuG0
outputId: a831171a-1071-4f5e-e2f7-f52a0caae80b
---
iter_per_epoch_train = info.splits["train"].num_examples // BATCH_SIZE
lr_schedule = optax.linear_onecycle_schedule(MAX_EPOCHS * iter_per_epoch_train, PEAK_LR)

# plot the learning rate schedule
iterate_subsample = np.linspace(0, MAX_EPOCHS * iter_per_epoch_train, 100)
plt.title("Learning rate schedule", fontsize=20)
plt.plot(
    np.linspace(0, MAX_EPOCHS, len(iterate_subsample)),
    [lr_schedule(i) for i in iterate_subsample],
    lw=3,
)
plt.xlabel("Epochs", fontsize=18)
plt.ylabel("Learning rate", fontsize=18)
plt.grid()
plt.xlim((0, MAX_EPOCHS))
plt.ylim((0, PEAK_LR))
plt.show()
```

+++ {"id": "IVqnrZEEICx3"}

TODO: explain what we're doing in the next two cells

In the next two cells we'll initialize the variables and states. We also define a convenience function `dataset_stats` that we'll call once per epoch to collect the loss and accuracy of our solver over the test set.

```{code-cell} ipython3
:id: xLTQpLg1L3j5

opt = optax.sgd(lr_schedule, momentum=0.9, nesterov=False)

# We need has_aux=True because loss_fun returns batch_stats.
solver = OptaxSolver(
    opt=opt, fun=loss_accuracy, maxiter=MAX_EPOCHS * len(train_loader), has_aux=True
)

# Initialize parameters.
rng = jax.random.PRNGKey(0)
# Dummy data to initialize parameters and solver state.
dummy_data = jnp.ones((1,) + IMG_SIZE, jnp.float32)
dummy_targets = jnp.ones(1, int)
variables = net.init({"params": rng}, dummy_data)

var_params, var_batch_stats = variables["params"], variables["batch_stats"]
```

```{code-cell} ipython3
:id: Fu9Lu_dC0u3I

# Define parameter update function.
solver_state = solver.init_state(
    var_params, var_batch_stats, (dummy_data, dummy_targets)
)


def dataset_stats(params, var_batch_stats, data_loader):
  """Compute loss and accuracy over the dataset `data_loader` for `max_iter` items."""
  all_accuracy = []
  all_loss = []
  for batch in data_loader.as_numpy_iterator():
    loss, aux = loss_accuracy(params, var_batch_stats, batch, train=False)
    all_accuracy.append(aux["accuracy"])
    all_loss.append(loss)
  return {"loss": np.mean(all_loss), "accuracy": np.mean(all_accuracy)}
```

+++ {"id": "49L9gffyDBkO"}

Finally, we do the actual training. The next cell performs `'MAX_EPOCHS'` epochs of training. Within each epoch we iterate over the batched loader `train_loader_batched`, and once per epoch we also compute the test set accuracy and loss.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 49
  referenced_widgets: [faed8cddd8254d8599b8740a9e579167, 5e23953290a846b29f7b710eaced3a77,
    421700db6403447ba092c23430109761, 23a1dd842e86405aa91854830cb6def2, 2777c686d3d049c2b90ea5044dd5129b,
    056edd86e1ed4da593809fcda6ec5d93, 760554c73c334356804b34d8a8116cdf, 1b3e1beeb92b4fccaaf8de44bb9859d9,
    0e5403fff0ac4cd18cb736faf131cd7c, f73b16badda74f5c81a432b83b3a31aa, 3a6ed213dbee492fbafbe81aaa83ab3f]
id: loz4M9I5SLRu
outputId: 7340b07d-7788-4504-9dbb-c1f61dfc1118
---
train_accuracy = []
train_loss = []
# Compute test set accuracy at initialization
test_stats = dataset_stats(var_params, var_batch_stats, test_loader_batched)
test_accuracy = [test_stats["accuracy"]]
test_loss = [test_stats["loss"]]

# Training loop.
pbar = trange(MAX_EPOCHS, desc="Training progress", leave=True, unit="epochs")
for epoch in pbar:
  train_accuracy_epoch = []
  train_loss_epoch = []

  for batch in train_loader_batched.as_numpy_iterator():
    var_params, solver_state = solver.update(
        params=var_params,
        state=solver_state,
        bn_params=var_batch_stats,
        data=batch,
    )
    var_batch_stats = solver_state.aux["batch_stats"]
    train_accuracy_epoch.append(solver_state.aux["accuracy"])
    train_loss_epoch.append(solver_state.value)

  # once per epoch, make a pass over the test set to compute accuracy
  test_stats = dataset_stats(var_params, var_batch_stats, test_loader_batched)
  test_accuracy.append(test_stats["accuracy"])
  test_loss.append(test_stats["loss"])
  train_accuracy.append(np.mean(train_accuracy_epoch))
  train_loss.append(np.mean(train_loss_epoch))

  # update progress bar
  pbar.set_postfix({
      "test set accuracy": test_accuracy[-1],
      "train set accuracy": np.mean(train_accuracy_epoch),
  })
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 669
id: HV548_CRQuG0
outputId: d7ebfcef-ab78-42f0-c092-fae52b58fe3e
---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

plt.suptitle(f"{MODEL} on {DATASET}", fontsize=20)

ax1.plot(test_accuracy, lw=3, marker="s", markevery=5, markersize=10, label="test set")
ax1.plot(
    train_accuracy,
    lw=3,
    marker="^",
    markevery=5,
    markersize=10,
    label="train set (stochastic estimate)",
)
ax1.set_ylabel("Accuracy", fontsize=20)
ax1.grid()
ax1.set_xlabel("Epochs", fontsize=20)
ax1.set_xlim((0, MAX_EPOCHS))
ax1.set_ylim((0, 1))

ax2.plot(test_loss, lw=3, marker="s", markevery=5, markersize=10, label="test set")
ax2.plot(
    train_loss,
    lw=3,
    marker="^",
    markevery=5,
    markersize=10,
    label="train set (stochastic estimate)",
)
ax2.set_ylabel("Loss", fontsize=20)
ax2.grid()
ax2.set_xlabel("Epochs", fontsize=20)
ax2.set_xlim((0, MAX_EPOCHS))

# set legend at the bottom of the plot
ax1.legend(frameon=False, fontsize=20, ncol=2, loc=2, bbox_to_anchor=(0.3, -0.1))

ax2.set_yscale("log")

plt.show()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: dy2EnSiCL3j5
outputId: 32f25bab-f840-4daf-b436-7dd4b7f77a83
---
print("Final test set accuracy:", test_accuracy[-1])
```

```{code-cell} ipython3
:id: XQGDaL5WQuG1


```
