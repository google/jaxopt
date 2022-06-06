---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "CKB49u-HsfEb"}

# Dataset distillation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/jaxopt/blob/main/docs/notebooks/implicit_diff/dataset_distillation.ipynb)




Dataset distillation [Maclaurin et al. 2015](https://arxiv.org/pdf/1502.03492.pdf), [Wang et al. 2020](https://arxiv.org/pdf/1811.10959.pdf) aims to learn a small synthetic
training dataset such that a model trained on this learned data set achieves
small loss on the original training set.

+++ {"id": "T_1ezvj0ut0L"}

**Bi-level formulation**

Dataset distillation can be written formally as a bi-level problem, where in the
inner problem we estimate a logistic regression model $x^\star(\theta) \in
\mathbb{R}^{p \times k}$ trained on the distilled images $\theta \in
\mathbb{R}^{k \times p}$, while in the outer problem we want to minimize the
loss achieved by $x^\star(\theta)$ over the training set:

$$\underbrace{\min_{\theta \in \mathbb{R}^{k \times p}} f(x^\star(\theta), X_{\text{tr}}; y_{\text{tr}})}_{\text{outer problem}} ~\text{ subject to }~ x^\star(\theta) \in \underbrace{\text{argmin}_{x \in \mathbb{R}^{p \times k}} f(x, \theta; [k]) + \text{l2reg} \|x\|^2\,}_{\text{inner problem}}$$

where $f(W, X; y) := \ell(y, XW)$, and $\ell$ denotes the multiclass
logistic regression loss, $X_{\text{tr}}, y_{\text{tr}}$ are the samples and
target values in the train set, and $\text{l2reg} = 10^{-1}$ is a regularization
parameter that we found improved convergence.

```{code-cell}
:id: iQvA16DP8zhC

#@title Imports
%%capture
%pip install jaxopt flax
```

```{code-cell}
:id: 7lXrLlDi9FiC

import itertools
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt

import jax
from jax import numpy as jnp

from jaxopt import GradientDescent
from jaxopt import objective

jax.config.update("jax_platform_name", "cpu")
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 208
  referenced_widgets: [c334015f1ea947bb989e477ebb15a686, acb448d7153e4e46b54521f98afca562,
    844222f868204ef9a33eb47f76d87db8, f9e172ac2805409ea0e164622648d414, 1f67bb4c53124dbaa2e5ae2bfd0c298f,
    aa8282fe603043b18ae4c6ee03a0ae65, bd2824a64f5a4d1b9bee416077653861, c2a3133a84454a61904be05661aa6982,
    d5a08e82176b410d820189a4c653feb3, 59a557f2a61b4c7785f8e7f3cdf55928, a956598eafb84694a6b21bbc8a375fb3]
id: EQCtC92k9iXJ
outputId: b9b5ded7-3cac-4193-a2a6-c0cbcabd711e
---
#@title Load mnist
mnist_train, ds_info = tfds.load(name="mnist", split="train", with_info=True)
images_train = jnp.array([ex['image'].ravel() for ex in tfds.as_numpy(mnist_train)]) / 255.0
labels_train = jnp.array([ex['label'] for ex in tfds.as_numpy(mnist_train)])

mnist_test = tfds.load(name="mnist", split="test")
images_test = jnp.array([ex['image'].ravel() for ex in tfds.as_numpy(mnist_test)]) / 255.0
labels_test = jnp.array([ex['label'] for ex in tfds.as_numpy(mnist_test)])
```

```{code-cell}
:id: MAllQsyD_ppy

#@title Inner Problem

# these are the parameters of the logistic regression problem (inner problem)
params = jnp.ones((28 * 28, 10))

# amount of L2 reglarization of the inner problem. This helps both the
# convergence of the inner problem and the computation of the hypergradient
l2reg = 1e-1

inner_loss = objective.l2_multiclass_logreg
gd = GradientDescent(fun=inner_loss, tol=1e-3, maxiter=500)
```

```{code-cell}
:id: DfH4FyBRBDKH

#@title Outer Problem
rng = jax.random.PRNGKey(0)

# Distilled images (initialized at random, to be learned). These are the
# parameters of the outer problem
distilled_images = jax.random.normal(rng, (10, 28 * 28)) / (28 * 28)
distilled_labels = jnp.arange(10)
```

```{code-cell}
:id: pcw_H-EvBazg

# we now construct the outer loss and perform gradient descent on it
def outer_loss(img):
    # inner_sol is the solution to the inner problem, which computes the
    # model trained on the 10 images from distilled_images. This makes
    # the problem bi-level, as the objective depends itself on the solution
    # of an optimization problem  (inner_sol)
    inner_sol = gd.run(params, l2reg, (img, distilled_labels)).params
    return objective.l2_multiclass_logreg(
        inner_sol, 0, (images_train, labels_train))

gd_outer = GradientDescent(fun=outer_loss, tol=1e-3, maxiter=50)
```

```{code-cell}
:id: 2RY9bDWNCF_0

#@title Results
distilled_images = gd_outer.run(distilled_images).params
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 499
id: 1jQFS5i7BuIA
outputId: 5c87cd37-47e9-42ec-b0bd-30c41ae4af04
---
# Plot the learnt images
fig, axarr = plt.subplots(2, 5, figsize=(10 * 5, 2 * 10))
plt.suptitle("Distilled images", fontsize=40)

for k, (i, j) in enumerate(itertools.product(range(2), range(5))):
    img_i = distilled_images[k].reshape((28, 28))
    axarr[i, j].imshow(
        img_i / jnp.abs(img_i).max(), cmap=plt.cm.gray_r, vmin=-1, vmax=1)
    axarr[i, j].set_xticks(())
    axarr[i, j].set_yticks(())
plt.show()
```

```{code-cell}
:id: 0bM6pDvjYbCr


```
