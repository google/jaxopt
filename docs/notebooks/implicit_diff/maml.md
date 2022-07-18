---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3.9.7 ('base')
  language: python
  name: python3
---

# Few-shot Adaptation with Model Agnostic Meta-Learning (MAML)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/jaxopt/blob/main/docs/notebooks/deep_learning/maml.ipynb)

by [*Fabian Pedregosa*](https://fa.bianp.net) and [*Paul Vicol*](https://www.paulvicol.com/), based on an [initial JAX example](https://github.com/ericjang/maml-jax/blob/master/maml.ipynb) by Eric Jang.

This notebook shows how to use Model Agnostic Meta-Learning (MAML) for few-shot adaptation on a simple regression task. This example appears in section 5.1 of [(Finn et al. 2017)](https://arxiv.org/pdf/1703.03400.pdf).


### References

https://blog.evjang.com/2019/02/maml-jax.html

```{code-cell} ipython3
import jax
from jax import numpy as jnp
from jax import random

import haiku as hk

import matplotlib.pyplot as plt
```

# Problem setup

We consider a multi-task problem, where each task involves regressing from the input to the output of a sine wave. The different tasks have different amplitude and phase of the sinusoid.

```{code-cell} ipython3
def generate_task(key, n_samples=100, min_phase=0, max_phase=jnp.pi, min_x=-5., max_x=5, min_amplitude=0.1, max_amplitude=0.5):
    """Generate a toy 1-D regression dataset."""
    amplitude = random.uniform(key) * (max_amplitude - min_amplitude) + min_amplitude
    
    key, _ = random.split(key)
    phase = random.uniform(key) * (max_phase - min_phase) + min_phase

    key, _ = random.split(key)
    x_train = random.uniform(key, shape=(n_samples,)) * (max_x - min_x) + min_x
    x_train = x_train.reshape(-1, 1)  # Reshape to feed into MLP later
    y_train = jnp.sin(phase * x_train) * amplitude
    return x_train, y_train, phase, amplitude
```

```{code-cell} ipython3
key = random.PRNGKey(0)
fig = plt.figure(figsize=(12, 6))

tasks = []
for task in range(3):

    key, subkey = random.split(key)
    x_train, y_train, phase, amplitude = generate_task(key, n_samples=100)
    # save the samples for later
    xs = jnp.linspace(-5, 5, 100)
    ys = jnp.sin(phase * xs) * amplitude
    tasks.append((x_train, y_train))

    plt.plot(xs, ys, linewidth=3, label=f'ground truth for task {task+1}')

plt.xlim((-5, 5))
plt.legend(loc='upper center', fontsize=14, bbox_to_anchor=(0.5, -0.1), fancybox=True, framealpha=0.3, ncol=3)
plt.show()
```

What we observe, are samples drawn these distributions (XXXX not clear what distributions means here)

```{code-cell} ipython3
fig = plt.figure(figsize=(12, 6))
for task, (x_train, y_train) in enumerate(tasks):
    plt.scatter(x_train, y_train, label=f"Observed samples for task {task+1}")
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
plt.legend(loc='upper center', fontsize=14, bbox_to_anchor=(0.5, -0.1), fancybox=True, framealpha=0.3, ncol=3)
plt.show()
```

The loss is the mean-squared error between the prediction and true value
$$
f(x) = TODO
$$

+++

The regressor is a neural network model with 2 hidden layers of size 40 with ReLU nonlinearities.

```{code-cell} ipython3
def net_fn(x):
  mlp = hk.nets.MLP([40, 40, 1])
  return mlp(x)

net = hk.without_apply_rng(hk.transform(net_fn))

key = random.PRNGKey(3)
x = jnp.ones([10, 1])
params = net.init(key, x)

@jax.jit
def inner_loss(params, x_train, y_train):
  preds = net.apply(params, x_train)
  mse_loss = jnp.mean((preds - y_train)**2)
  return mse_loss
```

```{code-cell} ipython3

```
