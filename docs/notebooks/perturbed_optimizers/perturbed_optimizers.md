---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: base
  language: python
  name: python3
---

Copyright 2022 Google LLC

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

 https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

+++

# Perturbed optimizers

+++

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/jaxopt/blob/main/docs/notebooks/perturbed_optimizers/perturbed_optimizers.ipynb)

+++

We review in this notebook a universal method to transform any optimizer $y^*$ in a differentiable approximation $y_\varepsilon^*$, using pertutbations following the method of [Berthet et al. (2020)](https://arxiv.org/abs/2002.08676). JAXopt provides an implementation that we illustrate here on some examples.

Concretely, for an optimizer function $y^*$ defined by

$$y^*(\theta) = \mathop{\mathrm{arg\,max}}_{y\in \mathcal{C}} \langle y, \theta \rangle\, ,$$

we consider, for a random $Z$ drawn from a distribution with continuous positive distribution $\mu$

$$y_\varepsilon^*(\theta) = \mathbf{E}[\mathop{\mathrm{arg\,max}}_{y\in \mathcal{C}} \langle y, \theta + \varepsilon Z \rangle]$$

```{code-cell} ipython3
:id: 9WIWwRdSU51j

%%capture
%pip install jaxopt
```

```{code-cell} ipython3
:id: S6tLyyy9VCEw

import jax
import jax.numpy as jnp
import jaxopt
import time

from jaxopt import perturbations
```

+++ {"id": "EmYn_jNUFfw2"}

# Argmax one-hot

+++

We consider an optimizer, such as the following `argmax_one_hot` function. It transforms a real-valued vector into a binary vector with a 1 in the coefficient with largest magnitude and 0 elsewhere. It corresponds to $y^*$ for $\mathcal{C}$ being the unit simplex. We run it on an example input `values`.

+++ {"id": "84N-wAJ8GDK2"}

## One-hot function

```{code-cell} ipython3
:id: kMZnzhX4FjGj

def argmax_one_hot(x, axis=-1):
  return jax.nn.one_hot(jnp.argmax(x, axis=axis), x.shape[axis])
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: iynCk8734Wiz
outputId: 31e902b6-2dfa-4340-b11e-955b422bd313
---
values = jnp.array([-0.6, 1.9, -0.2, 1.1, -1.0])

one_hot_vec = argmax_one_hot(values)
print(one_hot_vec)
```

+++ {"id": "6rbNt-6zGb-J"}

## One-hot with pertubations

+++

Our implementation transforms the `argmax_one_hot` function into a perturbed one that we call `pert_one_hot`. In this case we use Gumbel noise for the perturbation.

```{code-cell} ipython3
:id: 7hQz6zuPwkpZ

N_SAMPLES = 100_000
SIGMA = 0.5
GUMBEL = perturbations.Gumbel()

rng = jax.random.PRNGKey(1)
pert_one_hot = perturbations.make_perturbed_argmax(argmax_fun=argmax_one_hot,
                                         num_samples=N_SAMPLES,
                                         sigma=SIGMA,
                                         noise=GUMBEL)
```

In this particular case, it is equal to the usual [softmax function](https://en.wikipedia.org/wiki/Softmax_function). This is not always true, in general there is no closed form for $y_\varepsilon^*$

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: f2gDpghJYZ33
outputId: adbe4b3c-4ec9-4f18-ea75-07619fd84cb3
---
rngs = jax.random.split(rng, 2)

rng = rngs[0]

pert_argmax = pert_one_hot(values, rng)
print(f'computation with {N_SAMPLES} samples, sigma = {SIGMA}')
print(f'perturbed argmax = {pert_argmax}')
jax.nn.softmax(values/SIGMA)
soft_max = jax.nn.softmax(values/SIGMA)
print(f'softmax = {soft_max}')
print(f'square norm of softmax = {jnp.linalg.norm(soft_max):.2e}')
print(f'square norm of difference = {jnp.linalg.norm(pert_argmax - soft_max):.2e}')
```

+++ {"id": "2U7rhtEAGpMV"}

## Gradients for one-hot with perturbations

+++

The perturbed optimizer $y_\varepsilon^*$ is differentiable, and its gradient can be computed with stochastic estimation automatically, using `jax.grad`.

We create a scalar loss `loss_simplex` of the perturbed optimizer $y^*_\varepsilon$

$$\ell_\text{simplex}(y_{\text{true}} = y_\varepsilon^*; y_{\text{true}})$$  

For `values` equal to a vector $\theta$, we can compute gradients of 

$$\ell(\theta) = \ell_\text{simplex}(y_\varepsilon^*(\theta); y_{\text{true}})$$
with respect to `values`, automatically.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 7H1LD4QhGtFI
outputId: fda1ce54-9e6c-4736-f3ff-05258129ee27
---
# Example loss function

def loss_simplex(values, rng):
  n = values.shape[0]
  v_true = jnp.arange(n) + 2
  y_true = v_true / jnp.sum(v_true)
  y_pred = pert_one_hot(values, rng)
  return jnp.sum((y_true - y_pred) ** 2)

loss_simplex(values, rngs[1])
```

We can compute the gradient of $\ell$ directly

$$\nabla_\theta \ell(\theta) = \partial_\theta y^*_\varepsilon(\theta) \cdot \nabla_1 \ell_{\text{simplex}}(y^*_\varepsilon(\theta); y_{\text{true}})$$

The computation of the jacobian $\partial_\theta y^*_\varepsilon(\theta)$ is implemented automatically, using an estimation method given by [Berthet et al. (2020)](https://arxiv.org/abs/2002.08676), [Prop. 3.1].

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: tjQatCE3GtFJ
outputId: b2454c6b-e3a9-4ed7-ca31-c49a24ce4594
---
# Gradient of the loss w.r.t input values

gradient = jax.grad(loss_simplex)(values, rngs[1])
print(gradient)
```

We illustrate the use of this method by running 200 steps of gradient descent on $\theta_t$ so that it minimizes this loss.

```{code-cell} ipython3
:id: MuNE2RX0GtFJ

# Doing 200 steps of gradient descent on the values to have the desired ranks

steps = 200
values_t = values
eta = 0.5

grad_func = jax.jit(jax.grad(loss_simplex))

for t in range(steps):
  rngs = jax.random.split(rngs[1], 2)
  values_t = values_t - eta * grad_func(values_t, rngs[1])
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 29TWHiH0GtFJ
outputId: e53faab6-ebae-4658-d0ce-48f2b06b6a8a
---
rngs = jax.random.split(rngs[1], 2)

n = values.shape[0]
v_true = jnp.arange(n) + 2
y_true = v_true / jnp.sum(v_true)

print(f'initial values = {values}')
print(f'initial one-hot = {argmax_one_hot(values)}')
print(f'initial diff. one-hot = {pert_one_hot(values, rngs[0])}')
print()
print(f'values after GD = {values_t}')
print(f'ranks after GD = {argmax_one_hot(values_t)}')
print(f'diff. one-hot after GD = {pert_one_hot(values_t, rngs[1])}')
print(f'target diff. one-hot = {y_true}')
```

+++ {"id": "4Vyh_a1bZT-s"}

# Differentiable ranking

+++ {"id": "QmVAjbJxFzUA"}

## Ranking function

+++

We consider an optimizer, such as the following `ranking` function. It transforms a real-valued vector of size $n$ into a vector with coefficients being a permutation of $\{0,\ldots, n-1\}$ corresponding to the order of the coefficients of the original vector. It corresponds to $y^*$ for $\mathcal{C}$ being the permutahedron. We run it on an example input `values`.

```{code-cell} ipython3
:id: -NKbR6TlZUTG

# Function outputting a vector of ranks

def ranking(values):
  return jnp.argsort(jnp.argsort(values))
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: iU69uMAoZncY
outputId: 70d4a0ec-c335-48c1-d04d-b5fa32e951c4
---
# Example on random values

n = 6

rng = jax.random.PRNGKey(0)
values = jax.random.normal(rng, (n,))

print(f'values = {values}')
print(f'ranking = {ranking(values)}')
```

+++ {"id": "5j1Vgfz_bb9u"}

## Ranking with perturbations

+++

As above, our implementation transforms this function into a perturbed one that we call `pert_ranking`. In this case we use Gumbel noise for the perturbation.

```{code-cell} ipython3
:id: Equ3_gDPbf5n

N_SAMPLES = 100
SIGMA = 0.2
GUMBEL = perturbations.Gumbel()

pert_ranking = perturbations.make_perturbed_argmax(ranking,
                                                   num_samples=N_SAMPLES,
                                                   sigma=SIGMA,
                                                   noise=GUMBEL)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: vMj-Dnudby_a
outputId: 827458ad-de4b-4b85-be72-0373c2a4b7f8
---
# Expectation of the perturbed ranks on these values

rngs = jax.random.split(rng, 2)

diff_ranks = pert_ranking(values, rngs[0])
print(f'values = {values}')

print(f'diff_ranks = {diff_ranks}')
```

+++ {"id": "aH6Ew85koQvU"}

## Gradients for ranking with perturbations

+++

As above, the perturbed optimizer $y_\varepsilon^*$ is differentiable, and its gradient can be computed with stochastic estimation automatically, using `jax.grad`.

We showcase this on a loss of $y_\varepsilon(\theta)$ that can be directly differentiated w.r.t. the `values` equal to $\theta$.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: O-T8y6N8cHzF
outputId: ad780191-cac8-40a9-ecfc-1b3442732aa5
---
# Example loss function

def loss_example(values, rng):
  n = values.shape[0]
  y_true = ranking(jnp.arange(n))
  y_pred = pert_ranking(values, rng)
  return jnp.sum((y_true - y_pred) ** 2)

print(loss_example(values, rngs[1]))
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: v7nzNwP-e68q
outputId: 1fc9ff56-c223-49f1-9acc-00a959e44c26
---
# Gradient of the objective w.r.t input values

gradient = jax.grad(loss_example)(values, rngs[1])
print(gradient)
```

As above, we showcase this example on gradient descent to minimize this loss.

```{code-cell} ipython3
:id: 0UObBP3QfCqq

steps = 20
values_t = values
eta = 0.1

grad_func = jax.jit(jax.grad(loss_example))

for t in range(steps):
  rngs = jax.random.split(rngs[1], 2)
  values_t = values_t - eta * grad_func(values_t, rngs[1])
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: p4iNxMoQmZRa
outputId: 16f75357-192d-4b5d-8467-20e5ab639e9b
---
rngs = jax.random.split(rngs[1], 2)

y_true = ranking(jnp.arange(n))

print(f'initial values = {values}')
print(f'initial ranks = {ranking(values)}')
print(f'initial diff. ranks = {pert_ranking(values, rngs[0])}')
print()
print(f'values after GD = {values_t}')
print(f'ranks after GD = {ranking(values_t)}')
print(f'diff. ranks after GD = {pert_ranking(values_t, rngs[1])}')
print(f'target diff. ranks = {y_true}')
```
