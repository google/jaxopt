# JAXopt

[**Status**](#status)
| [**Installation**](#installation)
| [**Documentation**](https://jaxopt.github.io)
| [**Examples**](https://github.com/google/jaxopt/tree/main/examples)
| [**Cite us**](#citeus)

Hardware accelerated, batchable and differentiable optimizers in
[JAX](https://github.com/google/jax).

- **Hardware accelerated:** our implementations run on GPU and TPU, in addition
  to CPU.
- **Batchable:** multiple instances of the same optimization problem can be
  automatically vectorized using JAX's vmap.
- **Differentiable:** optimization problem solutions can be differentiated with
  respect to their inputs either implicitly or via autodiff of unrolled
  algorithm iterations.

## Status<a id="status"></a>

JAXopt is no longer maintained nor developed. Alternatives may be found on the
JAX [website](https://docs.jax.dev/en/latest/). Some of its features (like
losses, projections, lbfgs optimizer) have been ported into
[optax](https://github.com/google-deepmind/optax). We are sincerely grateful for
all the community contributions the project has garnered over the years.

## Installation<a id="installation"></a>

To install the latest release of JAXopt, use the following command:

```bash
$ pip install jaxopt
```

To install the **development** version, use the following command instead:

```bash
$ pip install git+https://github.com/google/jaxopt
```

Alternatively, it can be installed from sources with the following command:

```bash
$ python setup.py install
```

## Cite us<a id="citeus"></a>

Our implicit differentiation framework is described in this
[paper](https://arxiv.org/abs/2105.15183). To cite it:

```
@inproceedings{jaxopt_implicit_diff,
author = {Blondel, Mathieu and Berthet, Quentin and Cuturi, Marco and Frostig, Roy and Hoyer, Stephan and Llinares-L\'{o}pez, Felipe and Pedregosa, Fabian and Vert, Jean-Philippe},
title = {Efficient and modular implicit differentiation},
year = {2022},
isbn = {9781713871088},
publisher = {Curran Associates Inc.},
address = {Red Hook, NY, USA},
abstract = {Automatic differentiation (autodiff) has revolutionized machine learning. It allows to express complex computations by composing elementary ones in creative ways and removes the burden of computing their derivatives by hand. More recently, differentiation of optimization problem solutions has attracted widespread attention with applications such as optimization layers, and in bi-level problems such as hyper-parameter optimization and meta-learning. However, so far, implicit differentiation remained difficult to use for practitioners, as it often required case-by-case tedious mathematical derivations and implementations. In this paper, we propose automatic implicit differentiation, an efficient and modular approach for implicit differentiation of optimization problems. In our approach, the user defines directly in Python a function F capturing the optimality conditions of the problem to be differentiated. Once this is done, we leverage autodiff of F and the implicit function theorem to automatically differentiate the optimization problem. Our approach thus combines the benefits of implicit differentiation and autodiff. It is efficient as it can be added on top of any state-of-the-art solver and modular as the optimality condition specification is decoupled from the implicit differentiation mechanism. We show that seemingly simple principles allow to recover many existing implicit differentiation methods and create new ones easily. We demonstrate the ease of formulating and solving bi-level optimization problems using our framework. We also showcase an application to the sensitivity analysis of molecular dynamics.},
booktitle = {Proceedings of the 36th International Conference on Neural Information Processing Systems},
articleno = {378},
numpages = {13},
location = {New Orleans, LA, USA},
series = {NIPS '22}
}
```

## Disclaimer

JAXopt was an open source project maintained by a dedicated team in Google
Research. It is not an official Google product.

