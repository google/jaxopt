# JAXopt

[**Installation**](#installation)
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
@article{jaxopt_implicit_diff,
  title={Efficient and Modular Implicit Differentiation},
  author={Blondel, Mathieu and Berthet, Quentin and Cuturi, Marco and Frostig, Roy 
    and Hoyer, Stephan and Llinares-L{\'o}pez, Felipe and Pedregosa, Fabian 
    and Vert, Jean-Philippe},
  journal={arXiv preprint arXiv:2105.15183},
  year={2021}
}
```

## Disclaimer

JAXopt is an open source project maintained by a dedicated team in Google Research, but is not an official Google product.
