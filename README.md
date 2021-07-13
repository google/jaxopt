# JAXopt

[**Installation**](#installation)
| [**Examples**](https://github.com/google/jaxopt/tree/main/examples)
| [**References**](#ref)

Hardware accelerated (GPU/TPU), batchable and differentiable optimizers in JAX.

## Installation<a id="installation"></a>

JAXopt can be installed with pip directly from github, with the following
command:

```bash
$ pip install git+https://github.com/google/jaxopt
```

Alternatively, it can be be installed from sources with the following command:

```bash
$ python setup.py install
```

## References<a id="ref"></a>

Our implicit differentiation framework is described in this
[paper](https://arxiv.org/abs/2105.15183). To cite it:

```
@article{jaxopt_implicit_diff,
  title={Efficient and Modular Implicit Differentiation},
  author={Blondel, Mathieu and Berthet, Quentin and Cuturi, Marco and Frostig, Roy and Hoyer, Stephan and Llinares-L{\'o}pez, Felipe and Pedregosa, Fabian and Vert, Jean-Philippe},
  journal={arXiv preprint arXiv:2105.15183},
  year={2021}
}
```

## Disclaimer

JAXopt is an open source project maintained by a dedicated team in Google Research, but is not an official Google product.
