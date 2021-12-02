:github_url: https://github.com/google/jaxopt/tree/master/docs

JAXopt Documentation
====================

Hardware accelerated, batchable and differentiable optimizers in JAX.

- **Hardware accelerated:** our implementations run on GPU and TPU, in addition
  to CPU.
- **Batchable:** multiple instances of the same optimization problem can be
  automatically vectorized using JAX's vmap.
- **Differentiable:** optimization problem solutions can be differentiated with
  respect to their inputs either implicitly or via autodiff of unrolled
  algorithm iterations.

Installation
------------

To install the latest release of JAXopt, use the following command::

    $ pip install jaxopt

To install the **development** version, use the following command instead::

    $ pip install git+https://github.com/google/jaxopt

Alternatively, it can be be installed from sources with the following command::

    $ python setup.py install

.. toctree::
   :maxdepth: 1
   :caption: Documentation

   basics
   unconstrained
   constrained
   quadratic_programming
   non_smooth
   stochastic
   objective_and_loss
   root_finding
   fixed_point
   implicit_diff

.. toctree::
   :maxdepth: 1
   :caption: API

   api

.. toctree::
   :maxdepth: 2
   :caption: Examples

   auto_examples/index

.. toctree::
   :maxdepth: 1
   :caption: About

   Authors <https://github.com/google/jaxopt/graphs/contributors>
   changelog
   Source code <https://github.com/google/jaxopt>
   Issue tracker <https://github.com/google/jaxopt/issues>

Support
-------

If you are having issues, please let us know by filing an issue on our
`issue tracker <https://github.com/google/jaxopt/issues>`_.

License
-------

JAXopt is licensed under the Apache 2.0 License.

Indices and tables
==================

* :ref:`genindex`
