:github_url: https://github.com/google/jaxopt/tree/master/docs

JAXopt
======

Hardware accelerated, batchable and differentiable optimizers in
`JAX <https://github.com/google/jax>`_.

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

    pip install jaxopt

To install the **development** version, use the following command instead::

    pip install git+https://github.com/google/jaxopt

Alternatively, it can be be installed from sources with the following command::

    python setup.py install

.. toctree::
   :maxdepth: 1
   :caption: Documentation

   basics
   unconstrained
   constrained
   quadratic_programming
   non_smooth
   stochastic
   root_finding
   fixed_point
   nonlinear_least_squares
   linear_system_solvers
   implicit_diff
   objective_and_loss
   line_search

.. toctree::
   :maxdepth: 1
   :caption: API

   api

.. toctree::
   :maxdepth: 2
   :caption: Examples

   notebooks/index
   auto_examples/index

.. toctree::
   :maxdepth: 1
   :caption: About

   Authors <https://github.com/google/jaxopt/graphs/contributors>
   changelog
   Source code <https://github.com/google/jaxopt>
   Issue tracker <https://github.com/google/jaxopt/issues>
   developer

Support
-------

If you are having issues, please let us know by filing an issue on our
`issue tracker <https://github.com/google/jaxopt/issues>`_.

License
-------

JAXopt is licensed under the Apache 2.0 License.


Citing
------

If this software is useful for you, please consider citing
`the paper <https://arxiv.org/abs/2105.15183>`_ that describes
its implicit differentiation framework:

.. code-block:: bibtex

   @article{jaxopt_implicit_diff,
     title={Efficient and Modular Implicit Differentiation},
     author={Blondel, Mathieu and Berthet, Quentin and Cuturi, Marco and Frostig, Roy
      and Hoyer, Stephan and Llinares-L{\'o}pez, Felipe and Pedregosa, Fabian
      and Vert, Jean-Philippe},
     journal={arXiv preprint arXiv:2105.15183},
     year={2021}
   }


Indices and tables
==================

* :ref:`genindex`
