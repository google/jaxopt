Basics
======

This section describes useful concepts across all JAXopt.

Pytrees
-------

`Pytrees <https://jax.readthedocs.io/en/latest/pytrees.html>`_ are an essential
concept in JAX and JAXopt. They can be thought as a generalization of vectors.
They are a way to structure parameters or weights using tuples and
dictionaries. Many solvers in JAXopt have native support for pytrees.

Double precision
----------------

JAX uses single (32-bit) floating precision by default. However, for some
algorithms, this may not be enough. Double (64-bit) floating precision can be
enabled by adding the following at the beginning of the file::

  jax.config.update("jax_enable_x64", True)
