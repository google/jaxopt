Root finding
============

This section is concerned with root finding, that is finding :math:`x` such
that :math:`F(x, \theta) = 0`.

Bisection
---------

.. autosummary::
  :toctree: _autosummary

    jaxopt.Bisection

Bisection is a suitable algorithm when :math:`F(x, \theta)` is one-dimensional
in :math:`x`.

Scipy wrapper
-------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.ScipyRootFinding
