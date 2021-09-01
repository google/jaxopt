Non-smooth optimization
=======================

This section is concerned with problems of the form

.. math::

    \min_{x} f(x, \theta) + g(x, \upsilon)

where :math:`f(x, \theta)` is a smooth function,
:math:`x` are the parameters with respect to which the function is minimized,
:math:`\theta` are optional extra arguments,
:math:`g(x, \upsilon)` is possibly non-smooth,
and :math:`\upsilon` are extra parameters :math:`g` may depend on.


Proximal gradient
-----------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.ProximalGradient


Block coordinate descent
------------------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.BlockCoordinateDescent

Proximal operators
------------------

Proximal gradient and block coordinate descent do not access :math:`g(x, \upsilon)`
directly but instead require its associated proximal operator

.. math::

    \text{prox}_{g}(x', \upsilon, \eta) :=
    \underset{x}{\text{argmin}} ~ \frac{1}{2} ||x' - x||^2 + \eta g(x, \upsilon).

The following operators are available.

.. autosummary::
  :toctree: _autosummary

    jaxopt.prox.make_prox_from_projection
    jaxopt.prox.prox_none
    jaxopt.prox.prox_lasso
    jaxopt.prox.prox_non_negative_lasso
    jaxopt.prox.prox_elastic_net
    jaxopt.prox.prox_group_lasso
    jaxopt.prox.prox_ridge
    jaxopt.prox.prox_non_negative_ridge
