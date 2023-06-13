Perturbed optimization
======================

The perturbed optimization module allows to transform a non-smooth function such as a max or arg-max into a differentiable function using random perturbations. This is useful for optimization algorithms that require differentiability, such as gradient descent (e.g. see :doc:`Notebook </notebooks/perturbed_optimizers/perturbed_optimizers>` on perturbed optimizers).


Max perturbations
-----------------

Consider a maximum function of the form:

.. math::

    F(\theta) = \max_{y \in \mathcal{C}} \langle y, \theta\rangle\,,

where :math:`\mathcal{C}` is a convex set.



.. autosummary::
  :toctree: _autosummary

    jaxopt.perturbations.make_perturbed_max




The function :meth:`jaxopt.perturbations.make_perturbed_max` transforms the function :math:`F` into a the following differentiable function using random perturbations:


.. math::

    F_{\varepsilon}(\theta) = \mathbb{E}\left[ F(\theta + \varepsilon Z) \right]\,,

where :math:`Z` is a random variable. The distribution of this random variable can be specified through the keyword argument ``noise``. The default is a Gumbel distribution, which is a good choice for discrete variables. For continuous variables, a normal distribution is more appropriate.


Argmax perturbations
--------------------

Consider an arg-max function of the form:

.. math::

    y^*(\theta) = \mathop{\mathrm{arg\,max}}_{y \in \mathcal{C}} \langle y, \theta\rangle\,,

where :math:`\mathcal{C}` is a convex set.


The function :meth:`jaxopt.perturbations.make_perturbed_argmax` transforms the function :math:`y^\star` into a the following differentiable function  using random perturbations:


.. math::

    y_{\varepsilon}^*(\theta) = \mathbb{E}\left[ \mathop{\mathrm{arg\,max}}_{y \in \mathcal{C}} \langle y, \theta + \varepsilon Z \rangle \right]\,,

where :math:`Z` is a random variable. The distribution of this random variable can be specified through the keyword argument ``noise``. The default is a Gumbel distribution, which is a good choice for discrete variables. For continuous variables, a normal distribution is more appropriate.


.. autosummary::
  :toctree: _autosummary

    jaxopt.perturbations.make_perturbed_argmax


Scalar perturbations
--------------------

Consider any function, :math:`f` that is not necessarily differentiable, e.g. piecewise-constant of the form:

.. math::

    f(\theta) = g(y^*(\theta))\,,

where :math:`\mathop{\mathrm{arg\,max}}_{y \in \mathcal{C}} \langle y, \theta\rangle`` and :math:`\mathcal{C}` is a convex set.


The function :meth:`jaxopt.perturbations.make_perturbed_fun` transforms the function :math:`f` into a the following differentiable function  using random perturbations:

.. math::

    f_{\varepsilon}(\theta) = \mathbb{E}\left[ f(\theta + \varepsilon Z) \right]\,,

where :math:`Z` is a random variable. The distribution of this random variable can be specified through the keyword argument ``noise``. The default is a Gumbel distribution, which is a good choice for discrete variables. For continuous variables, a normal distribution is more appropriate. This can be particulary useful in the example given above, when :math:`f` is only defined on the discrete set, not its convex hull, i.e.

.. math::

    f_{\varepsilon}(\theta) = \mathbb{E}\left[ g(\mathop{\mathrm{arg\,max}}_{y \in \mathcal{C}} \langle y, \theta + \varepsilon Z \rangle) \right]\,,


.. autosummary::
  :toctree: _autosummary

    jaxopt.perturbations.make_perturbed_fun


Noise distributions
-------------------

The functions :meth:`jaxopt.perturbations.make_perturbed_max`, :meth:`jaxopt.perturbations.make_perturbed_argmax` and :meth:`jaxopt.perturbations.make_perturbed_fun` take a keyword argument ``noise`` that specifies the distribution of random perturbations. Pre-defined distributions for this argument are the following:

.. autosummary::
  :toctree: _autosummary

    jaxopt.perturbations.Normal
    jaxopt.perturbations.Gumbel




.. topic:: References

    Berthet, Q., Blondel, M., Teboul, O., Cuturi, M., Vert, J. P., & Bach, F. (2020). `Learning with differentiable pertubed optimizers <https://arxiv.org/pdf/2002.08676.pdf>`_. Advances in neural information processing systems, 33.
