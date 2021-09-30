Fixed Point resolution
======================

This section is concerned with fixed-point resolution, that is finding :math:`x` such
that :math:`T(x, \theta) = x`. 

Simple Fixed Point Iteration
----------------------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.FixedPointIteration

This is the simplest algorithms for fixed-point resolution. It computes the limit of the sequence :math:`x_{n+1}=T(x_n, \theta)` 
which is guaranteed to exist if :math:`x\mapsto T(x,\theta)` is a **contractive map**. See `Banach fixed-point theorem <https://en.wikipedia.org/wiki/Banach_fixed-point_theorem>`_
for more details.

Usage of FixedPointIteration::

  from jaxopt import FixedPointIteration

  def T(x, theta):  # contractive map
    return 0.5 * x + theta

  fpi = FixedPointIteration(fixed_point_fun=T)
  x_init = jnp.array(0.)
  theta = jnp.array(0.5)
  print(fpi.run(x_init, theta).params)

``FixedPointIteration`` successfully finds the root ``x = 1``.  

Differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fixed point computations are efficiently differentiable::

  from jaxopt import FixedPointIteration

  def T(x, theta):  # contractive map
    return 0.5 * x + theta

  fpi = FixedPointIteration(fixed_point_fun=T, implicit_diff=True)
  x_init = jnp.array(0.)
  theta = jnp.array(0.5)

  def fixed_point(x, theta):
    return fpi.run(x, theta).params

  print(jax.grad(fixed_point, argnums=1)(x_init, theta))  # only gradient
  print(jax.value_and_grad(fixed_point, argnums=1)(x_init, theta))  # both value and gradient  

Note that :math:`x(\theta)=2\theta` so :math:`\nabla x(\theta)=2`. 

Under the hood, we use the implicit function theorem in order to differentiate the root.
See the :ref:`implicit differentiation <implicit_diff>` section for more details.

Anderson Acceleration
---------------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.AndersonAcceleration

Anderson acceleration is an iterative method that aims to compute the next iterate :math:`x_{n}` as
a linear combination of the :math:`m` last iterates :math:`[x_{n-m},x_{n-m+1},\ldots x_{n-1}]`. The optimal coefficients
of the linear combination are computed 'on the fly' at each iteration. As a result, not only the convergence is faster
but the convergence conditions are weakened, allowing to tackle problems FixedPointIteration could not.
See `this paper <https://arxiv.org/abs/1909.04638>`_ for more details.
  
The size of the history :math:`m` (denoted ``history_size`` below) plays a crucial role in method performance. 
An higher :math:`m` could speed up the convergence at the cost of higher memory consumption, and more numerical instabilities.
Those numerical instabilities can be mitigated by increasing the ``ridge`` regularization hyper-parameter.

Anderson's acceleration usage is similar to FixedPointIteration::

  from jaxopt import AndersonAcceleration

  def T(x, theta):  # contractive map
    return 0.5 * x + theta

  aa = AndersonAcceleration(fixed_point_fun=T, history_size=5, ridge=1e-6, tol=1e-5)
  x_init = jnp.array(0.)
  theta = jnp.array(0.5)
  print(aa.run(x0, theta).params)

For implicit differentiation::

  from jaxopt import AndersonAcceleration

  def T(x, theta):  # contractive map
    return 0.5 * x + theta

  aa = AndersonAcceleration(fixed_point_fun=T, history_size=5, ridge=1e-6, tol=1e-5, implicit_diff=True)
  x_init = jnp.array(0.)
  theta = jnp.array(0.5)

  def fixed_point(x, theta):
    return aa.run(x, theta).params

  print(jax.grad(fixed_point, argnums=1)(x_init, theta))  # only gradient
  print(jax.value_and_grad(fixed_point, argnums=1)(x_init, theta))  # both value and gradient


Equivalence with root finding
-----------------------------

Note that if :math:`x` is a fixed point of :math:`T` then :math:`x` is a root of :math:`F(x, \theta) = T(x, \theta) - x`. 
Reciproqually, if :math:`x` is the root of some :math:`F(x, \theta)` then it is also the fixed point of :math:`T(x, \theta) = F(x, \theta) + x`.
Hence, root finding and fixed-point resolutions are two different views of the same problem, leading to different approaches.  
We encourage the reader to take a look at :ref:`root finding <root_finding>` section and chose the most appropriate tool for the use-case.  
