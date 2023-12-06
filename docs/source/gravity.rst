.. _gravity:

Gravity In Cluster Generator
============================

It's an obvious statement that gravity plays a massive role in the construction of the galaxy cluster models provided by ``cluster_generator``. With
the continued debate in the literature regarding non-newtonian gravity, extensions of general relativity, and other "alternative" gravities,
it is ideal to have a modular implementation of the gravity related mathematical aspects of the code. To accomplish this, ``cluster_generator`` contains the
:py:mod:`gravity` module, which contains all of the necessary structures to allow for easily accessed and modified
gravitation. In this guide, we'll introduce the core concepts of the gravity module.

Using Different Gravity Theories
--------------------------------

If your goal is only to use an alternative gravity theory, the going is really simple! You can create a cluster exactly the same way as
you normally would; however, when you use the ``.from_dens_and_tden`` or any similar method, you can pass a ``gravity`` kwarg. For example, instead of

.. code-block:: python

    model = ClusterModel.from_dens_and_tden(1,10000,density_function,total_density_function,**kwargs)

you can use

.. code-block:: python

    model = ClusterModel.from_dens_and_tden(1,10000,density_function,total_density_function, gravity="<Some gravity to use>",**kwargs)

The value of the ``gravity`` kwarg can be either a gravity class (read below), or a string representation of the gravity name. The
API documentation for the gravity module (:py:mod:`gravity`) contains all of the available classes and the string names.

Looking Behind The Curtain
--------------------------

In essence, gravity is managed by classes like the archetypal :py:class:`gravity.Newtonian`, which in turn is an example of
a descendant from the abstract class :py:class:`gravity.Gravity`. Each gravity class is (loosely) composed of 3 key method:

- The ``.compute_gravitational_field`` method (i.e. :py:meth:`gravity.Newtonian.compute_gravitational_field`).
- The ``.compute_potential`` method (i.e. :py:meth:`gravity.Newtonian.compute_potential`).
- The ``.compute_dynamical_mass`` method (i.e. :py:meth:`gravity.Newtonian.compute_dynamical_mass`).

Each of these methods **must** be defined for any valid gravity class. Each of these methods takes a ``fields`` argument,
containing the available radial profiles (just like the :py:class:`model.ClusterModel` class). Additionally, most of these methods
take a ``method`` kwarg, which can be used to enforce a particular method for completing the computation. If the method is not specified
manually, the code will seek out the first available method which requires only fields the user has provided.

MONDian Gravity
===============

Introduction
------------

Many of the non-newtonian gravitational theories provided in the cluster generator package are so called MONDian theories. These
include the two "archetypal" gravity classes :py:class:`gravity.AQUAL` and :py:class:`gravity.QUMOND` as well as a variety of different
extensions of these core theories. Because these theories are distinctly non-linear, this section will describe the basic premise of the
MONDian gravity theories.

Archetypes
----------

There are two archetypal MONDian theories. The first is the so called AQUAL formulation [MiBe84]_, represented by the :py:class:`gravity.AQUAL` class, and
QUMOND [Milgrom10]_, which is a quasi-linear formulation represented by the :py:class:`gravity.QUMOND` class. These are generically quite different theories; however,
they each result in modified Poisson equations. The AQUAL poisson equation is

.. math::

    \nabla \cdot \left[\nabla \Phi \mu\left(\frac{|\nabla \Phi|}{a_0}\right)\right],

where :math:`a_0` is the MOND constant, and :math:`\mu(x)` is the so called interpolation function. Similarly, the QUMOND formalism contains two Poisson equations:

.. math::

    \nabla^2 \Psi = 4\pi G \rho,

and

.. math::

    \nabla \cdot \left[\nabla \Psi \eta\left(\frac{|\nabla \Psi|}{a_0}\right)\right] = \nabla^2\Phi = 4\pi G \hat{\rho},

where :math:`\eta` is also called the interpolation function, and :math:`\hat{\rho}` is a "phantom" density.

In general, cluster generator manipulates the provided fields in such a way as to solve these equations for the necessary information. Unfortunately,
the theory is manifestly non-linear and so in almost all cases, at least one critical process in the code will include numerically solving a non-linear poisson equation.
As we will see, the interpolation functions play a massive role in these equations.

The Interpolation Maps
----------------------

We have discussed the existence of the functions :math:`\mu` and :math:`\eta`; however, more should be said on the subject. For consistent theoretical behavior,
we require that :math:`x \ll 1 \implies \mu(x) \sim x,\;\;x \gg 1 \implies \mu(x) \sim 1`. Similarly, :math:`x \ll 1 \implies \eta(x) \sim x^{-1/2},\;\; x\gg 1 \implies \eta(x) \sim 1`. As such,
the two are manifestly different; however, they may be connected by a curious link.

.. admonition:: Milgromian Inversion

    Let :math:`\mu(x)` and :math:`F(x)` be functions such that :math:`F(x)=x\mu(x)`. Furthermore, let :math:`G(x)` and `\eta(x)` be defined such that :math:`G(x)= x\eta(x)`. We then say
    that :math:`\mu \equiv \eta` under **Milgromian Inversion** (:math:`\mu(x) = \eta^\dagger(x)`) if and only if

    .. math::

        \forall x \in \mathbb{R} > 0, \; y = F(x) = x \mu(x),\;\;\text{and}\;\;x = G(y) = y\eta(y)

In light of this definition, we introduce a unique aspect of the cluster generator approach, which may be unfamiliar to even seasoned MOND theorists. For each theory, there is a specified
interpolation function :py:meth:`gravity.Mondian.interpolation_function`, as well as an inverse interpolation function :py:meth:`gravity.Mondian.inverse_interpolation_function`. Generically,
one needs only specify an interpolation function; however, if the correct Milgromian inverse is provided, then numerical inversion may be skipped in exchange for a more tractable analytical evaluation.
