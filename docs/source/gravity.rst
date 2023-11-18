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
