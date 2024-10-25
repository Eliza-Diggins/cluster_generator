"""
Pipelines Module
================

For every :py:class:`~models.abc.ClusterModel` instance, there is an associated "pipeline" subclassed from
:py:class:`pipelines._abc.Pipeline`. These pipelines pay a central role in ``cluster_generator``: they control how the
model solves its grids / solves new grids. What does this really mean?

- When a :py:class:`~models.abc.ClusterModel` instance is "solved" on a particular grid, it immediately
  passes to its ``pipeline`` and lets the pipeline do the heavy lifting.

This module is dedicated to the logic of the :py:class:`~pipelines._abc.Pipeline` class and the associated classes used
to construct it.

Pipelines
---------

(Main documentation: :py:class:`~pipelines._abc.Pipeline`)

The ``Pipeline`` class is (at the simplest level) a network with nodes corresponding to :py:class:`solvers._abc.Solver` instances
and edges corresponding to :py:class:`conditions._abc.Condition` instances. To solve a specific ``grid`` in a model, the pipeline
is traversed from the ``start`` node to the ``end`` node. Each :py:class:`solver._abc.Solver` instance is called on the grid and
the conditions are used to direct the path through the network.

In this way, models may be solved using a versatile and adaptive set of instructions which are highly extensible.
"""