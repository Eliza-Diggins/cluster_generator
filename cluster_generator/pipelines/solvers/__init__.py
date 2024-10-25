"""
Pipeline conditions for network edges

This module contains the :py:class:`Condition` class which is simply a wrapper for a function that takes a
pipeline, a grid, and a result, and returns either ``True`` or ``False``. These functions then makeup the edges of
the pipeline network and determine how the user may progress through the pipeline.
"""
from .abc import Solver
from .common import InjectionSolver,ProfileInjectionSolver,StaticSolver,NoOpSolver, solver

__all__ = ['Solver','InjectionSolver','ProfileInjectionSolver','StaticSolver','NoOpSolver','solver']