from abc import ABCMeta
from typing import TYPE_CHECKING, Iterator, List, Tuple, Type

if TYPE_CHECKING:
    from ._abc import ModelSolver

# Defining type annotations for use throughout the module.
FieldAlias = str
FieldOrder = List[FieldAlias]


class ModelSolverRegistry:
    """Registry for managing ModelSolver classes."""

    def __init__(self):
        self._registry = {}

    def register(self, solver_class: Type["ModelSolver"]):
        """
        Register a solver class in the registry.

        Parameters
        ----------
        solver_class : Type[ModelSolver]
            The solver class to register.

        Raises
        ------
        ValueError
            If a solver class with the same name already exists in the registry.
        """
        solver_name = solver_class.__name__
        if solver_name in self._registry:
            raise ValueError(f"Solver class '{solver_name}' is already registered.")
        self._registry[solver_name] = solver_class

    def keys(self) -> Iterator[str]:
        """
        Return an iterator over the solver names.

        Returns
        -------
        Iterator[str]
            An iterator over registered solver names.
        """
        return iter(self._registry.keys())

    def values(self) -> Iterator[Type["ModelSolver"]]:
        """
        Return an iterator over the solver classes.

        Returns
        -------
        Iterator[Type[ModelSolver]]
            An iterator over registered solver classes.
        """
        return iter(self._registry.values())

    def items(self) -> Iterator[Tuple[str, Type["ModelSolver"]]]:
        """
        Return an iterator over the (name, solver class) pairs.

        Returns
        -------
        Iterator[Tuple[str, Type[ModelSolver]]]
            An iterator over the (solver name, solver class) pairs.
        """
        return iter(self._registry.items())

    def get(self, solver_name: str) -> Type["ModelSolver"]:
        """
        Retrieve a solver class by its name.

        Parameters
        ----------
        solver_name : str
            The name of the solver class to retrieve.

        Returns
        -------
        Type[ModelSolver]
            The solver class corresponding to the provided name.

        Raises
        ------
        KeyError
            If the solver class is not found in the registry.
        """
        try:
            return self._registry[solver_name]
        except KeyError:
            raise KeyError(f"Solver class '{solver_name}' is not registered.")

    def list_solvers(self) -> List[str]:
        """
        List all registered solver class names.

        Returns
        -------
        List[str]
            A list of all registered solver class names.
        """
        return list(self._registry.keys())


# Singleton instance for default solver registration
DEFAULT_SOLVER_REGISTRY = ModelSolverRegistry()


class SolverMeta(ABCMeta):
    """
    Metaclass for automatically registering solver classes when they are created.
    """

    def __init__(cls, name, bases, clsdict):
        """
        Automatically registers a solver class with the default solver registry.

        Parameters
        ----------
        name : str
            Name of the class being created.
        bases : tuple
            Base classes for the new class.
        clsdict : dict
            Dictionary of attributes and methods for the class.
        """
        super().__init__(name, bases, clsdict)

        if name not in ["ModelSolver"]:
            DEFAULT_SOLVER_REGISTRY.register(cls)
