"""
Solver and Pipeline Framework for Cluster Models
=================================================

This module provides a flexible and extensible framework for solving fields in a `ClusterModel`
using a pipeline architecture. It introduces the concept of `Solver` classes responsible for
solving individual fields within a model and the `Pipeline` class to manage the orchestration of
solvers in a predefined sequence (field order). The framework is designed with modularity,
extensibility, and ease of use in mind, allowing developers to customize solvers, pipelines,
and model behaviors efficiently.

Overview
--------

In computational astrophysics and related fields, complex simulations often require solving a series
of equations or computing quantities in a specific order. Each step of this process may rely on
the result of previous computations, making it essential to define both the sequence of operations
and the solvers for each operation. This module solves this problem by providing two key abstractions:

1. **Solver**: A solver is responsible for solving a specific field in a model. Each solver works on
   one or more fields and can be customized to operate on a given grid within the `ClusterModel`. Solvers
   may also have a priority, which helps determine the most suitable solver when multiple are available.

2. **Pipeline**: A pipeline is a sequence of steps (fields) that need to be solved. It manages the
   association of fields with solvers and ensures that each solver is applied in the correct order.
   The pipeline also includes methods for loading and saving the pipeline configuration to an HDF5
   file, enabling persistent storage of simulation setups.

Key Features
------------

- **Modular Solvers**: The `Solver` abstract base class provides an interface for creating solvers
  that solve specific fields in the model. Developers can extend this class to define custom solvers
  that implement the `__call__` method. Each solver is associated with one or more fields it can
  solve, and developers can set priorities to control solver selection when multiple solvers are
  available for the same field.

- **Pipeline Orchestration**: The `Pipeline` class is responsible for managing the execution of
  solvers in a specific sequence. It builds a procedure based on the available solvers for each field
  in `FIELD_ORDER`, ensuring that solvers are run in the correct order. The pipeline also
  supports logging, progress tracking via `tqdm`, and saving/loading pipelines to and from HDF5
  files.

- **Solver Selection**: Each solver has an associated `PRIORITY`, which controls its selection when
  multiple solvers are available for the same field. The pipeline selects the solver with the lowest
  priority to ensure the most efficient or appropriate solver is used.

- **Persistence with HDF5**: Both solvers and pipelines can be saved to and loaded from HDF5 files.
  This feature allows developers to save a pipeline setup, including solver configurations, and
  resume the simulation or reapply the same configuration later.

How to Use
----------

### Creating a Solver
To create a custom solver, inherit from the `Solver` class and implement the `__call__` method.
This method performs the solving operation for the associated field. You can define multiple fields
a solver can solve by setting the `FIELDS` attribute, and you can define the solver's priority with
the `PRIORITY` attribute.

For example:

.. code-block:: python

    class MyDensitySolver(Solver):
        FIELDS = ['density']
        PRIORITY = 10  # Lower priority is preferred

        def __call__(self, grid: Grid, *args, **kwargs) -> None:
            # Perform operations to solve the density field
            grid['density'] = compute_density(grid, *args, **kwargs)

"""
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, List, Optional, Type, Union

import h5py
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from cluster_generator.solvers._types import (
    DEFAULT_SOLVER_REGISTRY,
    FieldAlias,
    FieldOrder,
    ModelSolverRegistry,
    SolverMeta,
)
from cluster_generator.utils import mylog, tqdmWarningRedirector

if TYPE_CHECKING:
    from cluster_generator.grids.grids import Grid
    from cluster_generator.models._abc import ClusterModel


class Solver(ABC, metaclass=SolverMeta):
    """
    Abstract base class for all solvers. Each solver is responsible for solving a specific field
    in the grid associated with a ClusterModel.

    Attributes
    ----------
    FIELDS : list of FieldAlias, optional
        The fields that the solver works on.
    PRIORITY : int, optional
        The priority of the solver. Lower priority values are preferred when multiple solvers
        are available.

    Parameters
    ----------
    model : ClusterModel
        The model associated with this solver.
    field : str, optional
        The field that this solver is solving for. If not provided, it will default to the first
        entry in `FIELDS`.

    Raises
    ------
    ValueError
        If no field is specified or if the field is incompatible with the model.
    """

    FIELDS: Optional[List[FieldAlias]] = None
    PRIORITY: Optional[int] = None

    def __init__(self, model: "ClusterModel", field: Optional[FieldAlias] = None):
        """
        Initialize the solver instance.

        Parameters
        ----------
        model : ClusterModel
            The model associated with this solver.
        field : str, optional
            The field that this solver is solving for. If not provided, it will default to the
            first entry in `FIELDS`.

        Raises
        ------
        ValueError
            If no field is specified or if the field is incompatible with the model.
        """
        self.model = model
        self.field = field or (
            self.FIELDS[0] if self.FIELDS and len(self.FIELDS) == 1 else None
        )

        if not self.field:
            raise ValueError(
                f"Solver {self.__class__.__name__} requires a field to be specified."
            )

        # Validate that the solver is valid to run.
        if not self.check_availability(self.model, self.field):
            raise ValueError(
                f"{self} failed availability check. The field '{self.field}' or model '{self.model}' is incompatible."
            )

    @abstractmethod
    def __call__(self, grid: "Grid", *args, **kwargs) -> None:
        """
        Perform the solving operation on the given grid. Must be implemented by subclasses.

        Parameters
        ----------
        grid : Grid
            The grid on which the solver will operate.
        *args, **kwargs
            Additional arguments for the solving operation.
        """
        pass

    def __str__(self) -> str:
        """
        Returns
        -------
        str
            String representation of the solver showing the class name and field.
        """
        return f"<{self.__class__.__name__}: Field={self.field}>"

    def __repr__(self) -> str:
        """
        Returns
        -------
        str
            Detailed string representation, including the solver class, field, and priority.
        """
        return f"(Solver Class {self.__class__.__name__}): Field={self.field}, Priority={self.PRIORITY}."

    def __eq__(self, other: object) -> bool:
        """
        Check for equality between two solvers.

        Parameters
        ----------
        other : object
            The other solver to compare against.

        Returns
        -------
        bool
            True if the solvers are equal, False otherwise.
        """
        if not isinstance(other, Solver):
            return NotImplemented
        return (self.__class__.__name__ == other.__class__.__name__) and (
            self.field == other.field
        )

    def __ne__(self, other: object) -> bool:
        """
        Check for inequality between two solvers.

        Parameters
        ----------
        other : object
            The other solver to compare against.

        Returns
        -------
        bool
            True if the solvers are not equal, False otherwise.
        """
        return not self.__eq__(other)

    def __le__(self, other: "Solver") -> bool:
        """
        Compare two solvers based on their priority.

        Parameters
        ----------
        other : Solver
            The other solver to compare against.

        Returns
        -------
        bool
            True if this solver's priority is less than or equal to the other solver's priority.
        """
        return self.PRIORITY <= other.PRIORITY

    def __ge__(self, other: "Solver") -> bool:
        """
        Compare two solvers based on their priority.

        Parameters
        ----------
        other : Solver
            The other solver to compare against.

        Returns
        -------
        bool
            True if this solver's priority is greater than or equal to the other solver's priority.
        """
        return self.PRIORITY >= other.PRIORITY

    def __lt__(self, other: "Solver") -> bool:
        """
        Compare two solvers based on their priority.

        Parameters
        ----------
        other : Solver
            The other solver to compare against.

        Returns
        -------
        bool
            True if this solver's priority is less than the other solver's priority.
        """
        return self.PRIORITY < other.PRIORITY

    def __gt__(self, other: "Solver") -> bool:
        """
        Compare two solvers based on their priority.

        Parameters
        ----------
        other : Solver
            The other solver to compare against.

        Returns
        -------
        bool
            True if this solver's priority is greater than the other solver's priority.
        """
        return self.PRIORITY > other.PRIORITY

    def to_hdf5(self, hdf5_file: Union[str, h5py.File], group_path: str) -> None:
        """
        Save the solver instance to the specified HDF5 file and group path.

        Parameters
        ----------
        hdf5_file : str or h5py.File
            The path to the HDF5 file or an open h5py.File object where the solver will be saved.
        group_path : str
            The path within the HDF5 file where the solver will be stored.
        """
        close_file = isinstance(hdf5_file, str)
        if close_file:
            hdf5_file = h5py.File(hdf5_file, "a")

        try:
            group = hdf5_file.require_group(group_path)

            # Store solver class name and field as attributes
            group.attrs["class_name"] = self.__class__.__name__
            group.attrs["field"] = self.field

        finally:
            if close_file:
                hdf5_file.close()

    @classmethod
    def from_hdf5(
        cls, hdf5_file: Union[str, h5py.File], group_path: str, model: "ClusterModel"
    ) -> "Solver":
        """
        Load a solver instance from the specified HDF5 file and group path.

        Parameters
        ----------
        hdf5_file : str or h5py.File
            The path to the HDF5 file or an open h5py.File object from which the solver will be loaded.
        group_path : str
            The path within the HDF5 file where the solver is stored.
        model : ClusterModel
            The model that the solver will be associated with.

        Returns
        -------
        Solver
            An instance of the solver object initialized from the HDF5 file.

        Raises
        ------
        ValueError
            If the class specified in the HDF5 file cannot be found in the solver registry.
        """
        close_file = isinstance(hdf5_file, str)
        if close_file:
            hdf5_file = h5py.File(hdf5_file, "r")

        try:
            group = hdf5_file[group_path]

            # Retrieve solver class name, field, and priority from attributes
            class_name = group.attrs["class_name"]
            field = group.attrs["field"]

            # Fetch solver class from the registry or subclasses
            solver_class = cls._get_class_by_name(class_name)
            if solver_class is None:
                raise ValueError(
                    f"Unknown solver class '{class_name}' in file '{hdf5_file}'."
                )

            # Instantiate the solver with the loaded parameters
            return solver_class(model=model, field=field)

        finally:
            if close_file:
                hdf5_file.close()

    @staticmethod
    def _get_class_by_name(class_name: str) -> Optional[Type["Solver"]]:
        """
        Retrieve a solver class by its name. Searches through Solver subclasses.

        Parameters
        ----------
        class_name : str
            The name of the solver class.

        Returns
        -------
        Optional[Type[Solver]]
            The solver class, if found, otherwise None.
        """

        def find_in_subclasses(base_class: Type["Solver"]) -> Optional[Type["Solver"]]:
            for subclass in base_class.__subclasses__():
                if subclass.__name__ == class_name:
                    return subclass
                result = find_in_subclasses(subclass)
                if result:
                    return result
            return None

        return find_in_subclasses(Solver)

    @classmethod
    @abstractmethod
    def check_availability(cls, model: "ClusterModel", field: str) -> bool:
        """
        Check if this solver can be used for the provided model and field.

        Parameters
        ----------
        model : ClusterModel
            The model to check against.
        field : str
            The field that needs solving.

        Returns
        -------
        bool
            True if the solver is available for the given model and field, False otherwise.
        """
        pass


class Pipeline(ABC):
    """
    Abstract base class for a pipeline that manages the solving process for multiple fields
    in a `ClusterModel`. The pipeline orchestrates solvers in a specific order and ensures that
    all fields in the model are solved in sequence.

    Attributes
    ----------
    FIELD_ORDER : FieldOrder
        The ordered list of fields that the pipeline will solve for. Each field must have a corresponding solver.
    procedure : OrderedDict
        An ordered dictionary mapping each field to its corresponding solver.
    model : ClusterModel
        The model that the pipeline operates on.
    registry : ModelSolverRegistry
        The registry containing available solvers for the pipeline.

    Parameters
    ----------
    model : ClusterModel
        The model that the pipeline operates on.
    registry : ModelSolverRegistry, optional
        The registry from which solvers are drawn. If not provided, the default solver registry is used.
    """

    FIELD_ORDER: FieldOrder = None

    def __init__(self, model: "ClusterModel", registry: ModelSolverRegistry = None):
        """
        Initialize the pipeline and associate it with the provided model. The pipeline attempts
        to find appropriate solvers for each field in `FIELD_ORDER`.

        Parameters
        ----------
        model : ClusterModel
            The model associated with the pipeline.
        registry : ModelSolverRegistry, optional
            The registry from which solvers are drawn. If not provided, the default solver registry is used.
        """
        self.model = model
        self.registry: ModelSolverRegistry = registry or DEFAULT_SOLVER_REGISTRY

        # Construct an ordered dictionary of fields and solvers.
        self.procedure: OrderedDict[str, Optional[Solver]] = OrderedDict(
            {field: None for field in self.FIELD_ORDER}
        )
        self.build_procedure()

    def __str__(self) -> str:
        """
        Return a string representation of the pipeline, showing its class and associated model.

        Returns
        -------
        str
            A string representing the pipeline.
        """
        return f"<{self.__class__.__name__}, Model={self.model}>"

    def __repr__(self) -> str:
        """
        Return a string representation of the pipeline, displaying the procedure with fields and solvers.

        Returns
        -------
        str
            A string representing the pipeline procedure.
        """
        return str(self.procedure)

    def __getitem__(self, field: str) -> Optional[Solver]:
        """
        Access the solver assigned to a specific field.

        Parameters
        ----------
        field : str
            The field whose solver you want to access.

        Returns
        -------
        Solver or None
            The solver assigned to the field, or None if no solver is assigned.

        Raises
        ------
        KeyError
            If the field is not part of the pipeline.
        """
        if field not in self.procedure:
            raise KeyError(f"Field '{field}' is not part of the pipeline.")
        return self.procedure[field]

    def __setitem__(self, field: str, solver: Solver) -> None:
        """
        Assign a solver to a specific field in the pipeline.

        Parameters
        ----------
        field : str
            The field for which the solver will be assigned.
        solver : Solver
            The solver to assign to the field.

        Raises
        ------
        ValueError
            If the solver's field does not match the expected field.
        KeyError
            If the field is not part of the pipeline.
        """
        if solver.field != field:
            raise ValueError(
                f"Solver field '{solver.field}' does not match expected field '{field}'."
            )
        if field not in self.procedure:
            raise KeyError(
                f"Field '{field}' does not exist in the pipeline. Modify the pipeline class to add new fields."
            )
        self.procedure[field] = solver

    def __call__(self, grid: "Grid", *args, **kwargs) -> None:
        """
        Run the pipeline, solving each field in sequence for the provided grid. A progress bar is
        displayed using `tqdm`, and logging information is redirected during execution.

        Parameters
        ----------
        grid : Grid
            The grid on which the solvers will operate.
        *args, **kwargs
            Additional arguments passed to each solver during the call.

        Raises
        ------
        ValueError
            If the pipeline is not tractable (i.e., if some fields do not have assigned solvers).
        """
        if not self.is_tractable:
            raise ValueError(
                "The pipeline is not tractable. Ensure all fields have an assigned solver."
            )

        mylog.info("Solving %s with %s...", grid, self)
        with logging_redirect_tqdm(
            loggers=[mylog, grid.grid_manager.logger]
        ), tqdmWarningRedirector():
            with tqdm(
                total=len(self.procedure), desc=f"Solving {grid} (Setup)", unit="field"
            ) as pbar:
                for field, solver in self.procedure.items():
                    pbar.set_description(f"Solving {grid} ({field})")
                    logging.info("Solving field '%s' using %s...", field, solver)
                    solver(grid, *args, **kwargs)
                    pbar.update(1)

        logging.info(f"Pipeline completed for model: {self.model}")

    # Alias for __call__
    def solve(self, grid: "Grid", *args, **kwargs) -> None:
        """
        Alias for the `__call__` method to solve the pipeline.

        Parameters
        ----------
        grid : Grid
            The grid on which the solvers will operate.
        *args, **kwargs
            Additional arguments passed to each solver.
        """
        self.__call__(grid, *args, **kwargs)

    @property
    def is_tractable(self) -> bool:
        """
        Check if the pipeline is tractable (i.e., all fields have an assigned solver).

        Returns
        -------
        bool
            True if all fields have an assigned solver, False otherwise.
        """
        return all(solver is not None for solver in self.procedure.values())

    def build_procedure(self):
        """
        Build the procedure by identifying valid solvers for each field in `FIELD_ORDER`. Solvers
        are selected based on their availability and priority.
        """
        for field in self.procedure:
            available_solvers = [
                solver_class
                for solver_class in self.registry.values()
                if (solver_class.FIELDS is None) or (field in solver_class.FIELDS)
            ]
            candidate_solvers = [
                (solver_class, solver_class.PRIORITY)
                for solver_class in available_solvers
                if solver_class.check_availability(self.model, field)
            ]

            if not candidate_solvers:
                mylog.warning(
                    f"Failed to find a valid solver for the {field} step of {self}."
                )
                continue

            best_solver_class = min(candidate_solvers, key=lambda x: x[1])[0]
            self.procedure[field] = best_solver_class(self.model, field=field)

    def to_hdf5(self, hdf5_file: Union[str, h5py.File], group_path: str) -> None:
        """
        Save the pipeline instance to the specified HDF5 file.

        Parameters
        ----------
        hdf5_file : str or h5py.File
            The path to the HDF5 file or an open h5py.File object where the pipeline will be saved.
        group_path : str
            The path within the HDF5 file where the pipeline will be stored.
        """
        close_file = isinstance(hdf5_file, str)
        if close_file:
            hdf5_file = h5py.File(hdf5_file, "a")

        try:
            group = hdf5_file.require_group(group_path)
            group.attrs["FIELD_ORDER"] = self.FIELD_ORDER
            group.attrs["class_name"] = self.__class__.__name__

            for field, solver in self.procedure.items():
                if solver is not None:
                    solver.to_hdf5(group, field)
        finally:
            if close_file:
                hdf5_file.close()

    @classmethod
    def from_hdf5(
        cls, hdf5_file: Union[str, h5py.File], group_path: str, model: "ClusterModel"
    ) -> "Pipeline":
        """
        Load a pipeline instance from an HDF5 file.

        Parameters
        ----------
        hdf5_file : str or h5py.File
            The path to the HDF5 file or an open h5py.File object from which the pipeline will be loaded.
        group_path : str
            The path within the HDF5 file where the pipeline is stored.
        model : ClusterModel
            The model that the pipeline will be associated with.

        Returns
        -------
        Pipeline
            An instance of the pipeline object initialized from the HDF5 file.
        """
        close_file = isinstance(hdf5_file, str)
        if close_file:
            hdf5_file = h5py.File(hdf5_file, "r")

        try:
            group = hdf5_file[group_path]
            field_order = list(group.attrs["FIELD_ORDER"])
            class_name = group.attrs["class_name"]
            pipeline_class = cls._get_class_by_name(class_name)
            pipeline = pipeline_class(model)
            pipeline.FIELD_ORDER = field_order

            for field in field_order:
                solver_group = group[field]
                solver_class_name = solver_group.attrs["class_name"]
                solver_class = pipeline._get_class_by_name(solver_class_name)

                if solver_class is not None:
                    solver = solver_class.from_hdf5(
                        hdf5_file, group_path + "/" + field, model
                    )
                    pipeline.procedure[field] = solver

            return pipeline
        finally:
            if close_file:
                hdf5_file.close()

    @staticmethod
    def _get_class_by_name(class_name: str) -> Optional[Type["Pipeline"]]:
        """
        Retrieve a pipeline class by its name.

        Parameters
        ----------
        class_name : str
            The name of the pipeline class.

        Returns
        -------
        Optional[Type[Pipeline]]
            The pipeline class, if found, otherwise None.
        """

        def find_in_subclasses(
            base_class: Type["Pipeline"],
        ) -> Optional[Type["Pipeline"]]:
            for subclass in base_class.__subclasses__():
                if subclass.__name__ == class_name:
                    return subclass
                result = find_in_subclasses(subclass)
                if result:
                    return result
            return None

        return find_in_subclasses(Pipeline)

    def print_summary(self) -> None:
        """
        Print a summary of the pipeline, displaying each field, the assigned solver, and the solver's priority.
        """
        headers = ["Field", "Solver", "Priority"]
        col_widths = [
            max(len(headers[0]), max(len(field) for field in self.procedure.keys())),
            max(
                len(headers[1]),
                max(
                    len(solver.__class__.__name__) if solver else len("None")
                    for solver in self.procedure.values()
                ),
            ),
            max(
                len(headers[2]),
                max(
                    len(str(solver.PRIORITY)) if solver else len("N/A")
                    for solver in self.procedure.values()
                ),
            ),
        ]

        # Print formatted header row and separator
        header_row = f"{headers[0]:<{col_widths[0]}}  {headers[1]:<{col_widths[1]}}  {headers[2]:<{col_widths[2]}}"
        separator = "-" * (sum(col_widths) + 4)

        print(header_row)
        print(separator)

        # Print each row with formatting
        for field, solver in self.procedure.items():
            solver_name = solver.__class__.__name__ if solver else "None"
            priority = str(solver.PRIORITY) if solver else "N/A"
            row = f"{field:<{col_widths[0]}}  {solver_name:<{col_widths[1]}}  {priority:<{col_widths[2]}}"
            print(row)
