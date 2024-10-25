"""
Frequently used / needed Solver subclasses.

This module defines a system for creating and managing solvers in a pipeline-based framework.
Solvers are computational units responsible for executing tasks such as data processing or
field injection within the context of models and grids. The system allows for flexible solver
construction by providing both static (predefined) solvers and dynamically configured solvers
that can be constructed at runtime.

Serialization Support
---------------------
All solvers in this module support serialization to and from HDF5 using the `to_hdf5` and
`from_hdf5` methods. This allows solver instances to be saved to disk, transferred, and
reconstructed with their full state intact. The `dill` library is used to serialize and
deserialize solver functions, ensuring compatibility with complex Python objects.

Usage Examples
--------------
Basic solver definition:
>>> @solver()
>>> def example_solver(pipeline, grid):
>>>     # Solver logic here
>>>     pass

Customizing with validation and setup:
>>> @solver()
>>> def custom_solver(pipeline, grid):
>>>     # Solver logic here
>>>     pass

>>> @custom_solver.validator
>>> def validate(pipeline):
>>>     return True

>>> @custom_solver.setup_pipeline
>>> def setup_pipeline(pipeline):
>>>     pass

The module is designed to provide flexible and extensible solvers that integrate into
a pipeline-driven architecture, supporting dynamic solver construction and easy serialization.
"""
import json
from typing import TYPE_CHECKING, Callable, Any, Union

import dill
import h5py
import numpy as np
from functools import wraps
from cluster_generator.grids.grids import Grid
from cluster_generator.pipelines.solvers.abc import Solver
from cluster_generator.pipelines.solvers._types import SolverLike, Validator, SetupMethod

if TYPE_CHECKING:
    from cluster_generator.pipelines.abc import Pipeline
    from cluster_generator.grids._types import FieldAlias

class NoOpSolver(Solver):
    """
    A no-operation solver that performs no computation.

    The `NoOpSolver` is a placeholder solver that can be used within a pipeline
    to denote tasks that require no computation. This solver is useful for marking
    boundaries, such as the start or end of a pipeline sequence, without performing
    any action.

    This solver is part of the `Solver` hierarchy and is designed to integrate seamlessly
    into the pipeline system while performing no work. It can be useful in pipelines
    that need structural markers, but do not require any specific operations.

    Notes
    -----
    This solver does not interact with the pipeline, grid, or model in any way
    and serves purely as a placeholder in the pipeline sequence.

    .. note::

        This is what we use for the ``start`` and ``end`` nodes of the pipeline because they
        must be present but don't need to actually do anything.

    Examples
    --------
    Using the `NoOpSolver` to mark start and end tasks in a pipeline:

    >>> pipeline = Pipeline()
    >>> pipeline.add_task("start", NoOpSolver())
    >>> pipeline.add_task("end", NoOpSolver())
    """

    def __call__(self, *args):
        """
        Perform no operation when invoked.

        This method is called during pipeline execution, but for `NoOpSolver`,
        it does not modify the pipeline, grid, or any other context.

        Parameters
        ----------
        *args : tuple
            Any arguments passed during the pipeline execution.

        Returns
        -------
        None
            No value is returned since no operation is performed.
        """
        return None

    def _validate(self, pipeline: 'Pipeline') -> bool:
        """
        Always returns True, indicating that the `NoOpSolver` is valid.

        Since the `NoOpSolver` performs no work, validation always succeeds. This
        method is called by the pipeline during setup or execution to verify that
        the solver can be used in the pipeline.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline in which the solver is being validated.

        Returns
        -------
        bool
            Always returns True, indicating successful validation.
        """
        return True

    def _setup_in_pipeline(self, pipeline: 'Pipeline'):
        """
        Perform no setup for the pipeline.

        Since the `NoOpSolver` does not interact with the pipeline, this method
        is a placeholder and performs no operations. It is called by the pipeline
        during setup to allow solvers to prepare for execution.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline in which the solver is being set up.

        Returns
        -------
        None
            No setup is required, so no value is returned.
        """
        pass

    def _setup_in_model(self, pipeline: 'Pipeline'):
        """
        Perform no setup for the model.

        The `NoOpSolver` does not interact with the model, so this method is a
        placeholder and performs no operations. It is called by the pipeline to
        allow solvers to set up their model-specific state.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline that holds the reference to the model (if any).

        Returns
        -------
        None
            No model setup is required, so no value is returned.
        """
        pass

class StaticSolver(Solver):
    """
    A dynamically constructed solver with user-provided components such as a
    solver function, validation function, and setup functions.

    The `StaticSolver` class is used to create flexible solver instances, where
    the behavior of the solver is defined at runtime by passing in specific
    functions. This allows for solvers that are both reusable and configurable.

    Parameters
    ----------
    solver_func : callable
        The main solver function that performs computations or updates the pipeline.
    validate_func : callable, optional
        The function used to validate the solver. If not provided, a default validation
        function that always returns `True` is used.
    setup_pipeline_func : callable, optional
        The function used to set up the solver within the pipeline context. If not provided,
        a default no-op function is used.
    setup_model_func : callable, optional
        The function used to set up the solver within the model context. If not provided,
        a default no-op function is used.

    Attributes
    ----------
    _solver_func : callable
        The main solver function provided by the user.
    _validate_func : callable
        The validation function provided by the user (or the default function).
    _setup_pipeline_func : callable
        The pipeline setup function provided by the user (or the default function).
    _setup_model_func : callable
        The model setup function provided by the user (or the default function).
    """
    def __init__(self,
                 solver_func: SolverLike,
                 *args,
                 validate_func: Validator=None,
                 setup_pipeline_func: SetupMethod=None,
                 setup_model_func: SetupMethod=None,
                 **kwargs):
        """
        Initialize a StaticSolver with user-provided functions for solving, validation,
        and setup logic.

        Parameters
        ----------
        solver_func : callable
            The main solver function that executes during pipeline processing.
        validate_func : callable, optional
            Function to validate the solver's configuration. If not provided,
            a default function that always returns `True` is used.
        setup_pipeline_func : callable, optional
            Function to handle pipeline-specific setup. If not provided, a default
            no-op function is used.
        setup_model_func : callable, optional
            Function to handle model-specific setup. If not provided, a default
            no-op function is used.
        """
        super().__init__(*args, **kwargs)

        # Add the functions as core attributes. This allows us to access
        # the functions provided.
        self._solver_func = solver_func
        self._validate_func = validate_func if validate_func else self._default_validate
        self._setup_pipeline_func = setup_pipeline_func if setup_pipeline_func else self._default_setup
        self._setup_model_func = setup_model_func if setup_model_func else self._default_setup

    def __call__(self, pipeline: 'Pipeline', grid: 'Grid') -> Any:
        """
        Execute the solver function within the context of the pipeline and grid.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline in which the solver is being executed.
        grid : Grid
            The grid on which the solver operates.

        Returns
        -------
        Any
            A string representing the result of the solver, which could dictate
            the next step in the pipeline.
        """
        return self._solver_func(pipeline, grid)

    def _validate(self, pipeline) -> bool:
        """
        Validate the solver using the provided validation function.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline in which the solver is being validated.

        Returns
        -------
        bool
            Returns `True` if validation is successful, `False` otherwise.
        """
        return self._validate_func(pipeline)

    def _setup_in_pipeline(self, pipeline):
        """
        Perform pipeline-specific setup using the provided setup function.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline in which the solver is being set up.
        """
        self._setup_pipeline_func(pipeline)

    def _setup_in_model(self, pipeline):
        """
        Perform model-specific setup using the provided setup function.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline associated with the model context.
        """
        self._setup_model_func(pipeline)

    @staticmethod
    def _default_validate(_):
        """
        Default validation logic. Always returns `True`.

        Parameters
        ----------
        _ : Any
            Ignored.

        Returns
        -------
        bool
            Always returns `True`.
        """
        return True

    @staticmethod
    def _default_setup(pipeline):
        """
        Default setup logic. Does nothing.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline in which the solver is being set up.

        Returns
        -------
        None
        """
        pass

    def to_hdf5(self, group: h5py.Group):
        """
        Serialize the StaticSolver to HDF5.

        This method saves the solver function, validation function, and setup functions
        (if they differ from defaults) into an HDF5 file. These functions are stored
        as byte arrays using dill.

        Parameters
        ----------
        group : h5py.Group
            The HDF5 group where the solver will be stored.
        """
        # Save basic solver details
        super().to_hdf5(group)

        # Serialize the solver function (always required)
        group.create_dataset('solver_func', data=np.frombuffer(dill.dumps(self._solver_func), dtype='uint8'))

        # Serialize the validator and setup functions only if they differ from defaults
        if self._validate_func != self._default_validate:
            group.create_dataset('validate_func', data=np.frombuffer(dill.dumps(self._validate_func), dtype='uint8'))
        if self._setup_pipeline_func != self._default_setup:
            group.create_dataset('setup_pipeline_func',
                                 data=np.frombuffer(dill.dumps(self._setup_pipeline_func), dtype='uint8'))
        if self._setup_model_func != self._default_setup:
            group.create_dataset('setup_model_func',
                                 data=np.frombuffer(dill.dumps(self._setup_model_func), dtype='uint8'))

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> 'StaticSolver':
        """
        Deserialize a StaticSolver from HDF5.

        This method reads the solver function, validation function, and setup functions
        from an HDF5 group, reconstructing the StaticSolver instance.

        Parameters
        ----------
        group : h5py.Group
            The HDF5 group from which the solver will be loaded.

        Returns
        -------
        StaticSolver
            A reconstructed StaticSolver instance.
        """
        # Deserialize arguments
        args = json.loads(group.attrs['args'])
        kwargs = json.loads(group.attrs['kwargs'])

        # Deserialize the required solver function
        solver_func = dill.loads(bytes(group['solver_func'][:]))

        # Load optional functions, defaulting to the base class methods if not present
        validate_func = dill.loads(bytes(group['validate_func'][:])) if 'validate_func' in group else None
        setup_pipeline_func = dill.loads(
            bytes(group['setup_pipeline_func'][:])) if 'setup_pipeline_func' in group else None
        setup_model_func = dill.loads(bytes(group['setup_model_func'][:])) if 'setup_model_func' in group else None

        # Create and return a StaticSolver instance
        return cls(solver_func,
                   validate_func=validate_func,
                   setup_pipeline_func=setup_pipeline_func,
                   setup_model_func=setup_model_func,
                   *args,
                   **kwargs)

    def mark_validator(self, func: Callable) -> Callable:
        """
        Assign a validation function to the solver.
        Treats the function as static, so `self` is not required.
        """
        self._validate_func = func  # No need for staticmethod, already assuming it's static.
        func._remove_in_init = True
        return func

    def mark_pipeline(self, func: Callable) -> Callable:
        """
        Assign a pipeline setup function to the solver.
        Treats the function as static, so `self` is not required.
        """
        self._setup_pipeline_func = func  # No need for staticmethod, already assuming it's static.
        func._remove_in_init = True
        return func

    def mark_model(self, func: Callable) -> Callable:
        """
        Assign a model setup function to the solver.
        Treats the function as static, so `self` is not required.
        """
        self._setup_model_func = func  # No need for staticmethod, already assuming it's static.
        func._remove_in_init = True
        return func

class InjectionSolver(StaticSolver):
    """
    Solver for injecting values from a predefined profile into a grid field.

    This solver is designed to apply a function (e.g., a profile or transformation)
    to a specified field in a grid, using the grid's geometry and field units.

    Attributes
    ----------
    field : FieldAlias
        The field that will be injected into the grid.
    units : str
        The units of the field being injected. Defaults to an empty string.
    dtype : str
        The data type of the field. Defaults to 'float64'.

    """
    def __init__(self,
                 field: 'FieldAlias',
                 solver_func: Callable,
                 units: str = '',
                 dtype: str = 'float64',
                 **kwargs):
        """
        Initialize an InjectionSolver.

        Parameters
        ----------
        field : FieldAlias
            The field to be injected into the grid.
        solver_func : Callable
            The function to apply the profile to the grid.
        units : str, optional
            The units of the field. Default is an empty string.
        dtype : str, optional
            The data type of the field. Default is 'float64'.
        """
        super().__init__(solver_func, field, **kwargs)

        # Grab and set the additional attributes.
        self._solver_func = solver_func
        self.field = field
        self.units = units
        self.dtype = dtype

    def __call__(self, pipeline: 'Pipeline', grid: 'Grid') -> Any:
        """
        Execute the solver function within the context of the pipeline and grid.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline in which the solver is being executed.
        grid : Grid
            The grid on which the solver operates.

        Returns
        -------
        Any
            A string representing the result of the solver, which could dictate
            the next step in the pipeline.
        """
        grid.add_field_from_function(self._solver_func,
                                     self.field,
                                     dtype=self.dtype,
                                     units=self.units,
                                     geometry=pipeline.model.geometry)
        return
    def _setup_in_model(self,pipeline, overwrite: bool = False):
        # In order to setup in the model, all we need to do is register the universal field.
        pipeline.model.grid_manager.Fields.register_field(self.field,self.units,self.dtype)

    def _setup_in_pipeline(self, pipeline):
        # We have no tasks to perform when setting up in the pipeline.
        pass

    def _validate(self, pipeline) -> bool:
        """
        Validate the solver.

        Ensure that the model exists in the pipeline.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline in which the solver operates.

        Returns
        -------
        bool
            True if valid, False otherwise.
        """
        return pipeline.model is not None

    def to_hdf5(self, group: h5py.Group):
        """
        Serialize the StaticSolver to HDF5.

        This method saves the solver function, validation function, and setup functions
        (if they differ from defaults) into an HDF5 file. These functions are stored
        as byte arrays using dill.

        Parameters
        ----------
        group : h5py.Group
            The HDF5 group where the solver will be stored.
        """
        # Save basic solver details
        super().to_hdf5(group)

        # Serialize the solver function (always required)
        group.create_dataset('solver_func', data=np.frombuffer(dill.dumps(self._solver_func), dtype='uint8'))

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> 'StaticSolver':
        """
        Deserialize a StaticSolver from HDF5.

        This method reads the solver function, validation function, and setup functions
        from an HDF5 group, reconstructing the StaticSolver instance.

        Parameters
        ----------
        group : h5py.Group
            The HDF5 group from which the solver will be loaded.

        Returns
        -------
        StaticSolver
            A reconstructed StaticSolver instance.
        """
        # Deserialize arguments
        args = json.loads(group.attrs['args'])
        kwargs = json.loads(group.attrs['kwargs'])

        # Deserialize the required solver function
        solver_func = dill.loads(bytes(group['solver_func'][:]))

        # Create and return a StaticSolver instance
        return cls(solver_func,
                   *args,
                   **kwargs)

class ProfileInjectionSolver(InjectionSolver):
    """
    Solver for injecting values from a profile retrieved from the model.

    The ProfileInjectionSolver is a specialized version of the InjectionSolver,
    which automatically retrieves a profile from the model and injects it into
    the grid's field. This allows dynamic profile-based injection during model
    setup.

    Attributes
    ----------
    field : FieldAlias
        The field into which the profile will be injected.
    units : str
        The units of the field. Defaults to an empty string.
    dtype : str
        The data type of the field. Defaults to 'float64'.

    """

    def __init__(self,
                 field: 'FieldAlias',
                 units: str = '',
                 dtype: str = 'float64',
                 **kwargs
                 ):
        # Create a generic solver function as a place holder
        solver_func = lambda _,__: None
        super().__init__(field,solver_func,units=units,dtype=dtype, **kwargs)

        # Grab and set the additional attributes.
        self.field = self._args[0]
        self.units = units
        self.dtype = dtype

    def _setup_in_model(self,pipeline, overwrite: bool = False):
        # Setup the field like we do in all injection solvers.
        super()._setup_in_model(pipeline,overwrite=overwrite)

        # Additionally, we need to grab the profile and set
        # our _solver function.
        self._solver_func = pipeline.model.get_profile(self.field)

        if self._solver_func is None:
            raise ValueError(f"No profile for {self.field} in {pipeline.model}.")

    def _validate(self, pipeline) -> bool:
        # To validate, we need to ensure that the model exists.
        return pipeline.model is not None and pipeline.model.get_profile(self.field) is not None

    def to_hdf5(self, group: h5py.Group):
        """
        Serialize the StaticSolver to HDF5.

        This method saves the solver function, validation function, and setup functions
        (if they differ from defaults) into an HDF5 file. These functions are stored
        as byte arrays using dill.

        Parameters
        ----------
        group : h5py.Group
            The HDF5 group where the solver will be stored.
        """
        # Save basic solver details
        super(InjectionSolver,self).to_hdf5(group)


    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> 'StaticSolver':
        """
        Deserialize a StaticSolver from HDF5.

        This method reads the solver function, validation function, and setup functions
        from an HDF5 group, reconstructing the StaticSolver instance.

        Parameters
        ----------
        group : h5py.Group
            The HDF5 group from which the solver will be loaded.

        Returns
        -------
        StaticSolver
            A reconstructed StaticSolver instance.
        """
        # Deserialize arguments
        args = json.loads(group.attrs['args'])
        kwargs = json.loads(group.attrs['kwargs'])


        # Create and return a StaticSolver instance
        return cls(
                   *args,
                   **kwargs)

def solver(func: SolverLike) -> StaticSolver:
    func._remove_in_init = True
    return StaticSolver(func)

