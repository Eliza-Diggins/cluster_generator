"""
Solver base module

This module contains the base code for the Solver class.
"""
import json
from abc import ABC, abstractmethod
from typing import Optional, Type, TYPE_CHECKING

import h5py
from cluster_generator.grids.grids import Grid
from cluster_generator.utilities.general import find_in_subclasses
from cluster_generator.pipelines.solvers._except import SolverError,SolverSetupError,SolverValidationError

if TYPE_CHECKING:
    from cluster_generator.pipelines.abc import Pipeline

class Solver(ABC):
    """
    Abstract base class for solvers in a pipeline framework.

    The `Solver` class provides the interface for solvers, including methods for
    setup, validation, and execution. Each solver operates within the context of a
    pipeline and a grid, and is responsible for performing computational tasks or
    data transformations.

    Attributes
    ----------
    _args : tuple
        The positional arguments passed to the solver during initialization.
    _kwargs : dict
        The keyword arguments passed to the solver during initialization.
    validation_status : bool or None
        The result of the solver's validation, either `True`, `False`, or `None`
        if validation has not been performed.
    _MODEL_SETUP_FLAG : bool
        Indicates whether the solver has been set up in the model context.
    _PIPELINE_SETUP_FLAG : bool
        Indicates whether the solver has been set up in the pipeline context.
    _VALIDATION_FLAG : bool
        Indicates whether the solver has been validated.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a solver instance with provided arguments.

        The solver stores all initialization arguments to ensure that they can be
        serialized to HDF5 for later reconstruction.

        Parameters
        ----------
        *args : tuple
            Positional arguments for the solver.
        **kwargs : dict
            Keyword arguments for the solver.
        """
        self._args, self._kwargs = args, kwargs

        # Setup flags for tracking the status of model and pipeline setup, and validation.
        self._MODEL_SETUP_FLAG, self._PIPELINE_SETUP_FLAG, self._VALIDATION_FLAG = False, False, False

        # Placeholder for the validation result (None if not validated yet).
        self.validation_status: Optional[bool] = None

    @abstractmethod
    def __call__(self, pipeline: 'Pipeline', grid: 'Grid') -> str:
        """
        Execute solver operations on a grid.

        This abstract method must be implemented by subclasses to define
        the solver's behavior when called within a pipeline.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline instance in which the solver is being executed.
        grid : Grid
            The grid on which the solver operates.

        Returns
        -------
        str
            The output condition that determines the next step in the pipeline.
        """
        pass

    def __str__(self):
        """
        Return a simple string representation of the solver.

        This method returns a string that indicates the class name of the solver.

        Returns
        -------
        str
            A string indicating the class of the solver.
        """
        return f"<{self.__class__.__name__}>"

    def __repr__(self):
        """
        Return a detailed string representation of the solver for debugging.

        This method returns a detailed representation of the solver, including
        the class name, the arguments passed during initialization, and the
        keyword arguments.

        Returns
        -------
        str
            A detailed string representation of the solver instance.
        """
        return f"<{self.__class__.__name__}(args={self._args}, kwargs={self._kwargs})>"

    # -- Validators and Setup Procedures -- #
    def reset_validation_flag(self):
        """
        Reset the validation flag.

        This method sets the `_VALIDATION_FLAG` to `False`, indicating that the solver
        needs to be re-validated before it can be used again in a pipeline. It does not
        reset the actual validation status, which is handled separately by `reset_validation_status()`.

        Use this method when you want to force the solver to go through the validation process again.

        Returns
        -------
        None
        """
        self._VALIDATION_FLAG = False

    def reset_validation_status(self):
        """
        Reset the validation status.

        This method sets `validation_status` to `None`, indicating that the solver's validation
        status has not been determined. It allows the solver to be re-validated from scratch,
        regardless of previous results.

        Use this method in conjunction with `reset_validation_flag()` to fully reset the solver's validation state.

        Returns
        -------
        None
        """
        self.validation_status = None

    def reset_model_setup(self):
        """
        Reset the model setup flag.

        This method sets the `_MODEL_SETUP_FLAG` to `False`, indicating that the solver has not
        been set up in the model context. This forces the solver to re-run its model-specific setup
        tasks the next time it is used in a pipeline.

        Use this method if the model context has changed or if the solver needs to be reconfigured for a new model.

        Returns
        -------
        None
        """
        self._MODEL_SETUP_FLAG = False

    def reset_pipeline_setup(self):
        """
        Reset the pipeline setup flag.

        This method sets the `_PIPELINE_SETUP_FLAG` to `False`, indicating that the solver has not
        been set up in the pipeline context. This forces the solver to re-run its pipeline-specific
        setup tasks the next time it is used.

        Use this method if the pipeline context has changed or if the solver needs to be reconfigured for a new pipeline.

        Returns
        -------
        None
        """
        self._PIPELINE_SETUP_FLAG = False

    def reset_flags(self):
        """
        Reset all of the validation and setup flags for this solver.
        """
        self.reset_validation_flag()
        self.reset_model_setup()
        self.reset_pipeline_setup()
        self.reset_validation_status()

    def validate(self, pipeline: 'Pipeline', overwrite: bool = False, setup_if_needed: bool = False):
        """
        Validate the solver within the context of the pipeline.

        This method checks whether the solver's configuration is valid for use in the
        pipeline. If the solver has already been validated and `overwrite` is `False`,
        the cached validation result is returned. If `setup_if_needed` is `True`, the
        solver will automatically set itself up before validation.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline in which the solver is being validated.
        overwrite : bool, optional
            Whether to force re-validation, even if the solver has been previously validated.
        setup_if_needed : bool, optional
            Whether to perform setup before validation if the solver hasn't been set up yet.

        Returns
        -------
        bool
            The result of the validation (`True` if valid, `False` if invalid).

        Raises
        ------
        SolverValidationError
            If an error occurs during validation.
        """
        if not overwrite and self._VALIDATION_FLAG:
            return self.validation_status

        if setup_if_needed:
            self.setup(pipeline)

        try:
            self.validation_status = self._validate(pipeline)
            self._VALIDATION_FLAG = True
        except Exception as e:
            raise SolverValidationError(f"Failed to validate {self} in pipeline {pipeline}: {e}.")

        return self.validation_status

    def setup(self, pipeline: 'Pipeline', overwrite: bool = False):
        """
        Set up the solver in both the model and pipeline contexts.

        This method performs the necessary setup operations to prepare the solver
        for execution within the pipeline and model. Setup will only occur if it has
        not already been done, unless `overwrite` is `True`.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline in which the solver is being set up.
        overwrite : bool, optional
            Whether to force setup even if it has already been completed.
        """
        self.setup_in_pipeline(pipeline, overwrite=overwrite)
        self.setup_in_model(pipeline, overwrite=overwrite)

    def setup_in_model(self, pipeline: 'Pipeline', overwrite: bool = False):
        """
        Set up the solver in the model context.

        This method performs model-specific setup tasks, ensuring that the solver
        is properly configured to interact with the model.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline containing the model in which the solver operates.
        overwrite : bool, optional
            Whether to force re-setup even if setup has already been completed.

        Raises
        ------
        SolverSetupError
            If an error occurs during model setup.
        """
        if not overwrite and self._MODEL_SETUP_FLAG:
            return

        if pipeline.model is None:
            return

        try:
            self._setup_in_model(pipeline)
            self._MODEL_SETUP_FLAG = True
        except Exception as e:
            raise SolverSetupError(f"Failed to setup solver {self} in model {pipeline.model}: {e}.")

    def setup_in_pipeline(self, pipeline: 'Pipeline', overwrite: bool = False):
        """
        Set up the solver in the pipeline context.

        This method performs pipeline-specific setup tasks, ensuring that the solver
        is properly configured to operate within the pipeline.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline in which the solver is being set up.
        overwrite : bool, optional
            Whether to force re-setup even if setup has already been completed.

        Raises
        ------
        SolverSetupError
            If an error occurs during pipeline setup.
        """
        if not overwrite and self._PIPELINE_SETUP_FLAG:
            return

        try:
            self._setup_in_pipeline(pipeline)
            self._PIPELINE_SETUP_FLAG = True
        except Exception as e:
            raise SolverSetupError(f"Failed to setup solver {self} in pipeline {pipeline}: {e}.")

    @abstractmethod
    def _setup_in_model(self, pipeline: 'Pipeline'):
        """
        Abstract method for model-specific setup logic.

        This method must be implemented by subclasses to define any tasks needed
        to set up the solver in the context of the model.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline that holds the reference to the model.
        """
        pass

    @abstractmethod
    def _setup_in_pipeline(self, pipeline: 'Pipeline'):
        """
        Abstract method for pipeline-specific setup logic.

        This method must be implemented by subclasses to define any tasks needed
        to set up the solver in the context of the pipeline.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline in which the solver is being set up.
        """
        pass

    @abstractmethod
    def _validate(self,pipeline)-> bool:
        pass

    # -- IO Procedures -- #
    def to_hdf5(self, group: h5py.Group):
        """
        Serialize the Solver to HDF5.

        This method stores the solver's class name, arguments, and keyword arguments
        as attributes in the specified HDF5 group. Subclasses can extend this method
        to add additional attributes or datasets as needed.

        Parameters
        ----------
        group : h5py.Group
            The HDF5 group where the solver's data will be stored.
        """
        group.attrs['class_name'] = self.__class__.__name__
        group.attrs['args'] = json.dumps(self._args)
        group.attrs['kwargs'] = json.dumps(self._kwargs)

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> 'Solver':
        """
        Deserialize a Solver from HDF5.

        This method retrieves the class name, arguments, and keyword arguments
        from an HDF5 group and uses them to reconstruct the solver instance.

        Parameters
        ----------
        group : h5py.Group
            The HDF5 group containing the solver's serialized data.

        Returns
        -------
        Solver
            An instance of the solver class reconstructed from the HDF5 data.

        Raises
        ------
        ValueError
            If the class name is not recognized or deserialization fails.
        """
        class_name = group.attrs['class_name']
        solver_class = cls._get_class_by_name(class_name)
        if solver_class is None:
            raise ValueError(f"Unknown solver class '{class_name}'.")

        return solver_class.load_from_hdf5(group)

    @classmethod
    def load_from_hdf5(cls, group: h5py.Group) -> 'Solver':
        """
        Helper method to load a solver instance from HDF5.

        This method reads the arguments and keyword arguments from an HDF5 group
        and uses them to instantiate the solver.

        Parameters
        ----------
        group : h5py.Group
            The HDF5 group containing the solver's serialized data.

        Returns
        -------
        Solver
            A new solver instance created from the stored arguments and kwargs.
        """
        args, kwargs = json.loads(group.attrs['args']), json.loads(group.attrs['kwargs'])
        return cls(*args, **kwargs)

    @staticmethod
    def _get_class_by_name(class_name: str) -> Optional[Type["Solver"]]:
        """
        Retrieve a solver subclass by its class name.

        This method searches through subclasses of `Solver` to locate
        the one matching the provided class name.

        Parameters
        ----------
        class_name : str
            The name of the solver class to find.

        Returns
        -------
        Optional[Type["Solver"]]
            The solver class if found, otherwise None.
        """
        return find_in_subclasses(Solver, class_name)

    def log_status(self, logger):
        """
        Log the current status of the solver.

        This method provides detailed information about the solver's setup and validation status,
        which can be useful for debugging and monitoring the pipeline.

        Parameters
        ----------
        logger : logging.Logger
            A logger instance to which the status is logged.
        """
        logger.info(f"Solver {self.__class__.__name__} status:")
        logger.info(f"  Model setup: {self._MODEL_SETUP_FLAG}")
        logger.info(f"  Pipeline setup: {self._PIPELINE_SETUP_FLAG}")
        logger.info(f"  Validation status: {self.validation_status}")