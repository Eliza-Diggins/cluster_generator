from time import perf_counter
from typing import TYPE_CHECKING, Optional, Dict, Tuple, List, Type, Union, TypeVar

import h5py
from tqdm.contrib.logging import logging_redirect_tqdm

from cluster_generator.pipelines._except import PipelineError, PipelineClassError, PipelineHandleNotFoundError, \
    MissingScratchHandlerError
from cluster_generator.pipelines._types import TaskManager, EdgeManager, PipelineLogDescriptor
from cluster_generator.pipelines.conditions import Condition
from cluster_generator.pipelines.scratch_handlers.abc import ScratchSpaceHandler
from cluster_generator.pipelines.solvers import NoOpSolver, Solver, StaticSolver
import numpy as np

if TYPE_CHECKING:
    from cluster_generator.models.abc import ClusterModel
    from cluster_generator.pipelines.solvers._types import SolverLike
    from cluster_generator.pipelines.conditions._types import ConditionLike
    from cluster_generator.grids.grids import Grid
    from cluster_generator.pipelines._types import PipelineLogger

class PipelineMeta(type):
    solver_registry: dict[str, Solver] = {}
    condition_registry: dict[str, Condition] = {}
    _logger: 'PipelineLogger' = None

    def __init__(cls, name, bases, dct):
        # Setup the base class object as normal.
        super().__init__(name, bases, dct)

        # Parse through each of the attributes in the class. We need to sort out conditions
        # and solvers, register them, and then remove them from this namespace.
        for attr_name, attr_value in dct.items():
            # ensure that static methods get converted correctly.
            if isinstance(attr_value, staticmethod):
                attr_value = attr_value.__func__

            # Now check for the attribute status.
            if isinstance(attr_value, StaticSolver):
                cls.register_solver(attr_name,attr_value,overwrite=False)

            elif getattr(attr_value, '_is_condition',False):
                cls.register_condition(attr_name,attr_value,overwrite=False)

        # Cleanup. We now remove these from the name space.
        cls._cleanup_namespace(name,bases,dct)

    def _cleanup_namespace(cls,_,__,dct):
        # We find all of the subcomponents marked for removal and remove
        # them from the class namespace. They are retained in the registries
        # as desired.
        removable_attributes = []

        # Catch all of the actual registry items
        for attribute_name in list(cls.solver_registry.keys())+list(cls.condition_registry.keys()):
            removable_attributes.append(attribute_name)

        # Catch all of the solver subcomponents.
        for attribute_name,attribute_value in dct.items():
            if isinstance(attribute_value, staticmethod):
                attribute_value = attribute_value.__func__

            if getattr(attribute_value, '_remove_in_init',False):
                removable_attributes.append(attribute_name)

        for attribute_name in removable_attributes:
            del dct[attribute_name]
            delattr(cls, attribute_name)

    @classmethod
    def register_solver(cls, name: str, func: Union['SolverLike',Solver,StaticSolver], overwrite: bool = False):
        # Validate if a solver with the same name already exists and overwrite is not allowed
        if name in cls.solver_registry and not overwrite:
            raise PipelineClassError(
                f"Solver '{name}' is already registered in class {cls.__name__}. Use 'overwrite=True' to replace it.")

        # Type check: If func is not already a StaticMethodSolver or a solver-like instance
        if type(func) == Solver or type(func) == StaticSolver:
            cls.solver_registry[name] = func

        elif callable(func):
            cls.solver_registry[name] = StaticSolver(func)

        else:
            raise TypeError(f"Expected a callable or a Solver instance, but got {type(func).__name__}.")

    @classmethod
    def unregister_solver(cls, name: str):
        if name not in cls.solver_registry:
            raise KeyError(f"Solver '{name}' is not registered in {cls.__name__}.")
        del cls.solver_registry[name]

    @classmethod
    def register_condition(cls, name: str, func: 'ConditionLike', overwrite: bool = False):
        # Validate if a condition with the same name already exists and overwrite is not allowed
        if name in cls.condition_registry and not overwrite:
            raise ValueError(
                f"Condition '{name}' is already registered in class {cls.__name__}. Use 'overwrite=True' to replace it.")

        # Type check: If func is not already a Condition or a callable instance
        if not isinstance(func, Condition) and callable(func):
            # Wrap the function into a Condition
            func = Condition(func)
        elif not isinstance(func, Condition):
            raise TypeError(f"Expected a callable or a Condition instance, but got {type(func).__name__}.")

        # Register or overwrite the condition
        cls.condition_registry[name] = func

    @classmethod
    def unregister_condition(cls, name: str):

        if name not in cls.condition_registry:
            raise KeyError(f"Condition '{name}' is not registered in class {cls.__name__}.")
        del cls.condition_registry[name]

    @classmethod
    def get_condition(cls, name: str):
        try:
            return cls.condition_registry[name]
        except KeyError:
            raise KeyError(f"Class {cls.__name__} has no condition registered under name {name}.")

    @classmethod
    def get_solver(cls, name: str):
        try:
            return cls.solver_registry[name]
        except KeyError:
            raise KeyError(f"Class {cls.__name__} has no solver registered under name {name}.")

_ST = TypeVar('ST', bound=ScratchSpaceHandler)
class Pipeline(metaclass=PipelineMeta):
    # -- CLASS SETTINGS -- #
    # These are settings which may be modified in custom pipeline subclasses.
    # See the class documentation for an explanation of each of them.
    DEFAULT_SCRATCH_CLASS: Type[_ST] = ScratchSpaceHandler
    SCRATCH_GROUP_NAME: str = "SCRATCH"

    # Setting up fixed class attributes.
    logger = PipelineLogDescriptor()

    def __init__(self,
                 tasks: Dict[str, Solver] = None,
                 procedure: Dict[str, list[Tuple[str,'ConditionLike']]] = None,
                 model: 'ClusterModel' = None,
                 __handle__: h5py.Group = None,
                 __scratch__: ScratchSpaceHandler = None,
                 overwrite: bool = False):
        # Setting up the core network properties. This includes
        # initializing the tasks and edges.
        self.logger.info("Initializing pipeline...")
        self._initialize_tasks(tasks)
        self._initialize_procedure(procedure)

        # Managing additional attributes
        self._handle = __handle__
        self._scratch = __scratch__
        self._context = None

        # Loading the model
        self._model = None
        self._initialize_model(model, overwrite)

        self._VALIDATION_FLAG = False
        self._is_valid = None
        self._longest_path_map: Dict[str, int] = {}

    def _initialize_tasks(self, tasks: Dict[str, Solver]):
        self.logger.debug("\tInitializing tasks...")

        # Load tasks provided as arguments to the initialization process.
        tasks = tasks if tasks is not None else {}
        self.tasks: TaskManager = TaskManager(self,tasks)

        # Load tasks from the class structure.
        for task_name,task_object in self.__class__.solver_registry.items():
            if task_name in self.tasks:
                continue

            self.tasks[task_name] = task_object

        # Adding the start and stop tasks.
        self.tasks.setdefault('start', NoOpSolver())
        self.tasks.setdefault('end', NoOpSolver())

    def _initialize_procedure(self, procedure: Dict[str,List[Tuple[str,'ConditionLike']]]|None):
        self.logger.debug("\tInitializing pipeline procedure...")

        # Load the user provided procedures from the arguments at instantiation.
        procedure = procedure if procedure is not None else {}

        # Validating the user provided procedures to enforce type standards.
        # We iterate over the entire structure and enforce type constrains on the
        # Condition objects.
        for origin_node,edges in procedure.items():
            end_nodes, conditions = [],[]
            for end_node, condition in edges:
                # Add the ending node to our list of nodes.
                end_nodes.append(end_node)

                # Perform the type checking and coercion process.
                if isinstance(condition, Condition):
                    conditions.append(condition)
                elif callable(condition):
                    conditions.append(Condition(condition))
                elif isinstance(condition, str):
                    condition = self.__class__.condition_registry.get(condition, Condition(lambda _,__,r,_r=condition: r==_r))
                    conditions.append(condition)
                else:
                    conditions.append(Condition(lambda _,__,r,_r=condition: r == _r))

            procedure[origin_node] = [(dn,c) for dn,c in zip(end_nodes,conditions)]
        self.procedure: EdgeManager = EdgeManager(self,procedure)

    def _initialize_model(self,model: 'ClusterModel',overwrite):
        self.logger.debug("\tInitializing the model...")
        if model is not None:
            model.set_pipeline(self,overwrite=overwrite)

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}(nodes:{len(self.tasks)})>"

    def __repr__(self) -> str:
        return self.__str__()

    def __call__(self, grid: 'Grid', start: Optional[str] = 'start', max_steps: Optional[int] = np.inf) -> None:
        self.logger.info(f"[EXEC] Running pipeline on single grid: {grid}")
        # Call setup and validation. Here we need to check that there is a model to run on
        # and that we have a valid pipeline.
        if not self.validate_pipeline(require_setup=True, force=False, validate_tasks=True):
            raise PipelineError(f"Pipeline {self} failed pre-call validation. This likely indicates that this pipeline is"
                                f" either incomplete at the time it was called or that there was some verification process in"
                                f" the pipeline which failed. Please review the structure and constituent solvers.")
        if self._model is None:
            raise PipelineError("Pipelines CANNOT be run without a valid model. Please ensure that a model is specified for the"
                                " pipeline.")

        # Setup the runtime context. This is largely a question of setting up the progress bars
        # configuring max steps, etc.
        task = start
        step_count = 0
        self._context = {}

        with logging_redirect_tqdm(loggers=[self._model.grid_manager.logger, self.logger]):
            while task != 'end' and step_count < max_steps:
                step_count += 1
                start_time = perf_counter()

                # Find the solver and execute it on the grid.
                _solver = self.tasks[task]
                try:
                    result = _solver(self, grid)
                except Exception as e:
                    raise PipelineError(f"Error in task '{task}' for grid '{grid}': {e}")


                # From the result of the execution above, determine what the next step is.
                # We simply search all branches from the current position in the network.
                next_task_found = False
                previous_task = task
                for next_task, condition in self.procedure.get(task, []):
                    if condition(self, grid, result):
                        self.logger.trace(f"[EXEC] TASK={task}, GRID={grid}, RESULT={result}, NEXT_TASK={next_task}")
                        task = next_task
                        next_task_found = True
                        break
                # If no next task was found, terminate
                if not next_task_found:
                    if task == 'end':
                        break
                    else:
                        raise PipelineError(f"Cannot proceed from task '{task}' with result '{result}'.")

                # Log time taken for the task
                elapsed_time = perf_counter() - start_time
                self.logger.info(f"[EXEC] Completed task '{previous_task}' in {elapsed_time:.4f} s")
            self.logger.info(f"[EXEC] Finished pipeline for grid: {grid}")

    def __len__(self) -> int:
        return len(self.tasks)

    def __contains__(self, task_name: str) -> bool:
        return task_name in self.tasks

    def __getitem__(self, task_name: str) -> Optional[Solver]:
        return self.tasks[task_name]

    def __setitem__(self, task_name: str, task: Optional[Solver] = None) -> None:
        self.tasks[task_name] = task

    def __delitem__(self,task_name: str) -> None:
        # Delete the note from the tasks. We raise an error here if
        # it doesn't exist.
        if task_name in self.tasks:
            del self.tasks[task_name]
        else:
            raise KeyError(f"Cannot delete {task_name}. It is not present in {self}.")

        # Now remove it from procedures as well.
        if task_name in self.procedure:
            del self.procedure[task_name]

        for k,v in self.procedure.items():
            self.procedure[k] = [_v for _v in v if _v[0] != task_name]

    def __iter__(self) -> iter:
        """Iterate over the fields in the pipeline."""
        return iter(self.tasks)

    def validate_pipeline(self,require_setup: bool = False, force: bool = False, validate_tasks: bool = False) -> bool:
        if self._VALIDATION_FLAG and not force:
            return self._is_valid

        self.logger.debug("[VAL ] Validating...")
        result = True
        if validate_tasks:
            result= result & self.tasks.validate_all_tasks(overwrite=force,setup_if_needed=require_setup)

        # Now validate the pipeline itself.
        result = result & self.procedure.has_connection('start','end')

        self._VALIDATION_FLAG = True
        self._is_valid = result
        return result

    def ensure_setup(self,force: bool = False):
        # Iterate through each task and run the setup procedure.
        self.logger.debug("[VAL ] Setting Up...")
        for task_name,task in self.tasks.items():
            self.logger.trace("[STUP] Task=%s",task_name)
            task.setup(self,overwrite=force)
    @property
    def handle(self) -> h5py.Group:
        if self._handle is None:
            raise PipelineHandleNotFoundError(f"Pipeline {self} does not have an attached handle. To set it, save the pipeline to"
                                              " hdf5.")
        return self._handle
    @handle.setter
    def handle(self,value):
        if self._model is not None:
            raise ValueError("Cannot set handle while model is set.")

        self._handle = value

    @property
    def scratch(self) -> _ST:
        if self._scratch is None:
            raise MissingScratchHandlerError(f"Pipeline {self} does not have a scratch space configured. To configure a scratch"
                                             f" space, you must first setup scratch with .setup_scratch()")

        return self._scratch

    @property
    def model(self) -> 'ClusterModel':
        return self._model

    def set_model(self,model):
        if self._model is not None:
            self.unlink_model()

        model.set_pipeline(self)

    def unlink_model(self,delete_data=False):
        if self._model is None:
            raise IOError("There is no model to unlink")

        self._model.remove_pipeline(delete_data=delete_data,delete_pipeline=False)

    def setup_scratch(self,__scratch_class__: Type[_ST] =None,overwrite: bool = False) -> _ST:
        # Validating arguments and dealing with the overwrite logic.
        self.logger.info(f"Setting up scratch space for %s.",self)
        if __scratch_class__ is None:
            __scratch_class__ = self.__class__.DEFAULT_SCRATCH_CLASS

        if self._scratch is not None and not overwrite:
            raise PipelineError(f"Cannot set scratch handler for {self} because it already has a handler. Use overwrite=True to"
                                f" enable overwrite.")
        elif self._scratch is not None:
            self._scratch.delete()
            self._scratch = None
        else:
            pass

        # Initializing the scratch class.
        # This AUTOMATICALLY sets self._scratch to the expected value.
        __scratch_class__(self)
        return self.scratch

    def to_hdf5(self,handle: h5py.Group = None, set_handle: bool = True):
        # Validate the handle input.
        if handle is None:
            handle = self.handle

        if handle is None:
            raise IOError(f"Cannot write {self} to HDF5 because it has no handle and __handle__ was not specified.")

        # Writing components to HDF5
        self.tasks.to_hdf5(handle)
        self.procedure.to_hdf5(handle)

        if set_handle:
            self._handle = handle

    @classmethod
    def from_hdf5(cls,handle: h5py.Group,*args,**kwargs):
        tasks,procedure = TaskManager.from_hdf5(handle),EdgeManager.from_hdf5(handle)
        return cls(tasks,procedure,*args,__handle__=handle,**kwargs)

    def keys(self):
        return self.tasks.keys()

    def values(self):
        return self.tasks.values()

    def items(self):
        return self.tasks.items()

