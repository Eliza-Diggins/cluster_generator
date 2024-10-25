import logging
from typing import TYPE_CHECKING, List, Optional, Tuple
from cluster_generator.utilities.config import cgparams
from cluster_generator.utilities.logging import LogDescriptor
from cluster_generator.pipelines.solvers import Solver
from cluster_generator.pipelines.conditions import Condition
import h5py
from ._except import MissingTaskError
import numpy as np
if TYPE_CHECKING:
    from .abc import Pipeline
    from cluster_generator.pipelines.conditions._types import ConditionLike

# Defining type annotations for use throughout the module.
FieldAlias = str
FieldOrder = List[FieldAlias]

logging.addLevelName(5,"TRACE")

class PipelineLogger(logging.Logger):

    def __init__(self, name, level=logging.DEBUG):
        super().__init__(name, level)


    def trace(self, msg, *args, **kwargs):
        """
        Logs a message with TRACE level.
        """
        if self.isEnabledFor(5):
            self.log(5, msg, *args, **kwargs)

    def set_default_handler(self, formatter: Optional[logging.Formatter] = None):
        """
        Sets up a StreamHandler for console output.
        """
        if not self.hasHandlers():
            handler = logging.StreamHandler()
            handler.setLevel(self.level)

            # Use a provided formatter or default one
            formatter = formatter or logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.addHandler(handler)
class PipelineLogDescriptor(LogDescriptor):
    LOG_CLASS = PipelineLogger
    def configure_logger(self, logger):
        _handler = logging.StreamHandler()
        _handler.setFormatter(
            logging.Formatter(cgparams["logging"]["mylog"]["format"])
        )
        if len(logger.handlers) == 0:
            logger.addHandler(_handler)

class TaskManager(dict):
    """
    A TaskManager that holds a collection of tasks (Solvers) identified by a string key.
    It provides methods to add, remove, and validate tasks, as well as access them like a dictionary.
    """
    def __init__(self,pipeline: 'Pipeline', *args,**kwargs):
        # Generate the dictionary containing the tasks.
        super().__init__(*args, **kwargs)

        # Connect the pipeline.
        self.pipeline = pipeline
    def __setitem__(self, key: str, value: Solver):
        """
        Sets a task in the TaskManager.

        Parameters
        ----------
        key : str
            The name of the task.
        value : Solver
            The solver instance to assign.
        """
        if not isinstance(value, Solver):
            raise TypeError(f"Value must be a Solver instance, got {type(value)}.")
        super().__setitem__(key, value)

    def __delitem__(self, key: str):
        """
        Deletes a task from the TaskManager.

        Parameters
        ----------
        key : str
            The name of the task to delete.

        Raises
        ------
        KeyError
            If the task does not exist.
        """
        super().__delitem__(key)

    def __str__(self) -> str:
        """
        Provides a user-friendly string representation of the TaskManager.

        Returns
        -------
        str
            A concise description of the TaskManager including task names.
        """
        return f"<TaskManager | N={len(self)}>"

    def __repr__(self) -> str:
        """
        Provides a detailed string representation of the TaskManager.

        Returns
        -------
        str
            Detailed string representation of the TaskManager including tasks.
        """
        task_details = ", ".join([f"{name}: {task.__class__.__name__}" for name, task in self.items()])
        return f"<TaskManager({task_details})>"

    def add_task(self, task_name: str, task: Solver, overwrite: bool = False):
        # Validating that the task is legitimate.
        if task_name in self and not overwrite:
            raise ValueError(f"A task with the name '{task_name}' already exists. Use overwrite=True to replace it.")

        if not callable(task):
            raise TypeError(f"Cannot add task: must be callable, not {type(task)}.")

        self[task_name] = task

    def remove_task(self, task_name: str):
        if task_name in self:
            del self[task_name]
        else:
            raise KeyError(f"Task '{task_name}' does not exist in the TaskManager.")

    def validate_task(self, task_name: str, overwrite: bool = False, setup_if_needed: bool = False) -> bool:
        task: Solver = self[task_name]
        try:
            return task.validate(self.pipeline,overwrite=overwrite,setup_if_needed=setup_if_needed)
        except Exception as e:
            raise e

    def validate_all_tasks(self, overwrite: bool = False, setup_if_needed: bool = False) -> bool:
        r = []
        for task_name,task in self.items():
            self.pipeline.logger.trace("[VAL ] TASK=%s",task_name)

            try:
                r.append(self.validate_task(task_name,setup_if_needed=setup_if_needed,overwrite=overwrite))
            except Exception as e:
                r.append(False)

            if r[-1] == False:
                self.pipeline.logger.warning("Failed to validate %s",task_name)

        return all(r)



    def to_hdf5(self, handle: h5py.Group|h5py.File):
        header_group = handle.require_group("TASKS")

        for task_name, _solver in self.items():
            # Check if a task is a method in the pipeline.
            if task_name in self.pipeline.__class__.solver_registry:
                continue
            if task_name in ['start','end']:
                continue

            task_group = header_group.require_group(task_name)
            task_group.attrs['solver_present'] = _solver is not None

            if _solver is not None:
                _solver.to_hdf5(task_group)

    @classmethod
    def from_hdf5(cls, handle: h5py.Group):
        # Validating the handle that was passed.
        header_grp = handle["TASKS"]
        tasks = {}

        for task_name, task_grp in header_grp.items():
            solver_present = task_grp.attrs.get("solver_present", False)

            # If solver is present, load it; otherwise, use None
            if solver_present:
                tasks[task_name] = Solver.from_hdf5(task_grp)
            else:
                tasks[task_name] = None

        return tasks

class EdgeManager(dict):
    """
    Manages the connections (edges) between tasks in a pipeline.
    Each key is a task name, and the value is a list of tuples representing
    the destination tasks and the conditions under which they are triggered.
    """

    def __init__(self, pipeline: 'Pipeline', *args, **kwargs):
        """
        Initialize the EdgeManager with an associated pipeline.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline that this EdgeManager is associated with.
        """
        super().__init__(*args, **kwargs)
        self.pipeline = pipeline
        self._longest_path_map = {}

    def add_edge(self, from_task: str, to_task: str, condition: 'ConditionLike'):
        """
        Adds an edge between two tasks, with a specified condition.

        Parameters
        ----------
        from_task : str
            The task where the edge starts.
        to_task : str
            The task where the edge points to.
        condition : 'ConditionLike'
            The condition under which the edge is traversed.

        Raises
        ------
        MissingTaskError
            If the source or destination tasks are not found in the pipeline.
        """
        # Validating.
        if from_task not in self.pipeline.tasks:
            raise MissingTaskError(f"Task '{from_task}' not found in the pipeline.",from_task)
        if to_task not in self.pipeline.tasks:
            raise MissingTaskError(f"Task '{to_task}' not found in the pipeline.",to_task)

        if not callable(condition):
            raise TypeError(f"Condition must be callable, got {type(condition)}.")

        # If no edges exist for the `from_task`, initialize an empty list
        if from_task not in self:
            self[from_task] = []

        # Add the new edge
        self[from_task].append((to_task, condition))
        self._longest_path_map = {}

    def remove_edge(self, from_task: str, to_task: str):
        """
        Removes an edge between two tasks.

        Parameters
        ----------
        from_task : str
            The task where the edge starts.
        to_task : str
            The task where the edge points to.

        Raises
        ------
        ValueError
            If the edge does not exist.
        """
        if from_task not in self or not any(edge[0] == to_task for edge in self[from_task]):
            raise ValueError(f"Edge from '{from_task}' to '{to_task}' does not exist.")

        # Remove the edge
        self[from_task] = [edge for edge in self[from_task] if edge[0] != to_task]
        self._longest_path_map = {}

    def get_edges(self, task_name: str) -> List[Tuple[str, 'ConditionLike']]:
        """
        Retrieves all outgoing edges from a given task.

        Parameters
        ----------
        task_name : str
            The name of the task whose outgoing edges are retrieved.

        Returns
        -------
        List[Tuple[str, 'ConditionLike']]
            A list of (to_task, condition) tuples representing the outgoing edges.
        """
        return self.get(task_name, [])

    def __str__(self) -> str:
        """
        Provides a user-friendly string representation of the EdgeManager.

        Returns
        -------
        str
            A concise description of the EdgeManager including tasks and edges.
        """
        return f"<EdgeManager | {len(self)}>"

    def __repr__(self) -> str:
        """
        Provides a detailed string representation of the EdgeManager.

        Returns
        -------
        str
            Detailed string representation of the EdgeManager including tasks and edges.
        """
        edge_details = ", ".join([f"{from_task}: {len(edges)} edges" for from_task, edges in self.items()])
        return f"<EdgeManager({edge_details})>"

    def to_hdf5(self, handle: h5py.Group):
        """
        Saves the edge information to an HDF5 file.

        Parameters
        ----------
        handle : h5py.Group
            HDF5 file handle for saving edge data.
        """
        header_group = handle.require_group("TASKS")

        for start_edge, edges in self.items():

            task_group = header_group.require_group(start_edge)
            task_group.attrs['edges_nodes'] = [_e[0] for _e in edges]
            Condition.from_list_to_hdf5(task_group,[_e[1] for _e in edges])

    @classmethod
    def from_hdf5(cls, handle: h5py.Group):
        """
        Loads edge information from an HDF5 file and reconstructs the EdgeManager.

        Parameters
        ----------
        pipeline : Pipeline
            The associated pipeline.
        handle : h5py.Group
            HDF5 file handle for loading edge data.

        Returns
        -------
        EdgeManager
            The reconstructed EdgeManager instance.
        """
        _edge_manager = {}
        header_group = handle["TASKS"]

        for task_name, task_grp in header_group.items():
            edge_names = task_grp.attrs.get('edges_nodes',[])
            edges_conditions = Condition.from_hdf5_to_list(task_grp)

            _edge_manager[task_name] = [(en,ec) for en,ec in zip(edge_names, edges_conditions)]

        return _edge_manager

    def has_connection(self, start: str, end: str) -> bool:
        # Validation step. We simply check that the start exists. The end might not
        # exist if it has no downstream procedures.
        if start not in self:
            raise ValueError(f"Task {start} is not present in the pipeline procedure. "
                             f"This either indicates that it has no links or that it does not "
                             f"exist at all.")

        # Proceed with depth-first search. This should be sufficient for a network
        # of this size. These should never be complicated enough for issues to arise.
        stack = [start]
        visited = set()

        while stack:
            current_task = stack.pop()
            if current_task == end:
                return True
            if current_task in visited:
                continue
            visited.add(current_task)
            # Add all unvisited downstream tasks to the stack
            for next_task, _ in self.get(current_task, []):
                if next_task not in visited:
                    stack.append(next_task)

        return False

    def calculate_longest_path(self,task: str, max_recursion=99):
        # Validate that the task is actually in the map.
        if task not in self:
            raise ValueError(f"{self} failed to compute the longest path for {task} because it doesn't appear to have"
                             " any downstream edges.")

        # Run until the maximum recursion depth occurs.
        longest_path = self._calc_longest_path_rec(task,0,max_recursion)

        if longest_path < 0:
            raise ValueError(f"Failed to find a valid path from {task} to end.")

        return longest_path

    def _calc_longest_path_rec(self,task, rc_depth, rc_lim):
        # Perform simple validations. If we've overstepped the recursion depth, the task
        # is the end node, or we have a value, we can return immediately.
        if rc_lim < rc_depth:
            raise RecursionError(f"{self} failed to compute longest path because it reached the recursion depth.")

        if task == 'end':
            return 0
        elif task in self._longest_path_map:
            return self._longest_path_map[task]
        elif not len(self[task]):
            return -1

        # Initialize the max_length variable
        max_length = 0
        for next_task, _ in self[task]:
            # Recursively calculate the longest path from the downstream task
            path_length = 1 + self._calc_longest_path_rec(next_task,rc_depth+1,rc_lim)
            # Update max_length if a longer path is found
            max_length = max(max_length, path_length)

        # Cache the calculated longest path for the current task
        self._longest_path_map[task] = max_length

        return max_length
