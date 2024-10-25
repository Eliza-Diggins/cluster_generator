"""
Module for defining and managing conditions in pipeline-based systems.

This module provides the `Condition` class, which represents a condition
used to determine whether a specific state in a pipeline is valid. The
`Condition` class supports logical operations between conditions and
provides functionality for serializing/deserializing conditions to and
from various formats, including HDF5 and binary files.

Conditions can be combined using logical AND, OR, XOR, and NOT operations.
They can also be serialized to bytes using the `dill` protocol, and stored
in binary or HDF5 formats. Deserialization methods allow for reloading
conditions into pipeline systems.

Serialization exceptions are handled by custom error classes.
"""
from abc import ABC
from typing import Any, List, Tuple, TYPE_CHECKING, Callable, Collection
from cluster_generator.pipelines.conditions._types import ConditionLike
from cluster_generator.pipelines.conditions._except import ConditionError, ConditionSerializationError
import dill
import h5py
import numpy as np
import os

if TYPE_CHECKING:
    from cluster_generator.grids.grids import Grid
    from cluster_generator.pipelines.abc import Pipeline

class Condition(ABC):
    """
    Represents a callable condition used in pipeline-based systems.

    The `Condition` class defines a condition that determines whether a
    specific state in a pipeline is valid. Conditions are callable objects,
    and they can be combined with other conditions using logical operators
    such as AND, OR, XOR, and NOT.

    Additionally, conditions can be serialized and deserialized using the
    `dill` library. This allows conditions to be stored in binary or HDF5
    formats and reloaded when needed.
    """

    def __init__(self, condition: ConditionLike):
        """
        Initialize a Condition object.

        A condition is a callable object that checks whether a specific state in a pipeline
        is valid. If the condition is not a callable, an error is raised.

        Parameters
        ----------
        condition : ConditionLike
            A function or callable object that defines the condition logic.

        Raises
        ------
        ConditionError
            If the provided condition is not callable.
        """
        # Validate that the condition is callable.
        if not callable(condition):
            raise ConditionError(f"Condition {condition} (type={type(condition)}) is not a callable object."
                                     f" Direct initialization of a Condition class requires a function with signature"
                                     f" func(pipeline,grid,result) -> bool.")

        self.condition: ConditionLike = condition

    def __call__(self, pipeline: 'Pipeline',grid:'Grid', result: Any) -> bool:
        """
        Determines whether the condition is met based on the pipeline and result.

        This method executes the condition logic to check if the condition is satisfied.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline instance currently executing.
        grid : Grid
            The grid instance associated with the pipeline.
        result : Any
            The result from the previous task's solver in the pipeline.

        Returns
        -------
        bool
            True if the condition is met, False otherwise.
        """
        return self.condition(pipeline,grid,result)

    def _do_op(self, other: 'Condition', op: Callable[[bool, bool], bool]) -> 'Condition':
        """
        Apply a logical operation between two conditions.

        This method combines the current condition with another condition using a logical operation,
        such as AND, OR, or XOR.

        Parameters
        ----------
        other : Condition
            The other condition to apply the operation to.
        op : Callable[[bool, bool], bool]
            The logical operation to apply.

        Returns
        -------
        Condition
            A new Condition object representing the result of the logical operation.

        Raises
        ------
        ConditionError
            If the `other` object is not a Condition instance.
        """
        if not isinstance(other, Condition):
            raise ConditionError("The operation can only be performed between Condition instances.")

        return Condition(lambda pipeline, grid, result: op(self(pipeline, grid, result), other(pipeline, grid, result)))

    def __and__(self, other: 'Condition') -> 'Condition':
        """
        Combine two conditions with a logical AND operation.

        Parameters
        ----------
        other : Condition
            The other condition to combine with.

        Returns
        -------
        Condition
            A new Condition that is True if both conditions are met.
        """
        return self._do_op(other, lambda x, y: x and y)

    def __or__(self, other: 'Condition') -> 'Condition':
        """
        Combine two conditions with a logical OR operation.

        Parameters
        ----------
        other : Condition
            The other condition to combine with.

        Returns
        -------
        Condition
            A new Condition that is True if either condition is met.
        """
        return self._do_op(other, lambda x, y: x or y)

    def __xor__(self, other: 'Condition') -> 'Condition':
        """
        Combine two conditions with a logical XOR (exclusive OR) operation.

        Parameters
        ----------
        other : Condition
            The other condition to combine with.

        Returns
        -------
        Condition
            A new Condition that is True if one, but not both, conditions are met.
        """
        return self._do_op(other, lambda x, y: x ^ y)

    def __invert__(self) -> 'Condition':
        """
        Return the logical NOT of the condition.

        This negates the current condition, making it return True when it would have returned False, and vice versa.

        Returns
        -------
        Condition
            A new Condition that is the inverse of the current condition.
        """
        return Condition(lambda pipeline, grid, result, _f=self.condition: not _f(pipeline, grid, result))

    def to_dill(self) -> bytes:
        """
        Serialize the condition using the `dill` serialization protocol.

        .. warning::
           This may cause issues if you try to re-initialize from bytes in a different environment or across systems.

        Returns
        -------
        bytes
            The serialized condition as bytes.
        """
        return dill.dumps(self)

    @classmethod
    def from_dill(cls, data: bytes) -> 'Condition':
        """
        Deserialize a condition from bytes using dill.

        Parameters
        ----------
        data : bytes
            The serialized condition data.

        Returns
        -------
        Condition
            The deserialized Condition instance.
        """
        return dill.loads(data)

    @staticmethod
    def serialize_conditions(conditions: Collection['Condition']) -> Tuple[bytes, List[Tuple[int, int]]]:
        """
        Serialize a collection of conditions into bytes and track their offsets.

        This method serializes multiple conditions and returns a byte string along with a list of offsets.

        Parameters
        ----------
        conditions : Collection[Condition]
            A collection of Condition objects to serialize.

        Returns
        -------
        Tuple[bytes, List[Tuple[int, int]]]
            A tuple containing the serialized byte string and a list of (start, end) offsets for each condition.

        Raises
        ------
        ConditionSerializationError
            If any condition fails to serialize.
        """
        serialized_conditions = bytearray()
        offsets = []

        for condition in conditions:
            # Determine the starting offset.
            start_offset = len(serialized_conditions)

            # Add the serialized bytes to the conditions.
            try:
                condition_data = dill.dumps(condition) if condition else dill.dumps(None)
            except (dill.PicklingError, TypeError) as e:
                raise ConditionSerializationError(f"Failed to serialize condition {condition}: {e}")
            serialized_conditions.extend(condition_data)

            # Find the ending offset and add everything to the results.
            end_offset = len(serialized_conditions)
            offsets.append((start_offset, end_offset))

        return bytes(serialized_conditions), offsets

    @classmethod
    def deserialize_conditions(cls,serialized_data: bytes, offsets: List[Tuple[int, int]]) -> List['Condition']:
        """
        Deserialize a list of conditions from bytes using dill and offsets.

        This method takes a serialized byte string and the corresponding offsets to reconstruct the conditions.

        Parameters
        ----------
        serialized_data : bytes
            The serialized byte string containing the conditions.
        offsets : List[Tuple[int, int]]
            A list of (start, end) offsets for each condition in the serialized byte string.

        Returns
        -------
        List[Condition]
            A list of deserialized Condition objects.

        Raises
        ------
        ConditionSerializationError
            If any condition fails to deserialize.
        """
        # Create container to store located conditions in.
        conditions = []

        # Seek each conditional offset.
        for start, end in offsets:
            try:
                condition = cls.from_dill(serialized_data[start:end])
            except (dill.UnpicklingError, EOFError, TypeError) as e:
                raise ConditionSerializationError(f"Failed to deserialize condition at offset {start}:{end}: {e}")

            conditions.append(condition)

        return conditions

    @classmethod
    def from_list_to_hdf5(cls, hdf_group: h5py.Group, conditions: Collection['Condition']):
        """
        Save a list of conditions to an HDF5 group.

        This method serializes multiple conditions and stores them in the specified
        HDF5 group as two datasets: 'condition_bytes' and 'condition_offsets'.

        Parameters
        ----------
        hdf_group : h5py.Group
            The HDF5 group or file where the conditions will be serialized.
        conditions : Collection[Condition]
            The list of Condition objects to serialize and store.

        Raises
        ------
        ConditionSerializationError
            If serialization fails for any condition.
        """
        import base64
        # Serialize the list of conditions
        try:
            condition_bytes, offsets = cls.serialize_conditions(conditions)
            # Base64-encode the byte string to avoid any NULL characters
            encoded_data = base64.b64encode(condition_bytes).decode('ascii')
            offsets_array = np.array(offsets, dtype=np.int64)

            # Save the base64-encoded conditions and their offsets
            hdf_group.create_dataset("condition_bytes", data=encoded_data)
            hdf_group.create_dataset("condition_offsets", data=offsets_array)

        except Exception as e:
            raise ValueError(f"Error serializing conditions: {e}")

    @classmethod
    def from_hdf5_to_list(cls, hdf_group: h5py.Group) -> List['Condition']:
        """ Load a list of conditions from an HDF5 group. """
        import base64
        try:
            # Retrieve base64-encoded data and decode it
            encoded_data = hdf_group["condition_bytes"][()].decode('ascii')
            condition_bytes = base64.b64decode(encoded_data)
            condition_offsets = hdf_group["condition_offsets"][:]

            # Ensure the offsets are in the correct format
            if condition_offsets.shape[1] != 2:
                raise ValueError("Offsets array must be 2-dimensional with start and end values.")

            # Deserialize the conditions using the offsets
            offsets = [(int(start), int(end)) for start, end in condition_offsets]
            return cls.deserialize_conditions(condition_bytes, offsets)

        except KeyError as e:
            raise ValueError(f"Missing expected dataset in HDF5 group: {e}")
        except Exception as e:
            raise ValueError(f"Error deserializing conditions: {e}")

    def to_hdf5(self, hdf_group: h5py.Group):
        """
        Save a single condition to an HDF5 group.

        This method serializes the condition and stores it in the specified
        HDF5 group as 'condition_bytes' and 'condition_offsets'.

        Parameters
        ----------
        hdf_group : h5py.Group
            The HDF5 group or file where the condition will be serialized.
        """
        Condition.from_list_to_hdf5(hdf_group, [self])

    @classmethod
    def from_hdf5(cls, hdf_group: h5py.Group) -> List['Condition']:
        """
        Load conditions from an HDF5 group.

        This method reads the serialized condition bytes and offsets from the specified
        HDF5 group, deserializes them, and returns a list of Condition objects.

        Parameters
        ----------
        hdf_group : h5py.Group
            The HDF5 group or file from which the conditions will be deserialized.

        Returns
        -------
        List[Condition]
            The list of deserialized Condition objects.
        """
        return cls.from_hdf5_to_list(hdf_group)

    def to_binary(self, file_path: str, overwrite: bool = False):
        """
        Save a single condition to a binary file.

        This method serializes the condition using dill and writes the bytes to a file.

        Parameters
        ----------
        file_path : str
            The file path where the condition will be saved.
        overwrite : bool, optional
            If `True`, allows overwriting an existing file. If `False` (default),
            raises an error if the file already exists.

        Raises
        ------
        FileExistsError
            If the file already exists and `overwrite` is `False`.
        """
        # Check if file exists and overwrite is not allowed
        if os.path.exists(file_path) and not overwrite:
            raise FileExistsError(f"The file '{file_path}' already exists. Use overwrite=True to replace it.")

        # Write the serialized condition to the binary file
        with open(file_path, 'wb') as f:
            f.write(self.to_dill())

    @classmethod
    def from_binary(cls, file_path: str) -> 'Condition':
        """
        Load a single condition from a binary file.

        This method reads the serialized condition bytes from a file and deserializes
        it using dill.

        Parameters
        ----------
        file_path : str
            The file path from which the condition will be loaded.

        Returns
        -------
        Condition
            The deserialized Condition object.
        """
        with open(file_path, 'rb') as f:
            condition_bytes = f.read()
        return cls.from_dill(condition_bytes)


