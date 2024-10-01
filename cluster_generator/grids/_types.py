"""
Grid Types Module
===================

This module provides various utility classes for managing errors, containers, and type hints throughout the
:py:mod:`grids` module.

Notes
-----
This module is designed to support the management of multi-dimensional grids and their hierarchical
relationships. It leverages HDF5 for storage and lazy loading, allowing efficient handling of large
datasets that may not fit entirely in memory.

See Also
--------
- `h5py <https://www.h5py.org/>`_ : Python interface to the HDF5 binary data format.
- `numpy <https://numpy.org/>`_ : A fundamental package for scientific computing with Python.
- `unyt <https://unyt.readthedocs.io/>`_ : A package for handling, manipulating, and converting physical quantities with units.

"""
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Collection, Generic, Type, TypeVar, Union

import numpy as np
from h5py import Dataset
from numpy.typing import NDArray
from unyt import Unit

if TYPE_CHECKING:
    pass

# Type Aliases for readability
# Type Aliases for readability

ArrayAlias = Dataset
"""
Alias for the h5py Dataset type, representing a multi-dimensional array stored in an HDF5 file.
Used for handling data arrays within the HDF5-backed grid containers.
"""

UnitAlias = Union[Unit, str]
"""
Alias for the unit type, allowing the use of both unyt `Unit` objects and string representations.
Used to specify units of measurement for physical quantities in fields and metadata.
"""

DtypeAlias = Union[np.dtype, str, type]
"""
Alias for data type representation, accommodating numpy data types, string names of types,
or Python type objects. Used to specify the data type of arrays and datasets.
"""


# Type aliases
DomainShape = Union[Collection[int], NDArray[int]]
"""
Alias for shapes of grid domains, allowing them to be represented as lists, tuples,
or numpy arrays of integers. Used to define the dimensionality and size of grid structures.
"""

BoundingBox = Union[NDArray[float], Collection[float]]
"""
Alias for bounding boxes, allowing them to be represented as lists, tuples,
or numpy arrays of floats/ints. Used to define the spatial bounds of a grid structure.
"""

# Utility functions for coercion


def coerce_to_domain_shape(domain_shape: Any) -> NDArray[int]:
    """
    Coerce any input to a NumPy array of integers representing the domain shape.

    Parameters
    ----------
    domain_shape : Any
        Input that can be coerced into a domain shape (list, tuple, or NumPy array of integers).

    Returns
    -------
    NDArray[int]
        A NumPy array representing the domain shape.

    Raises
    ------
    ValueError
        If the input cannot be coerced into a valid domain shape.
    """
    try:
        # Convert the input to a NumPy array of integers
        domain_shape = np.asarray(domain_shape, dtype=np.int32)

        # Ensure it is 1-dimensional
        if domain_shape.ndim != 1:
            raise ValueError(
                f"Domain shape must be a 1-dimensional array, but got {domain_shape.ndim} dimensions."
            )

        # Ensure all values are positive
        if (domain_shape <= 0).any():
            raise ValueError(
                "All values in the domain shape must be positive integers."
            )

        return domain_shape

    except Exception as e:
        raise ValueError(f"Invalid domain shape: {e}")


def coerce_to_bounding_box(bbox: Any) -> NDArray[float]:
    """
    Coerce any input to a NumPy array representing the bounding box.

    Parameters
    ----------
    bbox : Any
        Input that can be coerced into a bounding box (2xN array of floats or integers).

    Returns
    -------
    NDArray[float]
        A 2xN NumPy array representing the bounding box.

    Raises
    ------
    ValueError
        If the input cannot be coerced into a valid bounding box.
    """
    try:
        # Convert the input to a NumPy array of floats
        bbox = np.asarray(bbox, dtype=np.float64)
        bbox = bbox.reshape(2, bbox.size // 2)
        return bbox

    except Exception as e:
        raise ValueError(f"Invalid bounding box: {e}")


GridLevel = TypeVar("GridLevel")
"""
Generic type variable representing a grid level object.
Used for type hinting in classes and methods that operate on grid levels.
"""

Grid = TypeVar("Grid")
"""
Generic type variable representing a grid object.
Used for type hinting in classes and methods that operate on grids.
"""

T = TypeVar("T")
"""
Generic type variable representing any type.
Used for general-purpose type hinting in the metadata descriptor and other generic classes.
"""

IndexType = TypeVar("IndexType")
"""
Generic type variable representing the index type of elements in a container.
Used for type hinting in element containers to specify the type of index used for accessing elements.
"""

ItemType = TypeVar("ItemType")
"""
Generic type variable representing the type of elements stored in a container.
Used for type hinting in element containers to specify the type of elements being managed.
"""


class GridError(Exception):
    """Base exception class for grid-related errors."""

    pass


class GridMetadataError(GridError):
    """Exception raised for errors in the grid metadata."""

    pass


class GridManagerError(Exception):
    """Base exception for GridManager related errors."""

    pass


class LevelNotFoundError(GridManagerError):
    """Raised when a specified level does not exist."""

    pass


class GridNotFoundError(GridManagerError):
    """Raised when a specified grid does not exist."""

    pass


class FieldNotFoundError(GridManagerError):
    """Raised when a specified field does not exist."""

    pass


# ===================================== #
# Metadata Descriptors                  #
# ===================================== #
class MetadataDescriptor(ABC):
    """
    Base class for metadata descriptors to handle metadata attributes for grid levels or grid managers.

    A `MetadataDescriptor` provides a consistent interface for accessing metadata attributes
    from memory or an HDF5 file, with support for defaults and requirements.

    Parameters
    ----------
    required : bool, optional
        Whether the metadata is required, by default False.
    default : Any, optional
        Default value for the metadata, by default None.

    Notes
    -----
    A `MetadataDescriptor` is designed to be used as a class attribute in classes that manage hierarchical grid data.
    It provides a consistent mechanism to handle metadata attributes such as grid dimensions, bounding boxes, or
    physical units. The descriptor will attempt to retrieve the metadata value from the following sources in order:

    1. **Instance Cache**:
       If the value is pre-loaded in the instance's `_meta` dictionary (accessed via :py:meth:`MetadataDescriptor.get_meta_cache`),
       it will be returned from there.

    2. **HDF5 File**:
       If the instance is HDF5-backed, the descriptor will attempt to load the value from the HDF5 file (accessed via
       :py:meth:`MetadataDescriptor.get_hdf5_ref`). This allows metadata to be stored persistently and loaded on demand.

    3. **Default Value**:
       If the value is not found in the instance or HDF5, and the descriptor has a default value specified,
       the default will be used.

    If the metadata is marked as `required` but is not found in any of these sources, a :py:exc:`GridMetadataError` will be raised.

    This design pattern ensures that metadata is consistently managed across in-memory and on-disk representations,
    making it easier to handle large and complex datasets.

    See Also
    --------
    :py:class:`GridManagerMetadataDescriptor`
        Descriptor for metadata stored in the GridManager's `HEADER` group.
    :py:class:`GridLevelMetadataDescriptor`
        Descriptor for metadata stored directly in the GridLevel's HDF5 group.
    :py:class:`GridMetadataDescriptor`
        Descriptor for metadata stored directly in the Grid's HDF5 group.
    :py:class:`FieldMetadataDescriptor`
        Descriptor for metadata stored in a field's HDF5 dataset attributes.

    Examples
    --------
    .. code-block:: python

        from h5py import File

        class MyMetadataDescriptor(MetadataDescriptor):
            \"\"\"Custom metadata descriptor for demonstration purposes.\"\"\"

            def get_meta_cache(self, instance):
                # Return the metadata cache of the instance.
                return instance._meta

            def get_hdf5_ref(self, instance):
                # Retrieve the HDF5 group reference for the instance.
                return instance._handle['metadata']

        class MyGridManager:
            \"\"\"A simple grid manager class using metadata descriptors.\"\"\"

            # Using MyMetadataDescriptor to handle metadata attributes
            NDIM = MyMetadataDescriptor(required=True, default=3)
            BBOX = MyMetadataDescriptor(required=False, default=[[0, 0, 0], [1, 1, 1]])

            def __init__(self, handle):
                # Initialize the instance with an HDF5 handle and a metadata cache.
                self._handle = handle
                self._meta = {}

        # Example usage
        with File('example.h5', 'w') as hdf5_file:
            # Create a metadata group in the HDF5 file
            metadata_group = hdf5_file.create_group('metadata')
            metadata_group.attrs['NDIM'] = 3
            metadata_group.attrs['BBOX'] = [[0, 0, 0], [1, 1, 1]]

            # Initialize a grid manager instance
            grid_manager = MyGridManager(hdf5_file)

            # Accessing metadata attributes using descriptors
            print(grid_manager.NDIM)  # Outputs: 3
            print(grid_manager.BBOX)  # Outputs: [[0, 0, 0], [1, 1, 1]]

            # Setting new values for metadata attributes
            grid_manager.BBOX = [[-1, -1, -1], [1, 1, 1]]
            print(grid_manager.BBOX)  # Outputs: [[-1, -1, -1], [1, 1, 1]]
    """

    def __init__(self, required: bool = False, default: Any = None):
        """
        Initialize a MetadataDescriptor.

        Parameters
        ----------
        required : bool, optional
            Whether the metadata is required, by default False.
        default : Any, optional
            Default value for the metadata, by default None.
        """
        self.required = required
        self.default = default

    def __set_name__(self, owner: Type[T], name: str):
        """
        Assign the name of the metadata attribute.

        Parameters
        ----------
        owner : Type[T]
            The class type to which the descriptor is being assigned.
        name : str
            The name of the metadata attribute.
        """
        self.name = name

    def __get__(self, instance: T, owner: Type[T]) -> Any:
        """
        Retrieve the metadata value from the instance, HDF5, or default.

        Parameters
        ----------
        instance : T
            The instance of the class holding this metadata.
        owner : Type[T]
            The class type of the instance.

        Returns
        -------
        Any
            The value of the metadata.

        Raises
        ------
        GridMetadataError
            If the metadata is required but cannot be found.
        """
        if instance is None:
            return self.default

        # Retrieve metadata from the instance's cache
        meta_cache = self.get_meta_cache(instance)
        if self.name not in meta_cache:
            # Load from HDF5 or use the default value
            value = self.load_from_hdf5(instance)

            if value is None:
                # use the default.
                value = self.default

            # Raise error if required metadata is missing
            if value is None and self.required:
                raise GridMetadataError(
                    f"Required metadata '{self.name}' is missing for instance {instance}."
                )

            # Cache the metadata value in the instance
            meta_cache[self.name] = value

        return meta_cache[self.name]

    def __set__(self, instance: T, value: Any):
        """
        Set the metadata value in both memory and HDF5 if applicable.

        Parameters
        ----------
        instance : T
            The instance of the class holding this metadata.
        value : Any
            The value to set for the metadata.
        """
        meta_cache = self.get_meta_cache(instance)
        meta_cache[self.name] = value
        self.write_to_hdf5(instance, value)

    def load_from_hdf5(self, instance: T) -> Any:
        """
        Load metadata from the HDF5 file.

        Parameters
        ----------
        instance : T
            The instance of the class holding this metadata.

        Returns
        -------
        Any
            The value of the metadata from the HDF5 file, or None if not present.
        """
        hdf5_ref = self.get_hdf5_ref(instance)
        if hdf5_ref is not None:
            return hdf5_ref.attrs.get(self.name)
        return None

    def write_to_hdf5(self, instance: T, value: Any) -> None:
        """
        Write metadata to the HDF5 file.

        Parameters
        ----------
        instance : T
            The instance of the class holding this metadata.
        value : Any
            The value to write for the metadata.

        Raises
        ------
        IOError
            If the HDF5 handle is not available for writing.
        """
        hdf5_ref = self.get_hdf5_ref(instance)
        if hdf5_ref is not None:
            hdf5_ref.attrs[self.name] = value
        else:
            raise IOError(
                f"Failed to write {self.name} attribute because HDF5 handle is not available."
            )

    def set_metadata_from_kwargs_or_hdf5(self, instance: T, kwargs: dict):
        """
        Set the metadata value from either kwargs or HDF5.

        Prioritizes metadata values provided in the kwargs (typically from user input during initialization).
        If present in kwargs, it is used and saved to HDF5 (if applicable). If not found in kwargs,
        attempts to load the metadata from the HDF5 file. If the value cannot be found in HDF5 and
        the descriptor has a default value, the default will be used.

        Parameters
        ----------
        instance : T
            The instance of the class (e.g., GridManager) using this descriptor.
        kwargs : dict
            Keyword arguments passed to the GridManager's `__init__` method, potentially containing metadata values.
        """
        meta_cache = self.get_meta_cache(instance)
        if self.name in kwargs:
            # Use the value from kwargs if provided
            value = kwargs[self.name]
            meta_cache[self.name] = value
            self.write_to_hdf5(instance, value)
        else:
            # Load from HDF5 or set the default value
            value = self.load_from_hdf5(instance)
            if value is None and self.default is not None:
                meta_cache[self.name] = self.default
                self.write_to_hdf5(instance, self.default)

    @abstractmethod
    def get_meta_cache(self, instance: T) -> dict:
        """
        Retrieve the metadata cache for the instance.

        Parameters
        ----------
        instance : T
            The instance of the class holding this metadata.

        Returns
        -------
        dict
            The metadata cache dictionary for the instance.
        """
        pass

    @abstractmethod
    def get_hdf5_ref(self, instance: T):
        """
        Retrieve the HDF5 attribute reference for the instance.

        Parameters
        ----------
        instance : T
            The instance of the class holding this metadata.

        Returns
        -------
        h5py.Group or h5py.Dataset or None
            The HDF5 reference object, or None if not applicable.
        """
        pass


class GridManagerMetadataDescriptor(MetadataDescriptor):
    """
    Metadata descriptor for :py:class:`GridManager` instances, handling metadata stored in the HDF5 file
    under the 'HEADER' group.

    This descriptor is used to access and modify metadata related to the overall grid management, such as
    grid dimensions, bounding boxes, or global attributes stored in the 'HEADER' group of the HDF5 file.

    Methods
    -------
    get_meta_cache(instance)
        Retrieve the metadata cache for the GridManager instance.
    get_hdf5_ref(instance)
        Retrieve the HDF5 attribute reference for the GridManager's 'HEADER' group.

    See Also
    --------
    :py:class:`MetadataDescriptor`
        Base class providing the main functionality and usage pattern.

    Examples
    --------
    .. code-block:: python

        # Example usage within a GridManager class
        class MyGridManager:
            NDIM = GridManagerMetadataDescriptor(required=True, default=3)
            BBOX = GridManagerMetadataDescriptor(default=[[0, 0, 0], [1, 1, 1]])

            def __init__(self, handle):
                self._handle = handle
                self._meta = {}

    """

    def get_meta_cache(self, instance: T) -> dict:
        return instance._meta

    def get_hdf5_ref(self, instance: T):
        try:
            return instance._handle["HEADER"]
        except KeyError:
            raise IOError("HDF5 file does not contain a 'HEADER' group.")
        except TypeError:
            raise GridManagerError(f"{instance} is closed.")


class GridLevelMetadataDescriptor(MetadataDescriptor):
    """
    Metadata descriptor for :py:class:`GridLevel` instances, handling metadata stored in the grid level's HDF5 group.

    This descriptor is used to manage metadata attributes that are specific to each grid level, such as
    refinement factors, level-specific dimensions, or grid attributes.

    Methods
    -------
    get_meta_cache(instance)
        Retrieve the metadata cache for the GridLevel instance.
    get_hdf5_ref(instance)
        Retrieve the HDF5 attribute reference for the GridLevel's HDF5 group.

    See Also
    --------
    :py:class:`MetadataDescriptor`
        Base class providing the main functionality and usage pattern.

    Examples
    --------
    .. code-block:: python

        # Example usage within a GridLevel class
        class MyGridLevel:
            CELL_SIZE = GridLevelMetadataDescriptor(required=True)
            REFINEMENT_FACTOR = GridLevelMetadataDescriptor(default=2)

            def __init__(self, handle):
                self._handle = handle
                self._meta = {}

    """

    def get_meta_cache(self, instance: T) -> dict:
        return instance._meta

    def get_hdf5_ref(self, instance: T):
        return instance._handle


class GridMetadataDescriptor(MetadataDescriptor):
    """
    Metadata descriptor for individual grids, managing metadata stored directly in the grid's HDF5 group.

    This descriptor handles grid-specific metadata, such as grid positions, dimensions, or physical quantities
    relevant to the specific grid.

    Methods
    -------
    get_meta_cache(instance)
        Retrieve the metadata cache for the Grid instance.
    get_hdf5_ref(instance)
        Retrieve the HDF5 attribute reference for the Grid's HDF5 group.

    See Also
    --------
    :py:class:`MetadataDescriptor`
        Base class providing the main functionality and usage pattern.

    Examples
    --------
    .. code-block:: python

        # Example usage within a Grid class
        class MyGrid:
            POSITION = GridMetadataDescriptor(required=True)
            SIZE = GridMetadataDescriptor(default=[1.0, 1.0, 1.0])

            def __init__(self, handle):
                self._handle = handle
                self._meta = {}

    """

    def get_meta_cache(self, instance: T) -> dict:
        return instance._meta

    def get_hdf5_ref(self, instance: T):
        return instance._handle


class FieldMetadataDescriptor(MetadataDescriptor):
    """
        Metadata descriptor for fields, handling metadata stored directly in the field's HDF5 dataset attributes.

        This descriptor manages field-specific metadata, such as data units, descriptions, or transformation
    parameters that are stored directly as attributes in the HDF5 dataset.

        Methods
        -------
        get_meta_cache(instance)
            Retrieve the metadata cache for the Field instance.
        get_hdf5_ref(instance)
            Retrieve the HDF5 attribute reference for the Field's HDF5 dataset.

        See Also
        --------
        :py:class:`MetadataDescriptor`
            Base class providing the main functionality and usage pattern.

        Examples
        --------
        .. code-block:: python

            # Example usage within a Field class
            class MyField:
                UNITS = FieldMetadataDescriptor(default="unitless")
                DESCRIPTION = FieldMetadataDescriptor(default="No description provided.")

                def __init__(self, buffer):
                    self.buffer = buffer
                    self._meta = {}

    """

    def get_meta_cache(self, instance: T) -> dict:
        return instance._meta

    def get_hdf5_ref(self, instance: T):
        return instance.buffer


# ===================================== #
# Element Containers                    #
# ===================================== #
class ElementContainer(
    OrderedDict[IndexType, ItemType], Generic[IndexType, ItemType], ABC
):
    """
    Base class for managing elements that are stored in an HDF5-backed container with LRU caching support.

    The `ElementContainer` class provides an abstract interface for managing collections of elements
    (e.g., grids, levels, fields) that are associated with an HDF5 handle. It supports lazy loading,
    unloading, and synchronization between in-memory representations and the HDF5 file, with an optional
    LRU caching mechanism to limit memory usage.

    Parameters
    ----------
    handle : h5py.Group or h5py.File
        The HDF5 handle used to store and retrieve elements. This can be either a group or a file handle,
        providing access to the HDF5 structure.

    Attributes
    ----------
    ERROR_TYPE : type
        The exception type to raise when an error related to the container occurs. By default, it is set to
        :py:class:`GridManagerError`.
    _handle : h5py.Group or h5py.File
        The HDF5 handle associated with the container.
    _index_to_hdf5_cache : dict
        Cache for converting element indices to HDF5-compatible labels.
    _hdf5_to_index_cache : dict
        Cache for converting HDF5-compatible labels to element indices.
    _max_cache_size : int or None
        The maximum number of elements allowed in the cache. If None, no limit is applied.

    Notes
    -----
    **Purpose and Design**:

    The `ElementContainer` class is designed as a base class to handle the management of elements
    (such as grids, levels, or fields) stored in an HDF5 file. It provides a consistent interface for
    loading, unloading, and accessing these elements in a lazy-loading fashion, minimizing memory
    usage and improving performance. Additionally, an LRU caching mechanism is integrated to further
    optimize memory usage.

    This class is intended to be subclassed with specific implementations of the following abstract methods:

    - :py:meth:`_index_to_hdf5`: Converts an element index to an HDF5-compatible string label.
    - :py:meth:`_index_from_hdf5`: Converts an HDF5-compatible string label back to an element index.
    - :py:meth:`load_element`: Loads a specific element from the HDF5 file into memory.
    - :py:meth:`load_existing_elements`: Identifies existing elements in the HDF5 file and adds them as placeholders.

    **LRU Caching**:

    The LRU (Least Recently Used) caching mechanism limits the number of elements stored in memory.
    When the cache exceeds its maximum size, the least recently accessed elements are removed from memory
    to make room for new elements.

    - The maximum cache size can be set using the class method :py:meth:`set_cache_size`.
    - If the cache size is set to `None` (the default), no limit is applied, and all accessed elements are kept in memory.

    **Lazy Loading Convention**:

    Lazy loading is a key feature of the `ElementContainer` class, allowing elements to be loaded into
    memory only when they are accessed, rather than at the time of container initialization. This reduces
    the memory footprint and speeds up initialization, especially when managing large datasets.

    - **Lazy Loading Implementation**:
      When an element is first accessed using the `__getitem__` method, the container checks if the element
      is already loaded in memory. If not, the `load_element` method is called to load the element from the
      HDF5 file, and it is then stored in memory.

      For example:

      .. code-block:: python

          element = container[index]  # This will trigger lazy loading if 'index' is not already in memory.

    - **Placeholders for Lazy Loading**:
      When `load_existing_elements` is called during initialization, each element in the HDF5 file is
      added as a placeholder (with a value of `None`) in the container. This indicates that the element
      is available in the HDF5 file but has not yet been loaded into memory.

      For example:

      .. code-block:: python

          # After initialization, container holds placeholders for elements in the HDF5 file.
          container.load_existing_elements()
          print(container)  # Output: {1: None, 2: None, 3: None}

    - **Unloading Elements**:
      Elements can be unloaded from memory using the :py:meth:`unload_element` method, which sets the
      element's value to `None` without removing it from the HDF5 file. This maintains the placeholder
      for the element, allowing it to be reloaded later if needed.

    **Implementation Details**:

    - **Indexing Conventions**:
      The `IndexType` generic parameter defines the type of index used for elements in the container.
      For example, a container managing grids may use tuples of integers `(i, j, k)` as indices, while a
      container for levels might use integers.

    - **HDF5 Compatibility**:
      The `_index_to_hdf5` and `_index_from_hdf5` methods ensure compatibility between in-memory indices
      and HDF5 labels. This allows seamless mapping between the Python interface and the HDF5 file structure.

      For example:

      .. code-block:: python

          def _index_to_hdf5(self, index: tuple[int, ...]) -> str:
              return f"GRID_{'_'.join(map(str, index))}"

          def _index_from_hdf5(self, label: str) -> tuple[int, ...]:
              return tuple(map(int, label.split("_")[1:]))

    - **Element Management**:
      Elements can be added to the container using the `__setitem__` method, and will be written to both
      memory and the HDF5 file. Deletion is handled by the `__delitem__` method, which removes the element
      from both the container and the HDF5 file.

      For example:

      .. code-block:: python

          container[1] = "Element data"  # Adds element to container and HDF5 file.
          del container[1]  # Removes element from both container and HDF5 file.

    - **Cache Management**:
      The `_index_to_hdf5_cache` and `_hdf5_to_index_cache` attributes store mappings between element indices
      and HDF5 labels. These caches are used to optimize lookups and prevent redundant computations during
      element access.

    This class provides a template for containers managing various types of elements like grids, fields,
    or levels within a hierarchical data structure. It abstracts the complexity of managing an HDF5-backed
    collection, enabling developers to focus on the specifics of their elements.

    See Also
    --------
    :py:class:`GridContainer` : Specialized container for managing grid objects.
    :py:class:`LevelContainer` : Container for managing grid levels within a grid manager.

    Examples
    --------
    .. code-block:: python

        import h5py

        # Example subclass implementation
        class MyElementContainer(ElementContainer[int, str]):
            def _index_to_hdf5(self, index: int) -> str:
                return f"Element_{index}"

            def _index_from_hdf5(self, label: str) -> int:
                return int(label.split("_")[1])

            def load_element(self, index: int) -> str:
                # This is a mock loading method. Typically, data would be loaded from the HDF5 handle.
                return f"Element data for index {index}"

            def load_existing_elements(self):
                for element in self._handle.keys():
                    super().__setitem__(self._index_from_hdf5(element), None)

        # Open an HDF5 file and create the container
        with h5py.File('example.h5', 'w') as f:
            container = MyElementContainer(f)
            container[1] = "Sample data for element 1"
            print(container[1])  # Output: Element data for index 1

            # Unload element 1
            container.unload_element(1)
            print(container[1])  # Re-loads from the HDF5 file (mock data in this example)
    """

    ERROR_TYPE = GridManagerError
    _max_cache_size = None

    @classmethod
    def set_cache_size(cls, size: int | None):
        """
        Set the maximum size of the cache.

        If the cache size is set to None, no limit is enforced, and all accessed elements are kept in memory.

        Parameters
        ----------
        size : int or None
            The maximum number of elements allowed in the cache. If None, no limit is applied.
        """
        if size is not None and size <= 0:
            raise ValueError("Cache size must be a positive integer or None.")
        cls._max_cache_size = size

    def __init__(self, handle):
        """
        Initialize the `ElementContainer`.

        This constructor sets up the HDF5 handle, initializes internal caches, and loads any existing elements
        from the HDF5 file into the container as placeholders.

        Parameters
        ----------
        handle : h5py.Group or h5py.File
            The HDF5 handle used to store and retrieve elements. This can be either a group or a file handle,
            providing access to the HDF5 structure.
        """
        super().__init__()

        # Setup the handle and the metadata attributes.
        self._handle = handle
        self._index_to_hdf5_cache = {}
        self._hdf5_to_index_cache = {}

        # Load all of the existing elements. These get
        # lazy loaded so they just become None
        self.load_existing_elements()

    def __getitem__(self, index: IndexType) -> ItemType:
        """
        Retrieve an element from the container, loading it from the HDF5 file if necessary.

        This method checks if the element is already loaded in memory. If not, it loads the element
        from the HDF5 file and caches it.

        Parameters
        ----------
        index : IndexType
            The index of the element to retrieve.

        Returns
        -------
        ItemType
            The element corresponding to the given index.

        Raises
        ------
        KeyError
            If the specified index does not exist in the container.
        """
        if index not in self:
            raise KeyError(f"{self} does not contain '{index}'.")
        if super().__getitem__(index) is None:
            element = self.load_element(index)
            super().__setitem__(index, element)
            self._update_cache(index)

        return super().__getitem__(index)

    def __setitem__(self, index: IndexType, value: ItemType | None):
        """
        Set an element in the container, updating both memory and HDF5.

        If a cache size limit is set, the least recently used elements may be evicted from the cache.

        Parameters
        ----------
        index : IndexType
            The index of the element to set.
        value : ItemType | None
            The element to store at the specified index, or None to unload it from memory.
        """
        super().__setitem__(index, value)
        self._update_cache(index)

    def __delitem__(self, index: IndexType):
        """
        Remove an element from the container and delete the corresponding HDF5 group or dataset.

        Parameters
        ----------
        index : IndexType
            The index of the element to be removed.

        Raises
        ------
        KeyError
            If the specified element does not exist in the container.
        """
        if index not in self:
            raise self.__class__.ERROR_TYPE(f"No element {index} in {self}.")

        # Convert index to HDF5 key and delete from HDF5 file if it exists
        hdf5_key = self._index_to_hdf5(index)
        if hdf5_key in self._handle:
            del self._handle[hdf5_key]

        # Remove from the in-memory container
        super().__delitem__(index)

    def __str__(self) -> str:
        """Return a string summarizing the number of elements managed."""
        return f"<{self.__class__.__name__}: Count={len(self)}>"

    def __repr__(self) -> str:
        """Return a detailed string summarizing the state of the container for debugging."""
        return (
            f"<{self.__class__.__name__}(count={len(self)}, "
            f"elements={list(self.keys())})>"
        )

    def _update_cache(self, index: IndexType):
        # Move the accessed element to the end to mark it as most recently used
        self.move_to_end(index)

        # Enforce cache size limits, and unload the least recently used element
        if self.__class__._max_cache_size is not None:
            while len(self) > self.__class__._max_cache_size:
                oldest_index, _ = self.popitem(last=False)
                self.unload_element(oldest_index)

    @abstractmethod
    def _index_to_hdf5(self, index: IndexType) -> str:
        """Convert an element index to an HDF5-compatible string.

        This method must be implemented by subclasses to specify how indices
        are converted to HDF5-compatible labels for storage and retrieval.

        Parameters
        ----------
        index : IndexType
            The index of the element.

        Returns
        -------
        str
            The HDF5-compatible label for the element.
        """
        pass

    @abstractmethod
    def _index_from_hdf5(self, label: str) -> IndexType:
        """Convert an HDF5-compatible string to an element index.

        This method must be implemented by subclasses to specify how HDF5 labels
        are converted back to element indices.

        Parameters
        ----------
        label : str
            The HDF5-compatible label for the element.

        Returns
        -------
        IndexType
            The index of the element.
        """
        pass

    @abstractmethod
    def load_element(self, index: IndexType) -> ItemType:
        """Load an element from the HDF5 file into memory.

        This method must be implemented by subclasses to specify how elements
        are loaded from the HDF5 file into the container.

        Parameters
        ----------
        index : IndexType
            The index of the element to load.

        Returns
        -------
        ItemType
            The element loaded from the HDF5 file.
        """
        pass

    @abstractmethod
    def load_existing_elements(self):
        """Identify and load existing elements from the HDF5 file.

        This method must be implemented by subclasses to scan the HDF5 file
        and add existing elements as placeholders in the container.

        Raises
        ------
        KeyError
            If an element is not found in the HDF5 file.
        """
        for element in self._handle.keys():
            super().__setitem__(element, None)

    def unload_element(self, index: IndexType):
        """
        Unload a specific element from memory, keeping a placeholder.

        This method unloads an element from memory but keeps its placeholder
        in the container, indicating it can be reloaded when needed.

        Parameters
        ----------
        index : IndexType
            The index of the element to unload.

        Raises
        ------
        KeyError
            If the specified index does not exist in the container.
        """
        if index not in self:
            raise KeyError(f"{index} is not in {self}.")
        super().__setitem__(index, None)

    def clear(self):
        """
        Unload all elements from the container and remove them from memory.

        This method will clear all elements from memory, but will not delete them from the HDF5 file.
        """
        for index in list(self.keys()):
            super().__setitem__(index, None)

    def values(self):
        """
        Override the values() method to lazily load elements during iteration.

        Yields
        ------
        ItemType
            The value corresponding to each key, loading the element if it's not already in memory.
        """
        for key in self.keys():
            yield self[key]  # This will trigger lazy loading via __getitem__

    def items(self):
        """
        Override the items() method to lazily load elements during iteration.

        Yields
        ------
        Tuple[IndexType, ItemType]
            The key-value pairs where the value is loaded lazily if necessary.
        """
        for key in self.keys():
            yield key, self[key]  # This will trigger lazy loading via __getitem__

    def keys(self):
        """
        Override the keys() method. No need for lazy loading here since it's just returning the keys.

        Returns
        -------
        Iterator[IndexType]
            An iterator over the keys of the container.
        """
        return super().keys()  # No change needed here, keys don't require lazy loading

    def load_all(self):
        """
        Load all elements into memory from the HDF5 file.

        This method will force all elements in the HDF5 file to be loaded into memory.
        """
        keys = list(self.keys())  # Store keys locally to avoid multiple calls
        for index in keys:
            if super().__getitem__(index) is None:
                super().__setitem__(index, self.load_element(index))
