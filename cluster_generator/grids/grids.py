"""
Grid Structure Module
=====================

This module provides a set of classes to manage Adaptive Mesh Refinement (AMR) grid structures
and their associated fields within an HDF5 file. The primary classes include
:py:class:`LevelContainer`, :py:class:`GridLevel`, :py:class:`GridContainer`,
:py:class:`Grid`, :py:class:`FieldContainer`, and :py:class:`Field`. These classes work
together to represent a hierarchical grid system, where each level and grid can
contain multiple fields, all of which are backed by HDF5 datasets.

The module is designed to handle large-scale grid data, with support for lazy loading,
unloading, and efficient memory management. It also provides tools for manipulating
grid refinement levels and their corresponding grids and fields.


Examples
--------

.. code-block:: python

    import h5py
    import numpy as np
    from unyt import unyt_array, Unit

    # Create an HDF5 file
    with h5py.File("example.h5", "w") as hdf5_file:
        # Initialize GridManager and LevelContainer
        grid_manager = GridManager(hdf5_file, ndim=3)
        level_container = LevelContainer(grid_manager)

        # Add a new level
        new_level = level_container.add_level(refinement_factor=2)

        # Add a grid to the new level
        grid_indices = (0, 0, 0)
        new_grid = new_level.add_grid(grid_indices, BBOX=np.array([[0, 0, 0], [1, 1, 1]]))

        # Add a field to the grid
        new_field = new_grid.add_field("density", dtype=np.float64, units="g/cm**3")

        # Set data in the field
        new_field[:] = np.random.rand(*new_grid.SHAPE)

        # Access data from the field
        print(new_field[:])

"""
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from unyt import Unit, unyt_array

from cluster_generator.grids._types import (
    BoundingBox,
    ElementContainer,
    FieldNotFoundError,
    GridError,
    GridLevelMetadataDescriptor,
    GridMetadataDescriptor,
    GridMetadataError,
    GridNotFoundError,
    LevelNotFoundError,
)

if TYPE_CHECKING:
    import logging

    from cluster_generator.geometry._abc import GeometryHandler
    from cluster_generator.grids.managers import GridManager
    from cluster_generator.profiles._abc import Profile


class LevelContainer(ElementContainer[int, "GridLevel"]):
    """
    Container for managing `GridLevel` objects within a :py:class:`GridManager`.

    This class provides methods for the efficient management of grid levels in an
    Adaptive Mesh Refinement (AMR) grid hierarchy. It handles the creation, deletion,
    and lazy loading of `GridLevel` objects, allowing for on-demand access and
    memory-efficient management of potentially large datasets. Each `GridLevel`
    corresponds to a specific level of refinement in the AMR hierarchy, with the
    `LevelContainer` tracking their properties and relationships.

    The `LevelContainer` is tightly integrated with the :py:class:`GridManager`, which
    acts as the main controller for the grid data and structure. The `LevelContainer`
    operates on the levels, adding new levels as required, managing the refinement
    factors, and ensuring that levels are correctly linked within the grid hierarchy.

    Parameters
    ----------
    grid_manager : :py:class:`GridManager`
        The `GridManager` instance to which this container belongs.

    Attributes
    ----------
    grid_manager : :py:class:`GridManager`
        The `GridManager` instance that owns this `LevelContainer`.
    COARSE_LEVEL : int
        The coarse level ID for the initial grid. This is usually the base level of the
        grid hierarchy, representing the least refined level.
    COARSE_CELL_SIZE : numpy.ndarray of float
        The cell size of the coarse grid, represented as an array of floats for each
        spatial dimension.
    refinement_factors : dict
        Dictionary mapping level IDs (integers) to their refinement factors relative to
        the base level. Each level has a unique refinement factor that defines how much
        more refined it is compared to its parent level.
    LEVEL_PREFIX : str
        Prefix used to label levels in the HDF5 file. This prefix is used to generate
        unique keys for each level in the HDF5 structure, such as "LEVEL_0", "LEVEL_1", etc.
    ERROR_TYPE : type
        The exception type to raise when a level is not found in the container. By
        default, this is set to :py:class:`LevelNotFoundError`.
    _index_to_hdf5_cache : dict
        Cache for converting integer indices to HDF5-compatible labels. This helps to
        speed up the conversion process when repeatedly accessing levels.
    _hdf5_to_index_cache : dict
        Cache for converting HDF5-compatible labels back to integer indices.

    Notes
    -----
    The `LevelContainer` class is designed to provide a high-level interface for
    managing the refinement levels of an AMR grid. It abstracts away the complexity
    of managing the hierarchical grid structure, allowing users to focus on the
    physics and data analysis. Each level in the container can contain multiple grids,
    and the `LevelContainer` ensures that these grids are correctly linked to their
    corresponding levels.

    The class uses lazy loading to optimize memory usage. When a level is first
    accessed, it is loaded into memory from the HDF5 file. If a level is not needed
    for some time, it can be unloaded to free up resources. This design pattern is
    particularly useful for large-scale simulations where memory constraints are a
    significant concern.

    .. warning::
        The `LevelContainer` assumes that the levels are added sequentially, starting
        from the base level (`COARSE_LEVEL`). Attempting to add or remove levels out
        of sequence may result in unexpected behavior or errors.

    See Also
    --------
    :py:class:`GridManager`
        The `GridManager` class that owns and controls the `LevelContainer`. The
        `GridManager` provides a higher-level interface for managing the entire grid
        structure and its associated metadata.
    :py:class:`GridLevel`
        The `GridLevel` class represents a single level in the AMR hierarchy. Each
        `GridLevel` can contain multiple grids, which are further managed by a
        :py:class:`GridContainer`.

    Examples
    --------
    .. code-block:: python

        # Create an HDF5 file and initialize GridManager and LevelContainer
        with h5py.File("example.h5", "w") as hdf5_file:
            grid_manager = GridManager(hdf5_file, ndim=3)
            level_container = LevelContainer(grid_manager)

            # Add a new level with a refinement factor of 3
            new_level = level_container.add_level(refinement_factor=3)

            # Access an existing level
            existing_level = level_container[0]

            # Remove a level
            level_container.remove_level(1)

    .. admonition:: Best Practices

        - **Sequential Level Management**: Always add or remove levels in sequence, starting
          from the base level. This ensures consistency in the AMR hierarchy and avoids
          potential errors in grid linkage.
        - **Lazy Loading**: Use lazy loading to manage memory effectively. Load levels only
          when needed, and unload them when they are no longer in use.
        - **Refinement Factors**: Choose appropriate refinement factors to balance between
          computational cost and the accuracy of your simulation.

    """

    LEVEL_PREFIX = "LEVEL_"
    ERROR_TYPE = LevelNotFoundError

    def __init__(self, grid_manager):
        """
        Initialize the `LevelContainer`.

        Parameters
        ----------
        grid_manager : :py:class:`GridManager`
            The `GridManager` instance to which this container belongs.

        Examples
        --------
        .. code-block:: python

            # Create an HDF5 file and initialize GridManager and LevelContainer
            with h5py.File("example.h5", "w") as hdf5_file:
                grid_manager = GridManager(hdf5_file, ndim=3)
                level_container = LevelContainer(grid_manager)
        """
        # Set up attributes.
        self.grid_manager = grid_manager
        self.COARSE_LEVEL = grid_manager.COARSE_LEVEL
        self.COARSE_CELL_SIZE = grid_manager.COARSE_CELL_SIZE
        self.refinement_factors = {}

        # initialize parent class
        super().__init__(grid_manager._handle)

        # Initialize the coarse grid if it isn't already present
        if self.COARSE_LEVEL not in self.keys():
            self._add_coarse_level()

    def __delitem__(self, index: int):
        """
        Remove a grid level from the container and delete its corresponding HDF5 group.

        Parameters
        ----------
        index : int
            The index of the level to remove.

        Raises
        ------
        ValueError
            If the level being removed is not the highest level.
        KeyError
            If the specified level does not exist in the container.

        Notes
        -----
        This method ensures that the deletion of a level will also update the
        parent grids in the previous level by resetting their children to empty arrays.
        This avoids broken references to deleted grids.

        Examples
        --------
        .. code-block:: python

            # Remove a level by its index
            del level_container[1]
        """
        if index not in self:
            raise self.__class__.ERROR_TYPE(f"Level {index} does not exist in {self}.")

        max_level = max(self.keys())
        if index < max_level:
            raise ValueError(
                f"Cannot remove level {index}. Only the highest level ({max_level}) can be removed."
            )

        # Remove from in-memory container and delete the corresponding HDF5 group
        hdf5_key = self._index_to_hdf5(index)
        if hdf5_key in self._handle:
            del self._handle[hdf5_key]

        # Remove the level from the container and refinement factors
        super().__delitem__(index)
        del self.refinement_factors[index]

        # In the (now highest level), we need to remove all the children for all the grids.
        # We replace the children with empty arrays.
        for grid in self[index - 1].Grids.values():
            grid.CHILDREN = np.zeros((0, self.grid_manager.NDIM), dtype="int32")

        self.logger.info("[DEL ] Removed level %d.", index)

    def _index_to_hdf5(self, index: int) -> str:
        """
        Convert a level index to an HDF5-compatible string label.

        Parameters
        ----------
        index : int
            The index of the level.

        Returns
        -------
        str
            The HDF5-compatible label for the level.

        Examples
        --------
        .. code-block:: python

            # Convert a level index to HDF5 label
            hdf5_label = level_container._index_to_hdf5(0)
        """
        if index not in self._index_to_hdf5_cache:
            self._index_to_hdf5_cache[index] = f"{self.LEVEL_PREFIX}{index}"
        return self._index_to_hdf5_cache[index]

    def _index_from_hdf5(self, label: str) -> int:
        """
        Convert an HDF5-compatible string label to a level index.

        Parameters
        ----------
        label : str
            The HDF5-compatible label for the level.

        Returns
        -------
        int
            The index of the level.

        Examples
        --------
        .. code-block:: python

            # Convert an HDF5 label to a level index
            level_index = level_container._index_from_hdf5("LEVEL_0")
        """
        if label not in self._hdf5_to_index_cache:
            self._hdf5_to_index_cache[label] = int(label.split(self.LEVEL_PREFIX)[1])
        return self._hdf5_to_index_cache[label]

    def _add_coarse_level(self):
        """
        Initialize and add the coarse grid level.

        This method creates the base grid level with no refinement.

        Returns
        -------
        None

        Examples
        --------
        .. code-block:: python

            # Add the coarse level manually
            level_container._add_coarse_level()
        """
        self.logger.info(
            "[NEW ] \t\tCreating coarse level (LEVEL_%d)...", self.COARSE_LEVEL
        )

        # Determine the relevant level metadata before creating the level.
        level_id = self.COARSE_LEVEL
        _ = self._handle.create_group(self._index_to_hdf5(level_id))
        refinement_factor = 1
        level_cell_size = self.COARSE_CELL_SIZE

        # Create the level and add it to the container.
        self[level_id] = GridLevel(
            self.grid_manager,
            level_id,
            CELL_SIZE=level_cell_size,
            REFINEMENT_FACTOR=refinement_factor,
        )
        self.refinement_factors[level_id] = refinement_factor

        # Directly create the coarse grid in that level.

        grid_id = tuple(
            0 for _ in range(self.grid_manager.NDIM)
        )  # Create grid_id with (0,0,...,0)
        hdf5_key = self[level_id].Grids._index_to_hdf5(grid_id)
        _ = self[level_id]._handle.create_group(hdf5_key)
        self.logger.info(
            "[NEW ] \t\tCreating coarse grid %s in (LEVEL_%d)...",
            grid_id,
            self.COARSE_LEVEL,
        )

        # Directly initializing Grid object without using add_grid()
        coarse_grid = Grid(
            grid_level=self[level_id],
            grid_id=grid_id,
            BBOX=self.grid_manager.BBOX,  # Use the full bounding box for the coarse grid
            PARENT=-1 * np.ones(self.grid_manager.NDIM),
            CHILDREN=np.zeros((0, self.grid_manager.NDIM)),
        )

        # Assign the grid to the Grids dictionary without using add_grid.
        self[level_id].Grids[grid_id] = coarse_grid

    def load_existing_elements(self):
        """
        Load all existing grid levels from the HDF5 file.

        This method scans the HDF5 file for existing levels and loads
        them into the container.

        Returns
        -------
        None

        Raises
        ------
        :py:class:`GridError`
            If metadata for a level cannot be loaded.

        Examples
        --------
        .. code-block:: python

            # Load all existing levels from the HDF5 file
            level_container.load_existing_elements()
        """
        for element in self._handle.keys():
            if self.LEVEL_PREFIX in element:
                index = self._index_from_hdf5(element)
                try:
                    self.refinement_factors[index] = self._handle[element].attrs[
                        "REFINEMENT_FACTOR"
                    ]
                    self[index] = None
                    self.logger.info(
                        "[INIT] \t\t\t Loaded LEVEL_%d, RF=%d.",
                        index,
                        self.refinement_factors[index],
                    )
                except KeyError:
                    raise GridError(f"Missing refinement factor for level {index}.")
                except Exception as e:
                    raise GridError(f"Failed to load metadata for level {index}: {e}")

    def add_level(self, refinement_factor: int = 2) -> "GridLevel":
        """
        Add a new grid level to the container.

        This method creates a new `GridLevel` in the AMR hierarchy and adds it to the
        `LevelContainer`. The new level will be a refinement of the previous highest
        level, based on the specified `refinement_factor`. The `refinement_factor`
        determines how much smaller the cell size of the new level will be compared
        to its parent level. For example, a refinement factor of 2 will halve the cell
        size in each spatial dimension, resulting in a total refinement factor of 2^NDIM.

        Parameters
        ----------
        refinement_factor : int, optional
            The refinement factor for the new level, by default 2. This factor determines
            the increase in resolution between the new level and its parent level.

        Returns
        -------
        :py:class:`GridLevel`
            The newly created `GridLevel` object. This object is fully initialized
            and can be used to add or access grids and fields.

        Raises
        ------
        ValueError
            If the base level is not initialized. This can happen if no levels have been
            added to the container, which would mean that the base level (`COARSE_LEVEL`)
            is missing.

        Notes
        -----
        The `add_level` method follows a strict hierarchical order. It only allows
        adding a new level as the next higher refinement level in the sequence.
        This ensures that the AMR hierarchy remains consistent and that levels are
        sequentially linked.

        The `refinement_factor` should be carefully chosen based on the physical
        requirements of the simulation. A higher refinement factor results in more
        refined grids, which can capture finer details but also significantly increase
        the computational cost.

        Examples
        --------
        .. code-block:: python

            # Create an HDF5 file and initialize GridManager and LevelContainer
            with h5py.File("example.h5", "w") as hdf5_file:
                grid_manager = GridManager(hdf5_file, ndim=3)
                level_container = LevelContainer(grid_manager)

                # Add a base level first
                base_level = level_container.add_level(refinement_factor=1)

                # Add a new level with a refinement factor of 3
                new_level = level_container.add_level(refinement_factor=3)

                print(f"Added new level: {new_level}, with cell size: {new_level.CELL_SIZE}")

        .. admonition:: Important
            The new level is created by refining the previous highest level.
            Ensure that the refinement factor is appropriate for your simulation
            to avoid unnecessary computational overhead.
        """
        # Ensure the base level exists.
        if self.COARSE_LEVEL not in self:
            raise ValueError("Cannot add level without base level initialization.")

        # Determine the parameters for the new level.
        level_id = max(self.keys()) + 1
        prev_level = self[level_id - 1]
        level_cell_size = prev_level.CELL_SIZE / refinement_factor

        self.logger.info(
            "[ADD ] Creating level %d with refinement %d...",
            level_id,
            refinement_factor,
        )

        _ = self._handle.create_group(self._index_to_hdf5(level_id))

        # Create the new level and add it to the container.
        self[level_id] = GridLevel(
            self.grid_manager,
            level_id,
            CELL_SIZE=level_cell_size,
            REFINEMENT_FACTOR=refinement_factor,
        )
        self.refinement_factors[level_id] = refinement_factor

        return self[level_id]

    def remove_level(self, level_index: int) -> None:
        """
        Remove a specific grid level from the container.

        This method removes the specified grid level from the `LevelContainer` and
        deletes its corresponding data in the HDF5 file. It performs consistency checks
        to ensure that only the highest (most refined) level can be removed, preventing
        the accidental removal of lower levels that would disrupt the AMR hierarchy.

        Parameters
        ----------
        level_index : int
            The index of the level to be removed. This index corresponds to the level
            ID in the AMR hierarchy and is used to locate the level within the container.

        Raises
        ------
        KeyError
            If the specified level does not exist in the container. This can occur if
            the `level_index` is not present in the container or if the level has already
            been removed.
        ValueError
            If the level is not the highest level. The method enforces that only the
            highest level can be removed to maintain the consistency of the AMR hierarchy.

        Notes
        -----
        The `remove_level` method should be used with caution. Removing a level will
        delete all the data associated with that level, including all grids and fields
        contained within it. This action is irreversible, so ensure that the level is
        no longer needed before calling this method.

        This method performs a consistency check to ensure that only the highest level
        can be removed. Attempting to remove a lower level will raise a `ValueError`.
        This is done to prevent breaking the AMR hierarchy and to protect the integrity
        of the simulation data.

        Examples
        --------
        .. code-block:: python

            # Create an HDF5 file and initialize GridManager and LevelContainer
            with h5py.File("example.h5", "w") as hdf5_file:
                grid_manager = GridManager(hdf5_file, ndim=3)
                level_container = LevelContainer(grid_manager)

                # Add a base level first
                base_level = level_container.add_level(refinement_factor=1)

                # Add a new level with a refinement factor of 3
                new_level = level_container.add_level(refinement_factor=3)

                # Remove the most refined level
                level_container.remove_level(1)
                print("Removed level 1")

                # Attempt to remove base level, which will raise an error
                try:
                    level_container.remove_level(0)
                except ValueError as e:
                    print(f"Error: {e}")

        .. admonition:: Caution
            Removing a level will delete all data associated with that level.
            Ensure that you have backed up any necessary data before calling this method.
        """
        self.__delitem__(level_index)

    def load_element(self, index: int) -> "GridLevel":
        """
        Load a `GridLevel` object for a given index.

        This method lazily loads a `GridLevel` object for the given index.

        Parameters
        ----------
        index : int
            The index of the level to load.

        Returns
        -------
        :py:class:`GridLevel`
            The loaded `GridLevel` object.

        Raises
        ------
        :py:class:`GridError`
            If the level cannot be loaded.

        Examples
        --------
        .. code-block:: python

            # Load a specific level
            loaded_level = level_container.load_element(0)
        """
        try:
            self.logger.debug("[LOAD] Reloading level %d in %s.", index, self)
            return GridLevel(self.grid_manager, index)
        except Exception as e:
            raise GridError(f"Failed to load level {index}: {e}")

    def unload_element(self, index: int):
        """
        Unload a specific grid level from memory.

        Parameters
        ----------
        index : int
            The index of the grid level to unload.

        Raises
        ------
        KeyError
            If the specified index does not exist in the container.

        Examples
        --------
        .. code-block:: python

            # Unload a specific level
            level_container.unload_element(0)
        """
        if index not in self:
            raise KeyError(f"{index} is not in {self}.")
        self.logger.debug("[LOAD] Unloading level %s in %s.", index, self)
        super().__setitem__(index, None)

    @property
    def logger(self):
        """
        Return the logger associated with the GridManager.

        Returns
        -------
        logging.Logger
            The logger for this `LevelContainer`.

        Examples
        --------
        .. code-block:: python

            # Log a message using the logger
            level_container.logger.info("This is a log message.")
        """
        return self.grid_manager.logger


class GridLevel:
    """Class representation of a specific level in an AMR grid hierarchy.

    This class is responsible for managing the grids at a
    specific level, as well as handling the metadata that defines the level's
    properties, such as cell size and refinement factors.

    This class supports both HDF5-backed and memory-backed grid management,
    allowing for efficient storage and retrieval of grid data. It also provides
    methods for initializing, updating, and deleting grids, as well as calculating
    derived properties like total refinement and grid allocation.

    Parameters
    ----------
    grid_manager : :py:class:`GridManager`
        The `GridManager` instance that owns this `GridLevel`.
    level : int, optional
        The index of this level in the AMR hierarchy. If not provided, this must
        be set later before the object can be used.
    **kwargs : dict, optional
        Additional metadata attributes for initialization. These include properties
        like `CELL_SIZE` that define the level's grid characteristics.

    Attributes
    ----------
    CELL_SIZE : NDArray[float]
        Descriptor for the size of each cell in the grids at this level.
    REFINEMENT_FACTOR : int
        Descriptor for the refinement factor of this level relative to its parent level.
    grid_manager : :py:class:`GridManager`
        The `GridManager` instance that owns this `GridLevel`.
    index : int
        The index of this level in the AMR hierarchy.
    _handle : h5py.Group or dict
        The underlying HDF5 group or in-memory dictionary for this level.
    _meta : dict
        A dictionary storing metadata for this level, either from initialization
        parameters or derived from the HDF5 file.
    Grids : :py:class:`GridContainer`
        Container for managing the grids at this level.

    Raises
    ------
    GridError
        If the HDF5 group for this level does not exist or cannot be accessed.

    Notes
    -----
    `GridLevel` objects are primarily intended to be used within the context of
    an AMR grid hierarchy managed by a `GridManager`. Each level represents a
    refinement of its parent level, with a corresponding decrease in cell size
    and increase in the number of grids.

    The `CELL_SIZE` and `REFINEMENT_FACTOR` attributes are critical for defining
    the properties of a level. These must be correctly set either during
    initialization or derived from the HDF5 file to ensure that grids are
    correctly initialized and managed.

    Examples
    --------
    .. code-block:: python

        # Assuming a GridManager instance exists and is named `grid_manager`:
        level = GridLevel(grid_manager, level=1, CELL_SIZE=[0.1, 0.1, 0.1])
        print(level)

        # Adding a grid to the level
        grid_indices = [0, 0, 0]
        level.add_grid(grid_indices, BBOX=[(0, 0, 0), (0.1, 0.1, 0.1)])
        print(f"Number of grids: {len(level)}")

    See Also
    --------
    :py:class:`Grid`
        Represents an individual grid within a `GridLevel`.
    :py:class:`GridManager`
        Manages the overall AMR hierarchy and all grid levels.
    """

    CELL_SIZE: NDArray[float] = GridLevelMetadataDescriptor(required=True)
    REFINEMENT_FACTOR: int = GridLevelMetadataDescriptor(required=True, default=1)

    # Cache for storing references to the MetadataDescriptors.
    _metadata_cache: ClassVar[dict[str, "GridLevelMetadataDescriptor"]] = {}

    def __init__(self, grid_manager: "GridManager", level=None, **kwargs):
        self.grid_manager = grid_manager
        self.index = level

        # Construct a viable HDF5 handle from the available information.
        self._handle = self.grid_manager._handle.get(f"LEVEL_{self.index}", None)
        if self._handle is None:
            raise GridError(f"HDF5 group for LEVEL_{self.index} does not exist.")

        # Load metadata from the provided kwargs or from the HDF5 file.
        self._meta = {}
        self._initialize_metadata(kwargs)

        # Derive additional metadata if possible.
        self._initialize_additional_metadata()

        # We now need to construct the grid dictionary. If this is memory backed,
        # its empty; otherwise we do need to populate it with HDF5 backed grids.
        self.Grids = GridContainer(self)

    def _initialize_metadata(self, kwargs: dict):
        """
        Initialize metadata from kwargs or the HDF5 file.

        This method initializes the metadata for the `GridLevel` based on
        the provided keyword arguments or by reading from the HDF5 file.
        Metadata such as `CELL_SIZE` must be correctly set for the level
        to function properly.

        Parameters
        ----------
        kwargs : dict
            Additional metadata attributes for initialization. These include
            properties like `CELL_SIZE` that define the level's grid characteristics.

        Raises
        ------
        GridMetadataError
            If required metadata is missing or incompatible.

        Notes
        -----
        This method validates the provided metadata against expected formats
        and constraints. If no metadata is provided for a required attribute,
        it attempts to load it from the HDF5 file. If that also fails, an
        exception is raised.
        """
        # Validate level data passed in the kwargs.
        cell_size = kwargs.get("CELL_SIZE", None)

        if cell_size is not None:
            try:
                cell_size = np.asarray(cell_size, dtype="f8").reshape(
                    (self.grid_manager.NDIM,)
                )
            except ValueError as e:
                raise GridMetadataError(f"Failed to validate CELL_SIZE: {e}.")

            kwargs["CELL_SIZE"] = cell_size

        for _, meta_class in self._get_metadata_descriptors().items():
            meta_class.set_metadata_from_kwargs_or_hdf5(self, kwargs)

    def _initialize_additional_metadata(self):
        """
        Initialize additional metadata based on existing metadata.

        This method calculates derived metadata attributes that depend on
        existing metadata. For example, it calculates the total refinement
        factor for the level based on its refinement relative to the coarse
        level.

        Raises
        ------
        GridMetadataError
            If additional metadata cannot be derived from existing metadata.

        Notes
        -----
        This method is called after `_initialize_metadata` and depends on
        the core metadata attributes like `CELL_SIZE` being correctly set.
        It is used to compute derived attributes such as `GRID_SIZE` and
        `TOTAL_REFINEMENT`.
        """
        try:
            # Validate CELL_SIZE and BLOCK_SIZE dimensions
            self.GRID_SIZE = self.CELL_SIZE * self.grid_manager.BLOCK_SIZE
        except Exception as e:
            raise GridMetadataError(
                f"Failed to initialize additional metadata for GridLevel {self.index}: {e}"
            )

        # Load the total refinement factor
        if self.index == self.grid_manager.COARSE_LEVEL:
            self.TOTAL_REFINEMENT = 1
        else:
            parent_total_refinement = self.grid_manager[self.index - 1].TOTAL_REFINEMENT
            self.TOTAL_REFINEMENT = parent_total_refinement * self.REFINEMENT_FACTOR

    def __repr__(self) -> str:
        """
        Debugging-oriented string representation of the GridLevel.

        Returns
        -------
        str
            A detailed string summarizing the GridLevel's state, including
            its index, number of grids, and metadata.

        Notes
        -----
        This representation is intended to provide a quick overview of
        the GridLevel's properties for debugging and logging purposes.
        It includes all metadata and the count of grids contained in the level.
        """
        num_grids = len(self.Grids)
        metadata = {
            key: getattr(self, key, None)
            for key in self._get_metadata_descriptors().keys()
        }
        return (
            f"<GridLevel(index={self.index}, num_grids={num_grids}, "
            f"metadata={metadata})>"
        )

    def __str__(self):
        """
        User-friendly string representation of the GridLevel.

        Returns
        -------
        str
            A simpler string summarizing the level's index and number of grids.

        Notes
        -----
        This representation is intended to be more concise than `__repr__`,
        providing only the most essential information about the level.
        It is useful for general user feedback and logging.
        """
        return f"<GridLevel {self.index}: Grids={len(self.Grids)}>"

    def __len__(self) -> int:
        """
        Return the number of grids in this GridLevel.

        Returns
        -------
        int
            Number of grids managed by the GridLevel.

        Notes
        -----
        This method provides a quick way to determine the size of the level
        in terms of the number of grids it contains. It is equivalent to
        calling `len(level.Grids)`.
        """
        return len(self.Grids)

    def __getitem__(self, grid_id: tuple[int, ...]) -> "Grid":
        """
        Access a specific grid in the Grids collection.

        Parameters
        ----------
        grid_id : tuple[int,...]
            Identifier for the grid. This identifier is a tuple of integers
            that uniquely specifies the grid's position within the level.

        Returns
        -------
        :py:class:`Grid`
            The `Grid` object corresponding to the `grid_id`.

        Raises
        ------
        KeyError
            If the `grid_id` is not found in the Grids collection.

        Notes
        -----
        This method provides direct access to grids within the level by their
        unique identifier. It raises a `KeyError` if the requested grid does
        not exist, ensuring that only valid grids are accessed.
        """
        return self.Grids[grid_id]

    def __setitem__(self, grid_id: tuple[int, ...], grid: "Grid") -> None:
        """
        Add or update a grid in the Grids collection.

        Parameters
        ----------
        grid_id : tuple[int,...]
            Identifier for the grid. This identifier is a tuple of integers
            that uniquely specifies the grid's position within the level.
        grid : :py:class:`Grid`
            The `Grid` object to be added or updated.

        Notes
        -----
        This method allows for the addition or update of grids within the level.
        If a grid with the specified `grid_id` already exists, it will be
        replaced with the new `grid` object. If not, the grid will be added
        to the collection.
        """
        self.Grids[grid_id] = grid

    def __delitem__(self, grid_id: tuple[int, ...]) -> None:
        """
        Remove a grid from the Grids collection.

        Parameters
        ----------
        grid_id : tuple[int,...]
            Identifier for the grid to be removed. This identifier is a
            tuple of integers that uniquely specifies the grid's position
            within the level.

        Raises
        ------
        KeyError
            If the `grid_id` is not found in the Grids collection.

        Notes
        -----
        This method removes the grid specified by `grid_id` from the Grids
        collection. It also deletes the corresponding HDF5 group if the level
        is HDF5-backed. If the grid does not exist, a `KeyError` is raised.
        """
        del self.Grids[grid_id]

    def __contains__(self, grid_id: tuple[int, ...]) -> bool:
        """
        Check if a grid exists in the Grids collection.

        Parameters
        ----------
        grid_id : tuple[int,...]
            Identifier for the grid. This identifier is a tuple of integers
            that uniquely specifies the grid's position within the level.

        Returns
        -------
        bool
            True if the grid exists in the Grids collection, False otherwise.

        Notes
        -----
        This method allows for quick checks to see if a grid is present in
        the level. It is equivalent to calling `grid_id in level.Grids`.
        """
        return grid_id in self.Grids

    @classmethod
    def _get_metadata_descriptors(cls) -> dict[str, "GridLevelMetadataDescriptor"]:
        """
        Retrieve all metadata descriptors (class attributes that are instances
        of `GridLevelMetadataDescriptor`), including those in parent classes.

        Returns
        -------
        dict
            A dictionary of metadata descriptors with the attribute names as
            keys and the descriptor instances as values.

        Notes
        -----
        This method is used internally to gather all metadata descriptors for
        the class, including those inherited from parent classes. It caches
        the descriptors to avoid recomputation and ensures that all necessary
        metadata is available for initialization and validation.
        """
        if not cls._metadata_cache:
            descriptors = {
                name: attr
                for base_class in cls.__mro__
                for name, attr in vars(base_class).items()
                if isinstance(attr, GridLevelMetadataDescriptor)
            }
            cls._metadata_cache = descriptors
        return cls._metadata_cache

    def add_grid(self, indices: NDArray[int], **kwargs):
        """
        Add a new grid to this `GridLevel`.

        This method adds a grid at the specified indices in the AMR hierarchy,
        initializing it with the provided metadata attributes.

        Parameters
        ----------
        indices : numpy.ndarray of int
            The grid indices (i, j, k) representing the grid's position within the level's domain.
            This must be a 1D array with a length equal to the number of spatial dimensions (`NDIM`).
        **kwargs : dict, optional
            Additional parameters for the grid initialization, such as `BBOX` (bounding box),
            `dtype` (data type), and `units` (physical units).
        _validate : bool, optional
            Whether to validate the grid indices and ensure they are within bounds. If False,
            no validation will occur. Default is True.

        Returns
        -------
        :py:class:`Grid`
            The newly created `Grid` object.

        Notes
        -----
        The `parent_index` is calculated from the grid indices to ensure that this grid's parent
        in the previous level is correctly referenced. Additionally, validation checks are
        performed to ensure the grid fits within the bounding box of the level, preventing
        grids from being created out of bounds.

        Examples
        --------
        .. code-block:: python

            new_grid = level.add_grid(np.array([0, 1, 0]), BBOX=[(0, 0, 0), (0.1, 0.1, 0.1)])
        """
        new_grid = self.Grids.add_grid(indices, **kwargs)

        return new_grid

    @property
    def logger(self) -> "logging.Logger":
        """
        Return the logger associated with the `GridManager`.

        This property provides access to the logger used for logging messages
        related to this `GridLevel`. It is typically used for debugging,
        information, and error messages.

        Returns
        -------
        logging.Logger
            The logger for this `GridLevel`.

        Notes
        -----
        The logger is shared with the `GridManager` that owns this level.
        This ensures consistent logging behavior across all levels and grids.
        """
        return self.grid_manager.logger

    @property
    def num_grids_max(self) -> int:
        """
        Return the maximum number of grids this level can support.

        This property computes the maximum number of grids that can exist
        at this level, based on the total refinement factor and the number
        of spatial dimensions. It is calculated as :math:`R_{\text{total}}^{N_{\text{dim}}}`,
        where :math:`R_{\text{total}}` is the total refinement factor of this
        level and :math:`N_{\text{dim}}` is the number of spatial dimensions.

        Returns
        -------
        int
            The maximum number of grids.

        Notes
        -----
        The maximum number of grids, :math:`N_{\text{grids}}`, at a specific
        level is given by:

        .. math::
            N_{\text{grids}} = \left( R_{\text{total}} \right)^{N_{\text{dim}}}

        where:

        - :math:`R_{\text{total}}` is the total refinement factor, defined as
          the product of all individual refinement factors from the base level
          to this level.
        - :math:`N_{\text{dim}}` is the number of spatial dimensions (e.g., 2 for 2D grids, 3 for 3D grids).

        This formula provides an upper limit on the number of grids that can be
        allocated at this level, assuming full refinement across all dimensions.

        Examples
        --------
        For a 2D grid level with a total refinement factor of 4:

        .. math::
            N_{\text{grids}} = 4^{2} = 16

        This means that the maximum number of grids this level can support is 16.
        """
        if "NGRID_MAX" not in self._meta:
            try:
                self._meta["NGRID_MAX"] = (
                    self.TOTAL_REFINEMENT**self.grid_manager.NDIM
                )
            except ZeroDivisionError:
                raise GridMetadataError(
                    "CELL_SIZE or COARSE_CELL_SIZE is zero, cannot compute NGRID_MAX."
                )
        return self._meta["NGRID_MAX"]

    @property
    def allocation_fraction(self) -> float:
        """
        Return the fraction of grids currently allocated relative to the maximum possible.

        This property provides a measure of how many grids have been allocated
        at this level, relative to the maximum number of grids that can be
        supported. It is calculated as `len(self.Grids) / self.num_grids_max`.

        Returns
        -------
        float
            The fraction of grids allocated.

        Notes
        -----
        This property is useful for monitoring the utilization of the level
        in terms of grid allocation. It can help identify when a level is
        nearing its capacity and may need to be refined further.
        """
        return len(self.Grids) / self.num_grids_max


class GridContainer(ElementContainer[tuple[int, ...], "Grid"]):
    """
    Container for managing `Grid` objects within a :py:class:`GridLevel`.

    This class provides methods for lazy loading and unloading of grids,
    adding new grids, and managing the correspondence between grid IDs and
    their HDF5 representations. It provides a high-level interface for
    working with grids at a specific level of a multi-level grid structure,
    enabling efficient data storage and retrieval.

    Parameters
    ----------
    level : :py:class:`GridLevel`
        The `GridLevel` instance to which this container belongs.

    Attributes
    ----------
    level : :py:class:`GridLevel`
        The `GridLevel` instance that owns this `GridContainer`.
    grid_manager : :py:class:`GridManager`
        The `GridManager` instance that owns the parent level.
    GRID_PREFIX : str
        Prefix used to label grids in the HDF5 file. By default, this is "GRID_".
    ERROR_TYPE : type
        The exception type to raise when a grid is not found.
    _index_to_hdf5_cache : dict
        Cache for converting tuple grid indices to HDF5-compatible labels.
    _hdf5_to_index_cache : dict
        Cache for converting HDF5-compatible labels to tuple grid indices.

    Notes
    -----
    This container acts as a specialized dictionary that maps grid IDs, represented
    as tuples of integers, to `Grid` objects. It leverages lazy loading to avoid
    loading all grids into memory at once, only instantiating them as needed.

    See Also
    --------
    :py:class:`GridManager`
        The main manager class responsible for handling grid levels and grids.
    :py:class:`GridLevel`
        Represents a single level in the multi-level grid structure.
    :py:class:`Grid`
        Represents a single grid in the multi-level structure.
    """

    GRID_PREFIX = "GRID_"
    ERROR_TYPE = GridNotFoundError

    def __init__(self, level: GridLevel):
        # Create the basic dictionary container using the
        # grid manager handle.
        super().__init__(level._handle)

        # Set up attributes.
        self.level = level
        self.grid_manager = self.level.grid_manager

    def _index_to_hdf5(self, grid_id: tuple[int, ...]) -> str:
        """
        Convert a tuple grid ID to an HDF5-compatible string.

        This method generates a string label that can be used as a key in the HDF5
        file for storing or retrieving a specific grid. The format is typically
        "GRID_i_j_k", where `i`, `j`, and `k` are the grid indices.

        Parameters
        ----------
        grid_id : tuple of int
            The tuple representing the grid ID. The length of the tuple should
            match the number of dimensions managed by the `GridManager`.

        Returns
        -------
        str
            The HDF5-compatible label for the grid.

        Notes
        -----
        This method uses an internal cache to optimize repeated conversions
        between tuple grid IDs and HDF5-compatible labels.

        Examples
        --------
        .. code-block:: python

            grid_id = (2, 3, 1)
            hdf5_label = grid_container._index_to_hdf5(grid_id)
            # hdf5_label will be 'GRID_2_3_1'
        """
        if grid_id not in self._index_to_hdf5_cache:
            self._index_to_hdf5_cache[
                grid_id
            ] = f"{self.GRID_PREFIX}{'_'.join(map(str, grid_id))}"
        return self._index_to_hdf5_cache[grid_id]

    def _index_from_hdf5(self, grid_id_str: str) -> tuple[int, ...]:
        """
        Convert an HDF5-compatible string label to a tuple grid ID.

        This method reverses the conversion performed by `_index_to_hdf5`, extracting
        the grid indices from a string label and returning them as a tuple of integers.
        The expected format is "GRID_i_j_k", where `i`, `j`, and `k` are the grid indices.

        Parameters
        ----------
        grid_id_str : str
            The HDF5-compatible label for the grid, typically in the format "GRID_i_j_k".

        Returns
        -------
        tuple of int
            The tuple representing the grid ID.

        Notes
        -----
        This method uses an internal cache to optimize repeated conversions between
        HDF5-compatible labels and tuple grid IDs.

        Examples
        --------
        .. code-block:: python

            hdf5_label = 'GRID_2_3_1'
            grid_id = grid_container._index_from_hdf5(hdf5_label)
            # grid_id will be (2, 3, 1)
        """
        if grid_id_str not in self._hdf5_to_index_cache:
            # Assuming the string format is "GRID_i_j_k"
            self._hdf5_to_index_cache[grid_id_str] = tuple(
                map(int, grid_id_str.split("_")[1:])
            )
        return self._hdf5_to_index_cache[grid_id_str]

    def __getitem__(self, grid_id: Union[int, tuple[int, ...]]) -> "Grid":
        """
        Retrieve a grid by its ID.

        This method allows indexing by an integer for 1D grids, and by a tuple for multi-dimensional grids.

        Parameters
        ----------
        grid_id : Union[int, tuple[int, ...]]
            The identifier for the grid. For 1D grids, an integer is allowed. For multi-dimensional grids,
            a tuple of integers is expected.

        Returns
        -------
        Grid
            The grid corresponding to the given grid ID.

        Raises
        ------
        KeyError
            If the grid ID does not exist.

        Notes
        -----
        For 1D grids, you can simply index with an integer, and it will be converted to a tuple.
        For example, `grid_container[0]` will work for 1D grids and be converted to `grid_container[(0,)]`.
        """
        # If the grid is 1D and an integer is provided, convert it to a tuple
        if isinstance(grid_id, int):
            grid_id = (grid_id,)

        return super().__getitem__(grid_id)

    def __setitem__(self, grid_id: Union[int, tuple[int, ...]], value: "Grid") -> None:
        """
        Set a grid by its ID.

        This method allows setting a grid using an integer for 1D grids, and by a tuple for multi-dimensional grids.

        Parameters
        ----------
        grid_id : Union[int, tuple[int, ...]]
            The identifier for the grid. For 1D grids, an integer is allowed. For multi-dimensional grids,
            a tuple of integers is expected.
        value : Grid
            The grid object to set.

        Notes
        -----
        For 1D grids, you can simply index with an integer, and it will be converted to a tuple.
        """
        # If the grid is 1D and an integer is provided, convert it to a tuple
        if isinstance(grid_id, int):
            grid_id = (grid_id,)

        return super().__setitem__(grid_id, value)

    def __delitem__(self, grid_id: Union[int, tuple[int, ...]]):
        """
        Remove a grid from the container and delete its corresponding HDF5 group.

        Parameters
        ----------
        grid_id : tuple[int, ...]
            The identifier for the grid to be removed, represented as a tuple of integers.

        Raises
        ------
        KeyError
            If the `grid_id` does not exist in the container.
        ValueError
            If the grid has children grids, preventing its removal.

        Notes
        -----
        In addition to removing the grid, this method ensures that the grid's parent in
        the previous level will no longer reference the deleted grid as its child. This
        update to the parent grid's children array prevents orphaned grid references.

        Examples
        --------
        .. code-block:: python

            # Remove a grid by its ID
            del grid_container[(1, 0, 2)]
        """
        if isinstance(grid_id, int):
            grid_id = (grid_id,)

        if grid_id not in self:
            raise self.__class__.ERROR_TYPE(f"Grid {grid_id} is not in {self}.")

        grid = self[grid_id]
        if grid.CHILDREN.shape[0] != 0:
            raise ValueError(
                f"Cannot remove grid {grid_id}. It has children: {grid.CHILDREN}."
            )

        # We now need to tell the parent that we deleted the child.
        parent_grid = self.grid_manager[self.level.index - 1][tuple(grid.PARENT)]
        parent_grid.CHILDREN = parent_grid.CHILDREN[
            ~np.all(parent_grid.CHILDREN == np.asarray(grid_id, dtype="int"), axis=1)
        ]

        # Remove the grid's HDF5 group if it exists
        hdf5_key = self._index_to_hdf5(grid_id)
        if hdf5_key in self._handle:
            del self._handle[hdf5_key]

        # Remove the grid from the in-memory container
        super().__delitem__(grid_id)

        self.logger.info("[DEL ] Removed grid %s.", grid_id)

    def load_element(self, grid_id: tuple[int, ...]) -> "Grid":
        """
        Load a `Grid` object for a given grid ID.

        This method lazily loads a `Grid` object for the given grid ID by accessing
        the HDF5 file and retrieving the necessary metadata and data. It only loads
        the grid into memory when needed, avoiding unnecessary memory usage.

        Parameters
        ----------
        grid_id : tuple of int
            The tuple representing the grid ID.

        Returns
        -------
        :py:class:`Grid`
            The loaded `Grid` object.

        Raises
        ------
        :py:class:`GridError`
            If the grid cannot be loaded due to missing data or metadata.

        Notes
        -----
        This method allows on-demand loading of grids, which is particularly useful
        for managing large datasets. It reduces memory usage by only loading grids
        when they are explicitly accessed.

        Examples
        --------
        .. code-block:: python

            grid_id = (1, 0, 2)
            grid = grid_container.load_element(grid_id)
            # The grid with ID (1, 0, 2) is now loaded into memory.
        """
        try:
            self.logger.debug("[LOAD] Reloading grid %s in %s.", grid_id, self)
            return Grid(self.level, grid_id)
        except Exception as e:
            raise GridError(f"Failed to load grid {grid_id}: {e}")

    def load_existing_elements(self):
        """
        Load existing grids from the HDF5 file into the container.

        This method scans the HDF5 file for existing grids, identified by the `GRID_PREFIX`,
        and loads them into the container as placeholders for lazy loading. It does not
        load the actual grid data into memory, but instead sets up the container to access
        them when needed.

        Returns
        -------
        None

        Notes
        -----
        This method is typically called during the initialization of the `GridContainer`
        to populate the container with all existing grids in the HDF5 file. It sets up the
        internal state for efficient lazy loading.

        Examples
        --------
        .. code-block:: python

            grid_container.load_existing_elements()
            # All existing grids in the HDF5 file are now registered in the container.
        """
        for element in self._handle.keys():
            if element.startswith(self.GRID_PREFIX):
                grid_id = self._index_from_hdf5(element)
                self[grid_id] = None  # Placeholder for lazy-loading

    def add_grid(self, indices: ArrayLike, **kwargs) -> "Grid":
        """
        Add a new grid to the container at the specified indices.

        Parameters
        ----------
        indices : ArrayLike
            The grid indices (i, j, k) representing the position within the level's domain.
        kwargs : dict
            Additional parameters for the grid initialization, passed to the Grid constructor.

        Returns
        -------
        Grid
            The newly added Grid instance.

        Raises
        ------
        GridError
            If the indices are invalid, negative, or the grid exceeds the level boundaries.
        ValueError
            If the indices are not integers or outside the level's valid range.
        """
        # Extract the "_validate" argument from kwargs, defaulting to True if not provided.
        _validate = kwargs.pop("_validate", True)

        # Validate and convert grid indices.
        try:
            indices = np.asarray(indices, dtype="int").reshape(
                (self.grid_manager.NDIM,)
            )
        except ValueError as e:
            raise GridError(f"Failed to validate indices in add_grid: {e}.")

        # Check if the grid already exists to avoid duplicates.
        grid_id = tuple(indices)
        if grid_id in self:
            raise GridError(f"Grid with ID {grid_id} already exists in the container.")

        # Perform validation if the _validate flag is set to True.
        if _validate:
            # Check index values.
            if np.any((indices < 0) | (indices >= self.level.TOTAL_REFINEMENT)):
                raise GridError(
                    f"Cannot add grid. Indices {indices} are out of bounds for TOTAL_REFINEMENT={self.level.TOTAL_REFINEMENT}."
                )

            # Calculate bounding box positions.
            left_corner_position = (
                self.grid_manager.BBOX[0, :] + indices * self.level.GRID_SIZE
            )
            right_corner_position = left_corner_position + self.level.GRID_SIZE

            # Validate grid boundaries against the level's domain.
            if np.any(right_corner_position > self.grid_manager.BBOX[1, :]):
                raise GridError(
                    f"Cannot create grid {indices}. The grid boundaries {left_corner_position} to {right_corner_position} "
                    f"exceed the domain boundaries {self.grid_manager.BBOX[1, :]}."
                )
        else:
            # Compute the bounding box without validation.
            left_corner_position = (
                self.grid_manager.BBOX[0, :] + indices * self.level.GRID_SIZE
            )
            right_corner_position = left_corner_position + self.level.GRID_SIZE

        # Build parents and children
        parent_index = indices // self.level.REFINEMENT_FACTOR
        children_index = np.zeros((0, self.grid_manager.NDIM))

        # Tell the parent about the new grid.
        try:
            parent_grid = self.grid_manager[self.level.index - 1][tuple(parent_index)]
        except KeyError:
            raise GridError(
                f"Cannot add grid {grid_id} to level {self.level.index} because its parent"
                f" {tuple(parent_index)} is not in level {self.level.index - 1}."
            )

        if parent_grid.CHILDREN.size == 0:
            parent_grid.CHILDREN = np.array([grid_id], dtype=int)
        else:
            # Append the new child grid_id to the parent's CHILDREN array.
            parent_grid.CHILDREN = np.vstack([parent_grid.CHILDREN, grid_id])

        # Convert to HDF5 string key and create a new HDF5 group for this grid.
        hdf5_key = self._index_to_hdf5(grid_id)
        if hdf5_key in self._handle:
            raise GridError(
                f"HDF5 group for grid {grid_id} already exists in the file."
            )
        _ = self._handle.create_group(hdf5_key)

        # Instantiate the new Grid and add it to the container.
        new_grid = Grid(
            grid_level=self.level,
            grid_id=grid_id,
            BBOX=np.vstack([left_corner_position, right_corner_position]),
            PARENT=parent_index,
            CHILDREN=children_index,
            **kwargs,
        )
        self[grid_id] = new_grid

        return new_grid

    def remove_grid(self, grid_id: tuple[int, ...]) -> None:
        """
        Remove a grid from the container.

        This method is an alias for `__delitem__`, which removes a grid from the container.
        It ensures that the grid is properly removed from memory. The grid's HDF5 data,
        if present, will also be deleted.

        Parameters
        ----------
        grid_id : tuple of int
            The identifier for the grid to be removed. This tuple uniquely represents
            the grid's position within the grid level.

        Raises
        ------
        KeyError
            If the `grid_id` does not exist in the container.

        Notes
        -----
        This method directly deletes the grid from the container and its associated
        HDF5 data. Use this method when you want to permanently remove the grid from
        both memory and persistent storage.

        Examples
        --------
        .. code-block:: python

            grid_id = (1, 0, 2)
            grid_container.remove_grid(grid_id)
            # The grid with ID (1, 0, 2) is removed from memory and its HDF5 data is deleted.
        """
        del self[grid_id]

    def unload_element(self, grid_id: tuple[int, ...]) -> None:
        """
        Unload a specific grid from memory, keeping a placeholder.

        This method removes the in-memory reference to a grid, replacing it with
        a placeholder object that indicates the grid exists but is not currently loaded.
        It is useful for freeing up memory without losing the grid's registration in
        the container.

        Parameters
        ----------
        grid_id : tuple of int
            The index of the grid to unload.

        Raises
        ------
        KeyError
            If the specified index does not exist in the container.

        Notes
        -----
        The unloaded grid can be reloaded into memory at any time using the `load_element`
        method. This is particularly useful for managing large datasets with limited memory.

        Examples
        --------
        .. code-block:: python

            grid_id = (0, 1, 1)
            grid_container.unload_element(grid_id)
            # The grid with ID (0, 1, 1) is now unloaded from memory.
        """
        if grid_id not in self:
            raise KeyError(f"Grid {grid_id} is not in {self}.")
        self.logger.debug("[LOAD] Unloading grid %s in %s.", grid_id, self)
        super().__setitem__(grid_id, None)

    @property
    def logger(self) -> "logging.Logger":
        """
        Return the logger associated with the `GridManager`.

        This property provides access to the logger used for logging messages related
        to this `GridContainer`. It is typically used for debugging, information, and
        error messages related to grid management.

        Returns
        -------
        logging.Logger
            The logger for this `GridContainer`.

        Notes
        -----
        The logger is shared with the `GridManager` that owns this container. This
        ensures consistent logging behavior across all levels and grids.

        Examples
        --------
        .. code-block:: python

            grid_container.logger.info("This is an info message for the GridContainer.")
            # Logs the message with the logger associated with the GridManager.
        """
        return self.grid_manager.logger


class Grid:
    """
    Represents a single grid within a `GridLevel`.

    A `Grid` object corresponds to a specific region of space within a `GridLevel`,
    and it manages fields associated with that region. Each `Grid` is identified
    by a unique index tuple and is backed by metadata stored in an HDF5 file.

    Parameters
    ----------
    grid_level : :py:class:`GridLevel`
        The `GridLevel` instance to which this grid belongs.
    grid_id : tuple of int
        The unique identifier for this grid within its level. This is typically a tuple
        of integers representing the grid's position within the level's domain.
    **kwargs : dict, optional
        Additional metadata such as bounding box (`BBOX`), passed to the metadata descriptors.

    Attributes
    ----------
    level : :py:class:`GridLevel`
        The `GridLevel` instance that owns this `Grid`.
    grid_manager : :py:class:`GridManager`
        The `GridManager` instance that owns the parent level.
    index : tuple of int
        The unique identifier for this grid within its level.
    _handle : h5py.Group
        The HDF5 group handle associated with this grid.
    Fields : :py:class:`FieldContainer`
        A container for managing `Field` objects within this grid.
    PARENT : tuple of int
        The indices of the parent grid in the previous level. If the grid is at the
        base level, it is set to the grid's own indices.
    CHILDREN : list of tuple of int
        A list of indices for all existing child grids in the next finer level. If the
        grid is at the finest level or has no children, an empty list is returned.

    Notes
    -----
    A `Grid` is responsible for storing and managing the physical fields associated
    with a particular region of the grid level. Each grid can have multiple fields,
    and these fields can be lazily loaded from the HDF5 file to optimize memory usage.

    See Also
    --------
    :py:class:`GridLevel`
        Represents a single level in the multi-level grid structure.
    :py:class:`Field`
        Represents a physical field associated with a specific grid.
    """

    BBOX: BoundingBox = GridMetadataDescriptor(required=True)
    PARENT: NDArray[int] = GridMetadataDescriptor(required=True)
    CHILDREN: NDArray[int] = GridMetadataDescriptor(required=True)

    # Cache for storing references to the MetadataDescriptors.
    _metadata_cache: ClassVar[dict[str, "GridMetadataDescriptor"]] = {}

    def __init__(self, grid_level: "GridLevel", grid_id: tuple[int, ...], **kwargs):
        """
        Initialize the Grid object.

        Parameters
        ----------
        grid_level : GridLevel
            The grid level to which this grid belongs.
        grid_id : int
            The unique identifier for this grid within its level.
        **kwargs : dict, optional
            Additional metadata such as bounding box (`BBOX`), passed to the metadata descriptors.
        """
        # Create general attributes.
        self.level = grid_level
        self.grid_manager = self.level.grid_manager
        self.index = grid_id
        self.grid_manager.logger.debug(
            f"[INIT] Loading Grid {grid_id} of level {self.level.index}..."
        )
        self._handle = self.level._handle[
            f"GRID_{'_'.join([str(i) for i in self.index])}"
        ]

        # Load metadata from the provided kwargs or from the HDF5 file.
        self._meta = {}
        self._initialize_metadata(kwargs)

        # Derive additional metadata if possible / needed.
        self._initialize_additional_metadata()

        # Initialize the Fields container (in-memory or HDF5-backed)
        self.Fields = FieldContainer(self)

    def _initialize_metadata(self, kwargs: dict):
        """
        Initialize metadata from kwargs or the HDF5 file.

        This method initializes the grid's metadata, such as the bounding box (`BBOX`),
        from the provided keyword arguments or by loading it from the associated HDF5 file.

        Parameters
        ----------
        kwargs : dict
            Additional metadata attributes for initialization.

        Raises
        ------
        :py:class:`GridMetadataError`
            If required metadata is missing or incompatible.

        Notes
        -----
        This method is called during the grid's initialization to set up its
        metadata attributes. It ensures that all required metadata is properly
        loaded and validated, either from the provided arguments or from the
        HDF5 file.

        Examples
        --------
        .. code-block:: python

            grid_metadata = {'BBOX': [[0, 0, 0], [1, 1, 1]]}
            grid._initialize_metadata(grid_metadata)
            # The grid's bounding box is now initialized with the provided metadata.
        """

        for _, meta_class in self._get_metadata_descriptors().items():
            meta_class.set_metadata_from_kwargs_or_hdf5(self, kwargs)

    def _initialize_additional_metadata(self):
        """
        Initialize additional metadata based on existing metadata.

        This method derives additional metadata attributes that are not directly
        provided, but can be computed from existing metadata. For example, it
        calculates the grid's shape based on the block size.

        Notes
        -----
        This method is called after the primary metadata has been initialized.
        It ensures that all derived attributes are properly set up before the
        grid is used.

        Raises
        ------
        :py:class:`GridMetadataError`
            If additional metadata cannot be derived from existing metadata.

        Examples
        --------
        .. code-block:: python

            grid._initialize_additional_metadata()
            # Additional metadata, such as grid shape, is now initialized.
        """

        self.SHAPE = self.grid_manager.BLOCK_SIZE

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the Grid object for debugging.

        Returns
        -------
        str
            A detailed string representation including the grid's level, index, shape, and metadata.
        """
        metadata = {
            key: getattr(self, key, None)
            for key in self._get_metadata_descriptors().keys()
        }
        return (
            f"<Grid(level={self.level.index}, index={self.index}, "
            f"shape={self.SHAPE}, metadata={metadata})>"
        )

    def __str__(self):
        """
        User-friendly string representation of the GridLevel.

        Returns
        -------
        str
            A simpler string summarizing the level's index and number of grids.
        """
        return f"<Grid {self.level.index}:{self.index}>"

    def __len__(self) -> int:
        """
        Return the number of grids in this GridLevel.

        Returns
        -------
        int
            Number of grids managed by the GridLevel.
        """
        return len(self.Fields)

    def __getitem__(self, field_name: str) -> "Field":
        """
        Retrieve a field from the grid by name.

        Parameters
        ----------
        field_name : str
            The name of the field to retrieve.

        Returns
        -------
        Field
            The requested Field object.

        Raises
        ------
        KeyError
            If the field name does not exist in the grid.
        """
        if field_name not in self.Fields:
            raise KeyError(f"Field '{field_name}' not found in Grid {self.index}.")
        return self.Fields[field_name]

    def __setitem__(self, field_name: str, field: "Field") -> None:
        """
        Set or update a field in the grid.

        Parameters
        ----------
        field_name : str
            The name of the field to set.
        field : Field
            The Field object to be set in the grid.

        Raises
        ------
        ValueError
            If the field name is invalid or cannot be set.
        """
        self.Fields[field_name] = field

    def __delitem__(self, field_name: str) -> None:
        """
        Remove a field from the grid.

        Parameters
        ----------
        field_name : str
            The name of the field to remove.

        Raises
        ------
        KeyError
            If the field name does not exist in the grid.
        """
        if field_name not in self.Fields:
            raise KeyError(f"Field '{field_name}' not found in Grid {self.index}.")
        del self.Fields[field_name]

    def __contains__(self, field_name: str) -> bool:
        """
        Check if a field exists in the grid.

        Parameters
        ----------
        field_name : str
            The name of the field to check.

        Returns
        -------
        bool
            True if the field exists in the grid, False otherwise.
        """
        return field_name in self.Fields

    def add_field(
        self, name: str, dtype: np.dtype, units: str, register: bool = True, **kwargs
    ):
        """
        Add a new field to the grid.

        This method creates and adds a new field to the grid with the specified name, data type,
        and units. If successful, it returns the newly created `Field` object. The field is
        registered within the grid and made available for further operations.

        Parameters
        ----------
        name : str
            The name of the new field to be added.
        dtype : numpy.dtype
            The data type of the field, such as `numpy.float64` or `numpy.int32`.
        units : str
            The physical units associated with the field, expressed as a string (e.g., "g/cm^3").
        register : bool, optional
            If `True`, the field is registered with the grid's `Fields` container (default is `True`).
        kwargs : dict, optional
            Additional keyword arguments to pass to the `add_field` method in the `Fields` container.

        Returns
        -------
        Field
            The newly created `Field` instance associated with this grid.

        Raises
        ------
        ValueError
            If a field with the specified name already exists in the grid.

        Notes
        -----
        This method delegates the actual field creation to the `Fields` container within the grid.
        It checks for duplicate field names and raises an error if a field with the same name
        already exists in the grid.

        Examples
        --------
        .. code-block:: python

            # Add a new "density" field to the grid with float64 data type and "g/cm^3" units.
            new_field = grid.add_field("density", dtype=np.float64, units="g/cm^3")

            # Add a new "temperature" field with additional options.
            new_field = grid.add_field("temperature", dtype=np.float32, units="K", register=False)
        """
        return self.Fields.add_field(name, dtype, units, register=register, **kwargs)

    def add_field_from_function(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        field_name: str,
        dtype: np.dtype = "f8",
        units: str = "",
        geometry: "GeometryHandler" = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Add a new field to the grid by applying a function to the grid coordinates.

        This method adds a new field to the grid, where the field values are generated
        by applying the provided function to the grid's coordinates. The function can
        optionally be wrapped with a geometry handler to transform the grid coordinates
        before computing the field values.

        Parameters
        ----------
        function : Callable[[np.ndarray], np.ndarray]
            A function that takes the grid coordinates as input and returns the field values.
            The input to this function is a numpy array of coordinates with shape `(NDIM, *BLOCK_SIZE)`.
        field_name : str
            The name of the field to be added.
        dtype : np.dtype, optional
            The data type of the field (default is 'f8', double precision floating point).
        units : str, optional
            The physical units of the field, expressed as a string (default is an empty string).
        geometry : GeometryHandler, optional
            A geometry handler object that can transform the grid coordinates before applying the function.
        overwrite : bool, optional
            Whether to overwrite the field if it already exists (default is False).
        kwargs : dict, optional
            Additional keyword arguments for field creation.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the field already exists and `overwrite` is set to False.

        Notes
        -----
        The function provided should generate field values based on the grid's spatial coordinates.
        If a geometry handler is provided, the grid coordinates will be transformed before
        being passed to the function. This is useful for fields that depend on specific geometric
        transformations (e.g., spherical or cylindrical coordinates).

        Examples
        --------
        .. code-block:: python

            # Define a function for the field
            def density_function(coords):
                x, y, z = coords
                return np.exp(-x**2 - y**2 - z**2)

            # Add a new "density" field to the grid
            grid.add_field_from_function(density_function, "density", dtype=np.float64, units="g/cm^3")
        """
        # Apply geometry transformation if provided
        final_function = (
            function
            if geometry is None
            else lambda grid_coords: function(
                *geometry.build_converter(self.AXES)(grid_coords)
            )
        )

        # Check if the field already exists
        if field_name in self.Fields and not overwrite:
            raise ValueError(
                f"Field '{field_name}' already exists in the grid. Set `overwrite=True` to replace it."
            )

        # Add the field to the grid
        self.add_field(
            field_name, dtype=dtype, units=units, register=kwargs.pop("register", False)
        )

        # Compute coordinates and evaluate the function to generate field values
        coords = self.opt_get_coordinates(
            self.BBOX, self.grid_manager.BLOCK_SIZE, self.level.CELL_SIZE
        )
        field_data = final_function(coords)

        # Set the field values
        self.Fields[field_name][:] = field_data

        # Commit changes after processing each level
        self.grid_manager.commit_changes()

    def add_field_from_profile(
        self,
        profile: "Profile",
        field_name: str,
        dtype: np.dtype = "f8",
        units: str = "",
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Add a new field to the grid using a Profile object.

        This method adds a field to the grid, where the field values are generated using
        a Profile object. The Profile object provides a callable function to compute the field
        values based on the grid coordinates. The Profile may also include a geometry handler
        to transform the coordinates before applying the function.

        Parameters
        ----------
        profile : Profile
            A Profile object that defines a callable function to generate field values. The Profile
            may also contain a geometry handler to transform the grid coordinates.
        field_name : str
            The name of the field to be added.
        dtype : np.dtype, optional
            The data type for the field (default is 'f8', double precision floating point).
        units : str, optional
            The physical units of the field (default is an empty string).
        overwrite : bool, optional
            Whether to overwrite the field if it already exists (default is False).
        kwargs : dict, optional
            Additional keyword arguments for field creation.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the field already exists and `overwrite` is set to False.

        Notes
        -----
        This method leverages the Profile's callable function to compute field values.
        The Profile may optionally include a geometry handler to apply transformations
        to the grid coordinates before generating the field values.

        Examples
        --------
        .. code-block:: python

            # Assuming we have a Profile object named 'density_profile'
            grid.add_field_from_profile(density_profile, "density", dtype=np.float64, units="g/cm^3")
        """
        # Call the add_field_from_function method using the Profile's function and geometry handler
        self.add_field_from_function(
            function=profile,
            field_name=field_name,
            dtype=dtype,
            units=units,
            geometry=profile.geometry_handler,
            overwrite=overwrite,
            **kwargs,
        )

    @classmethod
    def _get_metadata_descriptors(cls) -> dict[str, "GridMetadataDescriptor"]:
        """
        Retrieve all metadata descriptors for the `Grid` class, including inherited ones.

        This method collects all metadata descriptors defined in the `Grid` class and
        its parent classes. It uses caching to avoid recomputation, improving performance
        for repeated calls.

        Returns
        -------
        dict
            A dictionary of metadata descriptors, with attribute names as keys and
            descriptor instances as values.

        Notes
        -----
        Metadata descriptors are used to manage the metadata attributes of the grid,
        such as the bounding box (`BBOX`). This method provides a convenient way to
        access all descriptors in one place.

        Examples
        --------
        .. code-block:: python

            descriptors = Grid._get_metadata_descriptors()
            # Returns a dictionary of metadata descriptors for the Grid class.
        """
        import inspect

        if not cls._metadata_cache:
            descriptors = {}
            for base_class in inspect.getmro(cls):
                for name, attr in vars(base_class).items():
                    if isinstance(attr, GridMetadataDescriptor):
                        descriptors[name] = attr
            cls._metadata_cache = descriptors
        return cls._metadata_cache

    @staticmethod
    def opt_get_coordinates(
        bbox: NDArray[float], block_size: NDArray[int], cell_size: NDArray[float]
    ) -> NDArray[float]:
        # Generate slices in the form start:stop:n, where n is the number of points
        slices = tuple(
            slice(bbox[0, i] + (cell_size / 2), bbox[1, i], complex(0, block_size[i]))
            for i in range(len(block_size))
        )

        return np.mgrid[slices]

    def get_coordinates(self) -> np.ndarray:
        """
        Get the coordinates of each cell in the grid.

        This method calculates the physical coordinates of each cell in the grid based on the grid's
        bounding box (`BBOX`) and its shape (`BLOCK_SIZE`). The resulting coordinates are returned as
        a numpy array with shape `(NDIM, *BLOCK_SIZE)`, where `NDIM` is the number of spatial dimensions
        and `BLOCK_SIZE` defines the number of cells in each dimension.

        Returns
        -------
        np.ndarray
            An array of shape `(NDIM, *BLOCK_SIZE)` containing the coordinates of each cell in the grid.
            The first axis corresponds to the spatial dimensions (e.g., x, y, z), and the remaining axes
            correspond to the grid's cell indices in each dimension.

        Examples
        --------
        .. code-block:: python

            # Assuming a 2D grid with BLOCK_SIZE = (10, 10) and BBOX = [[0, 0], [1, 1]]
            coords = grid.get_coordinates()
            # coords will have shape (2, 10, 10), where:
            # coords[0, ...] represents the x-coordinates of each cell.
            # coords[1, ...] represents the y-coordinates of each cell.
        """
        return self.opt_get_coordinates(
            self.BBOX, self.grid_manager.BLOCK_SIZE, self.level.CELL_SIZE
        )

    @property
    def logger(self):
        """
        Access the logger associated with the `GridManager`.

        This property provides access to the logger used for logging messages related
        to this `Grid`. It is typically used for debugging, information, and error
        messages specific to grid operations.

        Returns
        -------
        logging.Logger
            The logger for this `Grid`.

        Notes
        -----
        The logger is shared with the `GridManager` that owns this grid. This ensures
        consistent logging behavior across all grids and levels managed by the same
        `GridManager`.

        Examples
        --------
        .. code-block:: python

            grid.logger.info("This is an info message for the Grid.")
            # Logs the message with the logger associated with the GridManager.
        """
        return self.grid_manager.logger


class FieldContainer(ElementContainer[str, "Field"]):
    """
    Container for managing `Field` objects within a `Grid`.

    This class provides methods for adding, removing, and loading fields
    associated with a specific `Grid` instance. Each field corresponds
    to a physical quantity on the grid and is backed by an HDF5 dataset.

    Parameters
    ----------
    grid : :py:class:`Grid`
        The `Grid` instance to which this `FieldContainer` belongs.

    Attributes
    ----------
    grid : :py:class:`Grid`
        The `Grid` instance that owns this `FieldContainer`.
    level : :py:class:`GridLevel`
        The `GridLevel` instance that owns the parent grid.
    grid_manager : :py:class:`GridManager`
        The `GridManager` instance that owns the parent level.
    ERROR_TYPE : type
        The exception type to raise when a field is not found.
    _index_to_hdf5_cache : dict
        Cache for converting field names to HDF5-compatible labels.
    _hdf5_to_index_cache : dict
        Cache for converting HDF5-compatible labels to field names.

    Notes
    -----
    The `FieldContainer` class is designed to manage multiple fields
    associated with a single grid. It allows for lazy loading of fields
    from an HDF5 file and provides methods for adding new fields and
    removing existing ones.

    See Also
    --------
    :py:class:`Grid`
        Represents a single grid within a `GridLevel`.
    :py:class:`Field`
        Represents a physical field associated with a specific grid.
    """

    ERROR_TYPE = FieldNotFoundError

    def __init__(self, grid: "Grid"):
        # Create the basic dictionary container using the grid's handle.
        super().__init__(grid._handle)

        # Set up attributes.
        self.grid = grid
        self.level = self.grid.level
        self.grid_manager = self.level.grid_manager

    def _index_to_hdf5(self, grid_id: str) -> str:
        """Convert a tuple grid ID to an HDF5-compatible string."""
        return grid_id

    def _index_from_hdf5(self, grid_id_str: str) -> str:
        """Convert an HDF5-compatible string to a tuple grid ID."""
        return grid_id_str

    def __delitem__(self, field_name: str):
        """
        Remove a field from the container and delete its corresponding HDF5 dataset.

        This method removes a field from the in-memory container and deletes
        its associated HDF5 dataset, if it exists. It raises a `KeyError`
        if the field is not found in the container.

        Parameters
        ----------
        field_name : str
            The name of the field to be removed.

        Raises
        ------
        KeyError
            If the field name does not exist in the container.
        IOError
            If the HDF5 handle is closed or the field's HDF5 dataset cannot be deleted.

        Notes
        -----
        Removing a field also deletes the corresponding HDF5 dataset,
        freeing up space in the HDF5 file.

        Examples
        --------
        .. code-block:: python

            field_container.__delitem__("density")
            # Removes the "density" field from the container and HDF5 file.
        """
        if field_name not in self:
            raise self.__class__.ERROR_TYPE(f"Field '{field_name}' is not in {self}.")

        # Remove the field's HDF5 dataset if it exists
        hdf5_key = self._index_to_hdf5(field_name)
        if hdf5_key in self._handle:
            del self._handle[hdf5_key]

        # Remove the field from the in-memory container
        super().__delitem__(field_name)
        self.logger.debug("[DEL ] Removed field '%s'.", field_name)

    def load_element(self, index: str) -> "Field":
        """
        Load a `Field` object by name, creating a new `Field` instance.

        This method lazily loads a `Field` object from the HDF5 file by its
        name, creating a new `Field` instance and returning it.

        Parameters
        ----------
        index : str
            The name of the field to load.

        Returns
        -------
        :py:class:`Field`
            The loaded `Field` object.

        Notes
        -----
        This method is used to retrieve a `Field` from the container when
        it is not already loaded in memory.

        Examples
        --------
        .. code-block:: python

            field = field_container.load_element("temperature")
            # Loads the "temperature" field from the HDF5 file.
        """

        self.logger.debug("[LOAD] Reloading field %s in %s.", index, self)
        return Field(index, self.grid)

    def load_existing_elements(self):
        """
        Load all existing fields from the HDF5 file into the container.

        This method scans the HDF5 file for existing fields and loads them
        into the container as placeholders for lazy loading. It does not
        fully load the field data into memory, but makes them available
        for subsequent access.

        Returns
        -------
        None

        Notes
        -----
        This method is typically called during the initialization of a
        `FieldContainer` to populate the container with existing fields
        from the HDF5 file.

        Examples
        --------
        .. code-block:: python

            field_container.load_existing_elements()
            # Loads all existing fields from the HDF5 file.
        """

        for element in self._handle.keys():
            self[element] = None

    def add_field(
        self, name: str, dtype: np.dtype, units: str, register: bool = True, **kwargs
    ) -> "Field":
        """
        Add a new field to the container.

        This method creates a new field in the HDF5 file and adds it to the
        in-memory container. The new field is initialized with the specified
        name, data type, and units.

        Parameters
        ----------
        name : str
            The name of the new field.
        dtype : numpy.dtype
            The data type of the field.
        units : str
            The units of the field.
        register : bool, optional
            If True, the field is registered with the `GridManager`'s field registry.
            This helps track all fields created across grids. Set to False if you do
            not want the field registered, typically for temporary fields.
        **kwargs : dict, optional
            Additional parameters for the field.

        Returns
        -------
        :py:class:`Field`
            The newly created `Field` instance.

        Raises
        ------
        ValueError
            If the field already exists in the container.

        Examples
        --------
        .. code-block:: python

            new_field = field_container.add_field("velocity", dtype=np.float64, units="m/s")
            # Adds a new "velocity" field to the container and HDF5 file.
        """
        if name in self:
            raise ValueError(
                f"Field '{name}' already exists in grid {self.grid.index}."
            )

        # Create a new HDF5 dataset in the grid's HDF5 group
        dataset = self._handle.create_dataset(name, shape=self.grid.SHAPE, dtype=dtype)
        dataset.attrs["units"] = units

        # Initialize the Field object and add it to the container
        new_field = Field(name, self.grid)
        self[name] = new_field

        # We need to register it in the Fields directory if its new. We use the register
        # kwarg because its a faster check that a container search. Thus, if we are adding
        # in bulk, we can skip this and lower the overhead.
        if register and (name not in self.grid_manager.Fields):
            self.grid_manager.Fields.register_field(name, units, dtype, **kwargs)

        self.logger.debug(f"[ADD ] Added field '{name}' to grid {self.grid.index}.")
        return new_field

    def unload_element(self, index: str):
        """
        Unload a specific field from memory, keeping a placeholder.

        This method unloads a field from memory, replacing it with a placeholder
        in the container. The field's data remains stored in the HDF5 file and
        can be reloaded later if needed.

        Parameters
        ----------
        index : str
            The name of the field to unload.

        Raises
        ------
        KeyError
            If the field name does not exist in the container.

        Notes
        -----
        Unloading a field frees up memory, but the field's metadata remains
        accessible in the container.

        Examples
        --------
        .. code-block:: python

            field_container.unload_element("velocity")
            # Unloads the "velocity" field from memory, keeping a placeholder.
        """

        if index not in self:
            raise KeyError(f"{index} is not in {self}.")
        self.logger.debug("[LOAD] Unloading field %s in %s.", index, self)
        super().__setitem__(index, None)

    @property
    def logger(self):
        """
        Return the logger associated with the `GridManager`.

        This property provides access to the logger used for logging messages
        related to this `FieldContainer`. It is typically used for debugging,
        information, and error messages specific to field operations.

        Returns
        -------
        logging.Logger
            The logger for this `FieldContainer`.

        Notes
        -----
        The logger is shared with the `GridManager` that owns this container.
        This ensures consistent logging behavior across all fields and grids
        managed by the same `GridManager`.

        Examples
        --------
        .. code-block:: python

            field_container.logger.info("This is an info message for the FieldContainer.")
            # Logs the message with the logger associated with the GridManager.
        """

        return self.grid_manager.logger


class Field(unyt_array):
    """
    Represents a physical field on a grid, stored as an HDF5-backed dataset.

    A `Field` object corresponds to a physical quantity defined on a specific
    grid. It is stored as a dataset in the HDF5 file and can be lazily loaded
    and accessed as needed. The `Field` class inherits from `unyt_array` and
    integrates unit-aware array operations.

    Parameters
    ----------
    name : str
        The name of the field. Must correspond to a dataset in the grid's HDF5 group.
    grid : :py:class:`Grid`
        The `Grid` instance to which this field belongs.
    **kwargs : dict, optional
        Additional parameters for the field initialization, such as units.

    Attributes
    ----------
    name : str
        The name of the field.
    units : :py:class:`unyt.Unit`
        The units of the field.
    dtype : numpy.dtype
        The data type of the field.
    buffer : h5py.Dataset
        The HDF5 dataset that backs this field.
    _grid : :py:class:`Grid`
        The `Grid` object associated with this field.

    Notes
    -----
    The `Field` class allows for efficient storage and retrieval of physical
    quantities associated with a grid. It supports unit-aware arithmetic
    operations and is designed to integrate seamlessly with the `Grid` and
    `GridLevel` classes.

    See Also
    --------
    :py:class:`Grid`
        Represents a single grid within a `GridLevel`.
    :py:class:`GridLevel`
        Represents a single level in the multi-level grid structure.
    """

    def __new__(cls, name: str, grid: "Grid", **kwargs):
        """
        Create a new `Field` object that references an HDF5-backed dataset.

        This method initializes a `Field` object by verifying the existence of
        the corresponding HDF5 dataset, validating its metadata, and associating
        it with the parent `Grid`.

        Parameters
        ----------
        name : str
            The name of the field (must correspond to a dataset in the grid's HDF5 group).
        grid : :py:class:`Grid`
            The `Grid` instance to which this field belongs.

        Returns
        -------
        :py:class:`Field`
            A new `Field` instance.

        Raises
        ------
        KeyError
            If the field name does not exist in the grid's HDF5 group.
        ValueError
            If the units attribute is missing or the dataset has an unsupported data type.

        Notes
        -----
        The method ensures that the HDF5 dataset associated with the field is valid and
        that the units are properly parsed and compatible with `unyt`. If the dataset’s
        dtype is not numeric, the field will not be created.

        Examples
        --------
        .. code-block:: python

            field = Field("density", grid)
            # Creates a new Field instance for the "density" dataset in the grid.
        """
        try:
            # Verify the dataset exists in the grid's HDF5 group.
            if name not in grid._handle:
                raise KeyError(
                    f"Field '{name}' not found in the HDF5 group for grid {grid.index}."
                )

            dataset = grid._handle[name]

            # Validate units and dtype, and provide more meaningful error messages.
            units_str = dataset.attrs.get("units")
            if units_str is None:
                raise ValueError(
                    f"Field '{name}' in grid {grid.index} is missing 'units' attribute."
                )

            try:
                units = Unit(units_str)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse units '{units_str}' for field '{name}' in grid {grid.index}: {e}"
                )

            # Validate dtype
            dtype = dataset.dtype
            if not np.issubdtype(dtype, np.number):
                raise ValueError(
                    f"Field '{name}' in grid {grid.index} has unsupported data type '{dtype}'."
                )

        except KeyError as e:
            raise KeyError(f"Failed to create Field: {e}")
        except ValueError as e:
            raise ValueError(f"Failed to create Field: {e}")

        # Initialize the unyt_array with empty data, using units and dtype.
        obj = super().__new__(cls, [], units=units)
        obj.name = name
        obj.units = units
        obj.dtype = dtype
        obj.buffer = dataset

        # Associate the Field with the grid.
        obj._grid = grid

        return obj

    def __setitem__(self, key: slice | int, value: Any):
        """
        Set a slice of the data in the field, ensuring units are consistent.

        This method updates a portion of the field's data, ensuring that the
        units of the new data are compatible with the field's units.

        Parameters
        ----------
        key : slice or int
            The slice or index of data to set.
        value : array-like
            The data to insert into the field.

        Raises
        ------
        ValueError
            If the value's units are incompatible with the field's units.

        Notes
        -----
        The field's HDF5 dataset is updated with the new data. This operation
        respects the field's units, converting the new data if necessary.

        Examples
        --------
        .. code-block:: python

            field[0:10, 0:10, 0:10] = unyt_array(np.ones((10, 10, 10)), units="g/cm**3")
            # Sets a portion of the field's data to a new array with compatible units.
        """
        try:
            if isinstance(value, unyt_array) and not value.units.is_compatible(
                self.units
            ):
                raise ValueError(
                    f"Units of value '{value.units}' are not compatible with field '{self.name}' units '{self.units}'."
                )

            # Write the data to the HDF5 dataset buffer
            self.buffer[key] = (
                value.to(self.units) if isinstance(value, unyt_array) else value
            )
        except Exception as e:
            raise ValueError(f"Failed to set data for field '{self.name}': {e}")

    def __getitem__(self, key: slice | int) -> unyt_array:
        """
        Retrieve a slice of the data from the field.

        This method retrieves a portion of the field's data, returning it as
        a `unyt_array` with the appropriate units.

        Parameters
        ----------
        key : slice or int
            The slice or index of data to retrieve.

        Returns
        -------
        :py:class:`unyt_array`
            The retrieved data, with units.

        Raises
        ------
        KeyError
            If the key is not a valid slice or index.

        Notes
        -----
        This method accesses the field's HDF5 dataset and returns the requested
        data as a `unyt_array`, preserving the field's units.

        Examples
        --------
        .. code-block:: python

            data = field[0:10, 0:10, 0:10]
            # Retrieves a portion of the field's data as a `unyt_array`.
        """

        try:
            data_slice = self.buffer[key]
            return unyt_array(data_slice, units=self.units)
        except Exception as e:
            raise KeyError(f"Failed to retrieve data from field '{self.name}': {e}")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Override numpy's ufunc behavior to ensure operations on `Field` return a `unyt_array`.

        This method customizes the behavior of numpy's universal functions (ufuncs) to
        ensure that operations involving `Field` objects return `unyt_array` instances
        with the appropriate units.

        Parameters
        ----------
        ufunc : numpy.ufunc
            The ufunc to be applied (e.g., addition, multiplication).
        method : str
            The ufunc method to apply.
        *inputs : tuple
            The input arrays for the ufunc.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        :py:class:`unyt_array`
            The result of the operation as a `unyt_array`.

        Notes
        -----
        This method ensures that arithmetic operations involving `Field` objects
        maintain unit consistency and return results in the expected format.

        Examples
        --------
        .. code-block:: python

            result = np.add(field1, field2)
            # Adds two `Field` objects and returns the result as a `unyt_array`.
        """

        def to_unyt_array(x):
            if isinstance(x, Field):
                return x.view(unyt_array)
            return x

        inputs = tuple(to_unyt_array(x) for x in inputs)
        result = ufunc(*inputs, **kwargs)

        if isinstance(result, unyt_array):
            result.units = self.units

        return result

    def __repr__(self) -> str:
        grid_info = (
            f"{self._grid.level.index}-{self._grid.index}" if self._grid else "NONE"
        )
        return f"<Field {grid_info}/{self.name}, units={self.units}, HDF5-backed>"

    def __str__(self) -> str:
        grid_info = (
            f"{self._grid.level.index}-{self._grid.index}" if self._grid else "NONE"
        )
        return f"<Field {grid_info}/{self.name}, units={self.units}>"

    @property
    def grid(self) -> "Grid":
        """
        Get the grid this field is associated with.

        This property returns the `Grid` object that the field is associated
        with. It allows for easy access to the parent grid from a `Field` object.

        Returns
        -------
        :py:class:`Grid`
            The grid object associated with this field.

        Notes
        -----
        This property provides a convenient way to access the parent grid of
        a `Field` object, enabling navigation between the field and its grid.

        Examples
        --------
        .. code-block:: python

            parent_grid = field.grid
            # Returns the grid associated with the field.
        """
        return self._grid
