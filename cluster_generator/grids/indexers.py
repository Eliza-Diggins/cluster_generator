from typing import TYPE_CHECKING, Any, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from cluster_generator.grids._types import ElementContainer

if TYPE_CHECKING:
    from cluster_generator.grids.managers import GridManager


class GridIndex:
    def __init__(
        self,
        bbox: NDArray[float],
        resolution: NDArray[int],
        grid_manager: "GridManager",
    ):
        """
        Initialize the GridIndex with the given bounding box, resolution, and GridManager.

        Parameters
        ----------
        bbox : np.ndarray
            A 2xN array representing the bounding box of the grid index. The first row contains the minimum
            coordinates (lower bounds) for each spatial dimension, and the second row contains the maximum
            coordinates (upper bounds). This defines the spatial extent of the grid.
        resolution : np.ndarray
            An array of integers representing the number of cells along each dimension in the grid index. This
            resolution determines the number of subdivisions within the bounding box.
        grid_manager : GridManager
            An instance of the GridManager class that manages the grids and coordinates operations like loading
            and unloading grid data.
        """
        self.BBOX = np.asarray(bbox, dtype=np.float64)
        self.RES = np.asarray(resolution, dtype="uint32")
        self.CELL_SIZE = (self.BBOX[1, :] - self.BBOX[0, :]) / self.RES
        self.grid_manager = grid_manager

        self.grid_manager.logger.info(
            "Constructing %s buffer for %s.", self.RES, self.grid_manager
        )

        self.NDIM = self.RES.size
        self.grid_manager.logger.debug("\t Computing coordinates...")
        self.coordinates = self._construct_coordinates()

        self.grid_manager.logger.debug("\t Constructing buffer...")
        self.buffer = np.zeros(
            (*self.RES, self.grid_manager.NDIM + 1), dtype=int
        )  # Initialize buffer for indices
        self.populate_index()  # Populate the index with grid data

    @staticmethod
    def opt_get_coordinates(
        bbox: NDArray[float], block_size: NDArray[int], cell_size: NDArray[float]
    ) -> NDArray[float]:
        """
        Optimized function to get the coordinates of each cell in the grid using `np.mgrid`.

        Parameters
        ----------
        bbox : np.ndarray
            A 2xN array representing the bounding box of the grid. The first row contains the minimum
            coordinates (lower bounds) for each spatial dimension, and the second row contains the maximum
            coordinates (upper bounds).
        block_size : np.ndarray
            An array of integers representing the number of cells along each dimension.
        cell_size: np.ndarray
            The size of a single cell along each dimension.

        Returns
        -------
        np.ndarray
            An array of shape `(N, *block_size)` where `N` is the number of dimensions. Each slice along the
            first axis contains the coordinates of the cells in that dimension, defining a mesh grid of points
            within the bounding box.

        Notes
        -----
        This method uses `np.mgrid` to efficiently compute the coordinates for each cell in the grid based on
        the bounding box and resolution (block size). The result is a structured array of coordinates that
        correspond to the center points of the cells.
        """
        # Generate slices in the form start:stop:n, where n is the number of points
        slices = tuple(
            slice(
                bbox[0, i] + (0.5 * cell_size[i]),
                bbox[1, i] - (0.5 * cell_size[i]),
                complex(0, block_size[i]),
            )
            for i in range(len(block_size))
        )
        return np.mgrid[slices]

    def _construct_coordinates(self) -> NDArray[float]:
        """
        Construct the coordinates buffer based on the bounding box and resolution.

        The coordinates will be evenly spaced within the specified bounding box for the given resolution.

        Returns
        -------
        np.ndarray
            An array of shape `(NDIM, *resolution)` containing the coordinates of each cell in the grid index.
        """
        return self.opt_get_coordinates(self.BBOX, self.RES, self.CELL_SIZE)

    def populate_index(self) -> None:
        """
        Populate the index buffer with references to the grid levels and IDs using hierarchical indexing.

        This method maps each point in the index grid to the corresponding grid in the `GridManager` using
        the hierarchical relationships defined by the PARENT and CHILDREN attributes. The buffer stores
        references to the most refined grids and their respective levels, effectively building a lookup
        table for fast querying of grid data based on spatial coordinates.

        Notes
        -----
        This method constructs a spatial index by traversing the grid hierarchy from the coarsest to the finest
        level, ensuring that each point in the buffer references the most refined grid available at that location.
        Grids that contain children grids will defer to their children for finer spatial subdivisions.
        """

        def update_buffer(grid, start_indices, end_indices) -> None:
            """
            Helper function to update the buffer with the grid's index information.

            Parameters
            ----------
            grid : Grid
                The grid object whose index information will be stored in the buffer.
            start_indices : np.ndarray
                The start indices in the buffer corresponding to the grid's bounding box.
            end_indices : np.ndarray
                The end indices in the buffer corresponding to the grid's bounding box.
            """
            # Create a tuple of slices for all dimensions
            slices = tuple(
                slice(start, end) for start, end in zip(start_indices, end_indices)
            )

            # Update the buffer with the level and grid information
            self.buffer[slices] = (grid.level.index, *grid.index)

        # Step 1: Preload grids and track loading status
        grids_to_process = []
        initial_load_state = {}

        # Record initial state of loading for levels
        for level_id in list(self.grid_manager.Levels.keys()):
            # Access the parent class version of 'get' to check the load state
            level = super(
                self.grid_manager.Levels.__class__, self.grid_manager.Levels
            ).get(level_id, None)
            level_loaded = level is not None
            initial_load_state[level_id] = {
                "level_loaded": level_loaded,
                "grid_loaded": {},
            }

            # Load the level if it wasn't initially loaded
            level = self.grid_manager[level_id]
            for grid_id in list(level.Grids.keys()):
                # Access the parent class version of 'get' to check the load state
                grid = super(level.Grids.__class__, level.Grids).get(grid_id, None)
                grid_loaded = grid is not None
                initial_load_state[level_id]["grid_loaded"][grid_id] = grid_loaded

                # Load the grid if it wasn't initially loaded
                grid = level[grid_id]
                # Check if the grid has children
                if grid.CHILDREN.shape[0] == 0:
                    grids_to_process.append(grid)
                else:
                    for child_id in grid.CHILDREN:
                        child_grid = self.grid_manager[level.index + 1][tuple(child_id)]
                        grids_to_process.append(child_grid)

        # Step 2: Process each grid in the collected list
        for grid in tqdm(
            grids_to_process, desc="Traversing Grid Structure", leave=True
        ):
            grid_bbox = grid.BBOX

            # Convert grid bounding box to resolution indices in buffer
            start_indices = np.floor(
                (grid_bbox[0] - self.BBOX[0]) / (self.BBOX[1] - self.BBOX[0]) * self.RES
            ).astype(int)
            end_indices = np.ceil(
                (grid_bbox[1] - self.BBOX[0]) / (self.BBOX[1] - self.BBOX[0]) * self.RES
            ).astype(int)

            # Ensure the indices are within bounds
            start_indices = np.clip(start_indices, 0, self.RES - 1)
            end_indices = np.clip(end_indices, 0, self.RES - 1)

            # Update buffer with the grid's information
            update_buffer(grid, start_indices, end_indices)

        # Step 3: Unload grids and levels based on initial load state
        for level_id, load_state in initial_load_state.items():
            # Unload grids if they were not initially loaded
            for grid_id, grid_loaded in load_state["grid_loaded"].items():
                if not grid_loaded:
                    self.grid_manager.Levels[level_id].Grids.unload_element(grid_id)

            # Unload level if it was not initially loaded
            if not load_state["level_loaded"]:
                self.grid_manager.Levels.unload_element(level_id)

    def query(self, point: NDArray[float]) -> Tuple[int, Tuple[int, ...]]:
        """
        Query the grid index to find which grid the given point belongs to.

        Parameters
        ----------
        point : np.ndarray
            A 1D array representing the coordinates of the point to be queried.

        Returns
        -------
        tuple[int, tuple[int, ...]]
            A tuple containing the level index and grid index that the point belongs to.

        Raises
        ------
        ValueError
            If the point is outside the bounding box of the grid index.
        """
        if np.any(point < self.BBOX[0]) or np.any(point > self.BBOX[1]):
            raise ValueError(
                f"Point {point} is outside of the bounding box {self.BBOX}."
            )

        # Convert point to resolution index
        resolution_index = np.floor(
            (point - self.BBOX[0]) / (self.BBOX[1] - self.BBOX[0]) * self.RES
        ).astype(int)
        resolution_index = np.clip(
            resolution_index, 0, self.RES - 1
        )  # Ensure the index is within bounds

        # Retrieve level and grid index from the buffer
        level_and_grid = self.buffer[tuple(resolution_index)]
        level_index = level_and_grid[0]
        grid_index = tuple(level_and_grid[1:])

        return level_index, grid_index


class FieldIndex:
    """
    A class representing an individual field in the grid, with attributes such as units, dtype, field name, and HDF5 alias.
    It also supports accessing and setting field data, and allows a LaTeX representation of the field.

    Attributes
    ----------
    hdf5_alias : str
        The HDF5 alias of the field, representing the actual path where the field is stored in the HDF5 structure.
    grid_manager : GridManager
        The grid manager that holds the data for the field.
    units : str
        The units of the field data.
    dtype : np.dtype
        The data type of the field.
    """

    def __init__(
        self,
        field_name: str,
        grid_manager: "GridManager",
        units: str = "",
        dtype: Any = "f8",
        **kwargs,
    ):
        self.name = field_name
        self.hdf5_alias = self.name
        self.grid_manager = grid_manager
        self.units = units
        self.dtype = dtype
        self._params = kwargs

    def __getitem__(self, index: tuple[int, tuple[int, ...]]) -> Any:
        """
        Get data from the field using the given index.

        Parameters
        ----------
        index : tuple[int, tuple[int, ...]]
            The index in the format `(level, (i, j, ...))`, where `level` is the refinement level in the AMR
            hierarchy and `(i, j, ...)` are the grid coordinates at that level. The grid coordinates specify
            the position of the grid within the level.

        Returns
        -------
        np.ndarray
            The data from the field at the specified index.

        Notes
        -----
        The returned data corresponds to the specific field stored at the given grid location. The index
        is a combination of the grid refinement level and its coordinates within that level.
        """
        level, grid_coords = index
        return self.grid_manager[level, grid_coords][self.hdf5_alias]

    def __setitem__(self, index: tuple[int, tuple[int, ...]], value: Any):
        """
        Set data in the field at the specified index.

        Parameters
        ----------
        index : tuple[int, tuple[int, ...]]
            The index in the format `(level, (i, j, ...))`, where `level` is the refinement level in the AMR
            hierarchy and `(i, j, ...)` are the grid coordinates at that level.
        value : Any
            The value to set at the specified index. If the value is a unit-aware array (such as `unyt_array`),
            its units will be validated against the field's units before assignment.

        Notes
        -----
        This method sets the data at the given grid location, ensuring unit compatibility if applicable. The
        data is written directly to the underlying HDF5 dataset.
        """
        level, grid_coords = index
        self.grid_manager[level, grid_coords][self.hdf5_alias] = value

    def __repr__(self):
        return f"<FieldIndex: F={self.name}, HA={self.hdf5_alias}, U={self.units}, T={self.dtype}>"

    def __str__(self):
        """
        Return a user-friendly string representation of the FieldIndex object.

        Returns
        -------
        str
            A string summarizing the field name, HDF5 alias, units, and dtype.
        """
        return f"<FieldIndex: {self.name}, U={self.units}, T={self.dtype}>"


class FieldIndexContainer(ElementContainer[str, FieldIndex]):
    """
    A container for managing FieldIndex objects. This container interacts with the 'FIELDS' group in the HDF5 file,
    loads FieldIndex objects lazily, and supports accessing field data by HDF5 alias or field name.
    """

    def __init__(self, grid_manager):
        """
        Initialize the FieldIndexContainer with the given grid manager.

        Parameters
        ----------
        grid_manager : GridManager
            The manager that handles the grids and fields.
        """
        self.grid_manager = grid_manager
        super().__init__(self.grid_manager._handle["FIELDS"])

    def _index_to_hdf5(self, index: str) -> str:
        """
        Convert a field name (index) to its HDF5 alias. The alias is the uppercased version of the field name.

        Parameters
        ----------
        index : str
            The field name.

        Returns
        -------
        str
            The HDF5 alias corresponding to the field name.
        """
        return index

    def _index_from_hdf5(self, label: str) -> str:
        """
        Convert an HDF5 alias back to the field name. The field name is the lowercased version of the alias.

        Parameters
        ----------
        label : str
            The HDF5 alias.

        Returns
        -------
        str
            The field name corresponding to the alias.
        """
        return label

    def __delitem__(self, index: str):
        # Remove the grid's HDF5 group if it exists
        hdf5_key = self._index_to_hdf5(index)
        if hdf5_key in self._handle:
            del self._handle[hdf5_key]

        # Remove the grid from the in-memory container
        super().__delitem__(index)

    def load_element(self, index: str) -> FieldIndex:
        """
        Load a FieldIndex from the 'FIELDS' group in the HDF5 file.

        Parameters
        ----------
        index : str
            The name of the field to load.

        Returns
        -------
        FieldIndex
            The FieldIndex object corresponding to the field.
        """
        hdf_alias = self._index_to_hdf5(index)
        field_group = self._handle[hdf_alias]

        # Load all attributes into a dictionary
        attributes = {key: value for key, value in field_group.attrs.items()}

        # Initialize and return the FieldIndex object
        return FieldIndex(
            field_name=index,
            grid_manager=self.grid_manager,
            units=attributes.pop("units", ""),
            dtype=attributes.pop("dtype", "f8"),
            **attributes,
        )

    def load_existing_elements(self):
        """
        Load existing fields from the 'FIELDS' group of the HDF5 file and add them as placeholders in the container.
        """
        fields_group = self._handle
        for field_name in fields_group.keys():
            super().__setitem__(self._index_from_hdf5(field_name), None)

    def register_field(self, field_name: str, units: str, dtype: str, **kwargs):
        """
        Add a new FieldIndex to the container and the HDF5 file.

        Parameters
        ----------
        field_name : str
            The name of the field to add. This is the name by which the field will be referenced in the
            grid system.
        units : str
            The units of the field data, which will be stored in the field's attributes. Units ensure that
            operations between fields are consistent, and they provide context for interpreting the field data.
        dtype : str
            The data type of the field, such as `'f8'` for 64-bit floating point numbers. This ensures that
            the data is stored efficiently and can be interpreted correctly during computation.
        **kwargs : dict, optional
            Additional attributes to store with the field, such as descriptions or metadata specific to
            the field's purpose.

        Notes
        -----
        This method adds a new field to the HDF5 structure and registers it with the `FieldIndexContainer`,
        allowing future access and retrieval through the container.
        """

        self.grid_manager.logger.info(f"[ADD ] Registering field {field_name}...")
        hdf5_label = self._index_to_hdf5(field_name)
        field_group = self._handle.create_group(hdf5_label)
        field_group.attrs["units"] = units
        field_group.attrs["dtype"] = dtype

        for k, v in kwargs.items():
            field_group.attrs[k] = v

        # Add the FieldIndex to the container
        self[field_name] = None
