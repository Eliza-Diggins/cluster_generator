"""
GridManager Module
==================

This module provides the `GridManager` class and associated utilities for managing adaptive mesh refinement (AMR)
grids backed by an HDF5 file. The `GridManager` serves as the foundation for creating and managing a hierarchy of
grids with different levels of refinement, allowing for efficient storage, retrieval, and manipulation of grid data.

Mathematical Formalism
----------------------

Let :math:`D` be a rectangular domain in :math:`\mathbb{R}^d`, where :math:`d` is the spatial dimension of the domain.
The adaptive mesh refinement (AMR) structure contains :math:`N` levels, :math:`L_1, L_2, \ldots, L_N`. Each level
:math:`L_i` has a refinement factor :math:`\gamma_i` such that the number of possible grids in level :math:`L_i`
is :math:`\gamma_i^d` times the number of grids in level :math:`L_{i-1}`.

The grid hierarchy is represented as follows:

.. math::

    G_{ijk\ldots}^{\alpha} = \left( i \cdot \gamma_{\alpha} + r_i, j \cdot \gamma_{\alpha} + r_j, \ldots \right)

where :math:`G_{ijk\ldots}^{\alpha}` represents a grid at position :math:`(i, j, k, \ldots)` in level :math:`\alpha`,
and :math:`r_i, r_j, \ldots` are offset values ranging from 0 to :math:`\gamma_{\alpha} - 1`.

Parent and child relationships between grids are defined as:

- **Parent Grid**: For a grid :math:`G_{ijk\ldots}^{\alpha}` on level :math:`\alpha`, the parent grid on level
  :math:`\alpha - 1` is given by:

  .. math::

    P_{ijk\ldots}^{\alpha-1} = \left( \left\lfloor \frac{i}{\gamma_{\alpha}} \right\rfloor,
    \left\lfloor \frac{j}{\gamma_{\alpha}} \right\rfloor, \ldots \right)

- **Children Grids**: For a grid :math:`G_{ijk\ldots}^{\alpha-1}` on level :math:`\alpha - 1`, the children grids on
  level :math:`\alpha` can be found by refining the coordinates:

  .. math::

    C_{ijk\ldots}^{\alpha} = \left( i \cdot \gamma_{\alpha} + r_i, j \cdot \gamma_{\alpha} + r_j, \ldots \right)

  where :math:`r_i, r_j, \ldots` take values from :math:`0` to :math:`\gamma_{\alpha} - 1`.

The `GridManager` class facilitates the creation, access, and manipulation of these hierarchical grid structures,
ensuring efficient handling of large datasets through HDF5 backing.

Notes
-----

- **Performance Considerations**: The `GridManager` class uses caching and optimized metadata handling to ensure
  efficient interactions with the HDF5 file. Developers should be aware of the potential overhead associated with
  frequent grid additions or deletions.
- **Metadata Management**: Metadata is automatically loaded from the HDF5 file or provided through keyword arguments
  during initialization. The metadata descriptors facilitate validation and ensure consistency.
- **Lazy Loading**: Grids and levels are loaded on-demand to minimize memory usage. Use the `load_element()` and
  `unload_element()` methods to control grid memory management explicitly.
- **Error Handling**: Custom exceptions such as `GridManagerMetadataError` are used to handle metadata-related errors.
  Developers should ensure that all required metadata is provided during initialization.

Examples
--------

Every :py:class:`GridManager` effectively represents an HDF5 file on disk. It can either be loaded or created. To
create a :py:class:`GridManager` for a new grid structure, the :py:meth:`GridManager.create` method should be used.

>>> grid_manager = GridManager.create("_gridtest.hdf5",overwrite=True,NDIM=3,BBOX=[[0,0,0],[1,1,1]],BLOCK_SIZE=(512,512,512))
>>> print(grid_manager)
<GridManager (HDF5-backed): Levels=1>

This will initialize a generic grid manager with a single (coarse level).
"""
import inspect
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict

import h5py
import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from cluster_generator.grids._types import (
    BoundingBox,
    GridManagerMetadataDescriptor,
    GridMetadataError,
)
from cluster_generator.grids.grids import LevelContainer
from cluster_generator.grids.indexers import FieldIndexContainer
from cluster_generator.grids.utils import GridManagerLogDescriptor
from cluster_generator.utilities.exceptions import tqdmWarningRedirector
from cluster_generator.utilities.io import HDF5FileHandler

if TYPE_CHECKING:
    import logging

    from cluster_generator.geometry.abc import GeometryHandler
    from cluster_generator.geometry._types import AxisOrder
    from cluster_generator.grids.grids import Field, Grid, GridLevel
    from cluster_generator.profiles._abc import Profile


class GridManager(ABC):
    """
    Abstract base class for adaptive mesh refinement manager systems.

    The `GridManager` class is designed to handle the hierarchical organization of grids stored in an HDF5 file.
    It provides functionality for managing metadata, handling grid levels, and performing file operations. The
    class acts as a foundation for more specific grid management implementations that require adaptive mesh
    refinement (AMR) techniques.

    Parameters
    ----------
    path : str or pathlib.Path
        The file path for HDF5 backing. The file will be created or modified.
    kwargs : dict, optional
        Additional metadata attributes for initialization, such as grid dimensions, bounding box, and units.
        This allows customization of the grid manager during instantiation.

        When reading a :py:class:`GridManager` instance from disk, these should not be necessary unless there
        is missing metadata on disk.

    Attributes
    ----------
    NDIM : int
        Number of dimensions for the grid. Default is 3.

        .. hint::

            ``NDIM`` must match the size of ``BLOCK_SIZE`` and the size of the second axis of ``BBOX``.

    BLOCK_SIZE : NDArray[int]
        The dimensions (number of cells) to place in the coarse grid.

        The lowest refinement level (the coarse level) contains exactly 1 grid with size ``BLOCK_SIZE``, which spans
        the entire domain. Every grid, regardless of level also has size ``BLOCK_SIZE``, but its bounding box may be
        smaller to accomodate refinement.

        .. hint::

            In many cases, the ``BLOCK_SIZE`` determines the balance between memory efficiency and computational
            efficiency. These should be small enough to load into memory, but making them too small will increase the number
            of total grids and slow things down.

    BBOX : NDArray[float]
        Bounding box for the grid. Defines the physical extent of the domain in physical units.

        .. note::

            The units of ``BBOX`` are determined by ``LENGTH_UNIT``.

    LENGTH_UNIT : str
        Unit for length measurements. Default is "kpc".
    TYPE : str
        Type of the `GridManager`. Default is "GENERIC". This attribute can be customized for specialized grid managers.

        .. admonition:: Developer Note

            This should be set as a class attribute for any subclasses of :py:class:`GridManager`.

    COARSE_LEVEL : int, optional
        Coarse level of the grid. Specifies the base level of the grid hierarchy. Default is ``0``.

        Generally, the ``COARSE_LEVEL`` may be zero without issue unless the grid structure is specifically tied
        to the value of each level. In most cases, refinement occurs independent of the starting level label.

    COARSE_CELL_SIZE : NDArray[float]
        Size of the cells at the base level. Calculated based on `BBOX` and `BLOCK_SIZE`.

        .. math::

            \Delta {\bf x}_i = \frac{{\bf L}_i}{{\bf N}_i},

        where :math:`\Delta {\bf x}_i` is the cell size in the :math:`i`'th axis and :math:`{\bf L}` and :math:`{\bf N}` are
        the bounding box size and the block size respectively.

    Levels : LevelContainer
        Container for managing the levels of the grid. Each level corresponds to a refinement level in the AMR hierarchy.
    logger : logging.Logger
        Logger for logging messages related to the `GridManager`.
    _meta : dict
        Internal dictionary for storing metadata attributes loaded from the HDF5 file or provided during initialization.
    _handle : h5py.File
        HDF5 file handle for interacting with the grid data.

    Notes
    -----
    - The `GridManager` class is designed to be extended for specific grid management scenarios, such as handling
      different types of data or incorporating additional metadata requirements. Subclasses should implement their
      own version of metadata descriptors and grid-specific methods.
    - Metadata is loaded either from the provided keyword arguments during instantiation or from the HDF5 file. If
      metadata is not provided in `kwargs` and does not exist in the HDF5 file, a `GridManagerMetadataError` is raised.
    - Grid data is lazily loaded and unloaded to optimize memory usage. This allows efficient management of large
      datasets without loading all data into memory at once.

    See Also
    --------
    GridLevel : Represents a level in the AMR hierarchy, containing a collection of grids.
    LevelContainer : Manages the levels of the grid hierarchy.
    Grid : Represents a single grid in the grid hierarchy.

    Examples
    --------

    .. rubric:: Creating a New GridManager

    This example demonstrates how to create a new `GridManager` instance backed by an HDF5 file. The `GridManager` will initialize with a specified grid structure and metadata.

    .. code-block:: python

        from cluster_generator.grids import GridManager

        # Create a new GridManager with a 3D domain, overwriting any existing file.
        gm = GridManager.create(
            path="my_grids.hdf5",
            overwrite=True,  # Overwrite the file if it already exists.
            NDIM=3,          # Number of dimensions.
            BLOCK_SIZE=[10, 10, 10],  # Dimensions of the grid domain.
            BBOX=[[0, 0, 0], [10, 10, 10]]  # Bounding box in physical units.
        )
        print(gm)
        # Output: <GridManager (HDF5-backed): Levels=0>

        # Add a refinement level to the grid manager.
        gm.add_level(refinement_factor=2)

        # Add a grid to the first level at indices (0, 0, 0).
        gm.add_grid(level_id=0, indices=[0, 0, 0])

        # Print the state of the GridManager
        print(gm)
        # Output: <GridManager (HDF5-backed): Levels=1>

        # Always close the GridManager after use.
        gm.close()

    .. rubric:: Loading an Existing GridManager

    This example shows how to load an existing `GridManager` from an HDF5 file. Once loaded, you can access and manipulate the grid structure and metadata.

    .. code-block:: python

        from cluster_generator.grids import GridManager

        # Load an existing GridManager from the specified HDF5 file.
        gm = GridManager("my_grids.hdf5")

        # Check the number of levels in the GridManager.
        print(len(gm))
        # Output: 1

        # Access a grid from the first level at indices (0, 0, 0).
        grid = gm[0][(0, 0, 0)]
        print(grid)
        # Output: <Grid 0-(0, 0, 0)>

        # Print the bounding box of the grid.
        print(grid.BBOX)
        # Output:
        # [[ 0.  0.  0.]
        #  [ 1.  1.  1.]]

        # Always close the GridManager after use.
        gm.close()

    References
    ----------
    .. [1] Berger, M., & Oliger, J. (1984). Adaptive mesh refinement for hyperbolic partial differential equations.
           Journal of Computational Physics, 53(3), 484-512.
    .. [2] Plewa, T., Linde, T., & Weirs, V. G. (2005). Adaptive Mesh Refinement - Theory and Applications.
           Lecture Notes in Computational Science and Engineering, 41.
    """

    logger: ClassVar["logging.Logger"] = GridManagerLogDescriptor()

    # Metadata descriptors used by the GridManager.
    NDIM: int = GridManagerMetadataDescriptor(required=True, default=3)
    AXES: "AxisOrder" = GridManagerMetadataDescriptor(required=True)
    BLOCK_SIZE: NDArray[int] = GridManagerMetadataDescriptor(required=True)
    BBOX: BoundingBox = GridManagerMetadataDescriptor(required=True)
    LENGTH_UNIT: str = GridManagerMetadataDescriptor(required=True, default="kpc")
    TYPE: str = GridManagerMetadataDescriptor(required=True, default="GENERIC")

    # Coarse level metadata.
    # Different grid managers may wish to configure the COARSE_LEVEL differently depending
    # on their approach. Generally, it's fine to have 0.
    COARSE_LEVEL: int = 0

    # Cache for storing references to metadata descriptors.
    _metadata_cache: ClassVar[Dict[str, "GridManagerMetadataDescriptor"]] = {}

    def __init__(self, path: str | Path, **kwargs):
        """
        Initialize the GridManager, always HDF5-backed.

        Parameters
        ----------
        path : str | Path
            The file path for HDF5 backing. The file will be created or modified.
        kwargs:
            Additional metadata attributes for initialization.
        """
        # Convert the path to a Path object and open the HDF5 file.
        self.path = Path(path)
        self._handle = HDF5FileHandler(str(self.path), mode="r+")
        self.logger.info("[INIT] Initializing GridManager at %s.", path)

        # Create sentinel Levels so that errors can print
        self.Levels = {}

        # Load metadata from the provided kwargs or from the HDF5 file.
        self._meta = {}
        self._initialize_metadata(kwargs)

        # Derive additional metadata if possible.
        self._initialize_additional_metadata()

        # Initialize and load the grid levels.
        self.logger.info("[INIT] \tLoading levels...")
        self.Levels = LevelContainer(self)
        self.logger.debug("[INIT] \tLoaded %s level(s).", len(self.Levels))

        # Initializing the fields container
        try:
            self.Fields = FieldIndexContainer(self)
        except KeyError:
            raise GridMetadataError(
                f"Failed to load GridManager from {self.path} because the FIELDS group is missing or corrupted."
                f" This may be due to unwanted deletion of the group or a bug in the manager's generation."
                f" The FIELDS group could be added manually, or a new GridManager may need to be created."
            )

        self.logger.info("[INIT] Completed initialization of %s.", self)

    def _initialize_metadata(self, kwargs: dict):
        """
        Initialize metadata from kwargs, HDF5 file, or defaults.

        This method loads metadata attributes such as NDIM, BBOX, and BLOCK_SIZE from user-provided inputs,
        HDF5 file, or defaults. It ensures type consistency and self-consistency between these attributes
        before final validation.

        Parameters
        ----------
        kwargs : dict
            Additional metadata attributes for initialization.

        Raises
        ------
        GridMetadataError
            If required metadata is missing or inconsistent with the expected configuration.
        """
        # Step 1: Load metadata from kwargs or defaults, ensuring type consistency.
        bbox = kwargs.get("BBOX")
        block_size = kwargs.get("BLOCK_SIZE")

        # Convert bbox to a consistent format
        if bbox is not None:
            try:
                bbox = np.asarray(bbox)
                if bbox.shape[0] != 2 or bbox.size % 2 != 0:
                    raise ValueError("BBOX should have shape (2, N).")
                bbox = bbox.reshape((2, bbox.size // 2))
                kwargs["BBOX"] = bbox
            except (ValueError, TypeError) as e:
                raise GridMetadataError(f"Invalid BBOX format: {e}")

        # Convert block_size to a consistent format
        if block_size is not None:
            try:
                block_size = np.asarray(block_size, dtype="uint32").flatten()
                kwargs["BLOCK_SIZE"] = block_size
            except (ValueError, TypeError) as e:
                raise GridMetadataError(f"Invalid BLOCK_SIZE format: {e}")

        # Determine NDIM from kwargs, or HDF5, or finally from the default value.
        ndim = kwargs.get("NDIM")
        if ndim is None:
            # Check if NDIM is available in the HDF5 file
            if "NDIM" in self._handle["HEADER"].attrs:
                ndim = self._handle["HEADER"].attrs["NDIM"]
            elif block_size is not None:
                ndim = block_size.size
            elif bbox is not None:
                ndim = bbox.shape[1]
            else:
                # Only use the default NDIM if it cannot be inferred from any source.
                ndim = self.__class__.__dict__["NDIM"].default

        kwargs["NDIM"] = ndim

        # Set the axis order as well
        if "AXES" not in kwargs:
            kwargs["AXES"] = ["x", "y", "z"][: kwargs["NDIM"]]

        # Step 2: Load metadata from HDF5 file if not provided in kwargs.
        for _, meta_class in self._get_metadata_descriptors().items():
            meta_class.set_metadata_from_kwargs_or_hdf5(self, kwargs)

        # Step 3: Re-validate loaded metadata to ensure self-consistency.
        self._validate_metadata_consistency()

    def _validate_metadata_consistency(self):
        """
        Validate metadata consistency between NDIM, BBOX, and BLOCK_SIZE.

        This method checks the consistency of metadata values loaded from the HDF5 file or kwargs
        to ensure that they are mutually compatible.

        Raises
        ------
        GridMetadataError
            If there is any inconsistency between NDIM, BBOX, or BLOCK_SIZE.
        """
        bbox = self.BBOX
        block_size = self.BLOCK_SIZE
        ndim = self.NDIM

        # Validate that BBOX has shape (2, N) where N equals NDIM
        if bbox.shape[0] != 2:
            raise GridMetadataError(
                f"BBOX should have shape (2, N), but got shape {bbox.shape}."
            )

        if bbox.shape[1] != ndim:
            raise GridMetadataError(
                f"BBOX has dimension {bbox.shape[1]}, which is inconsistent with NDIM={ndim}."
            )

        # Validate that BLOCK_SIZE has length N where N equals NDIM
        if block_size.size != ndim:
            raise GridMetadataError(
                f"BLOCK_SIZE has size {block_size.size}, which is inconsistent with NDIM={ndim}."
            )

    def _initialize_additional_metadata(self):
        """
        Initialize additional metadata based on existing metadata.

        Raises
        ------
        GridManagerMetadataError
            If additional metadata cannot be derived from existing metadata.
        """
        try:
            self.COARSE_CELL_SIZE = (self.BBOX[1, :] - self.BBOX[0, :]) / np.asarray(
                self.BLOCK_SIZE
            )
            self.logger.debug(
                "[INIT] \t\tCOARSE_CELL_SIZE = %s.", self.COARSE_CELL_SIZE
            )
        except ValueError as e:
            raise GridMetadataError(
                f"Failed to create COARSE_CELL_SIZE because BBOX and DOMAIN_DIMENSIONS are incompatible: {e}."
            )

    def __str__(self):
        """
        Return a human-readable string representation of the GridManager.

        Returns
        -------
        str
            A string summary of the GridManager, including dimensions and type.
        """
        return f"<GridManager ({self.__class__.TYPE}): Levels={len(self.Levels)}>"

    def __repr__(self):
        """
        Return a detailed string representation of the GridManager for debugging.

        Returns
        -------
        str
            A detailed string that includes path and core metadata like BBOX and DOMAIN_DIMENSIONS.
        """
        return (
            f"<GridManager path={self.path}, NDIM={self.NDIM}, "
            f"DOMAIN_DIMENSIONS={self.BLOCK_SIZE.tolist()}, BBOX={self.BBOX.tolist()}, "
            f"Levels={len(self.Levels)}>"
        )

    def __enter__(self):
        """
        Enter the context manager for the GridManager.
        Ensures that the HDF5 file is open and available for operations.
        """
        if self._handle is None:
            self._handle = h5py.File(self.path, "r+")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context manager, ensuring that the HDF5 file is closed properly.
        """
        self.close()

    def __del__(self):
        """
        Ensure the HDF5 file is closed when the GridManager is deleted.

        This method is called when the `GridManager` instance is about to be destroyed. It ensures that
        any open resources, such as the HDF5 file handle, are properly closed.

        Notes
        -----
        It is good practice to explicitly call the `close()` method rather than relying on `__del__`,
        to ensure resources are released in a timely manner.

        """
        self.close()

    def __len__(self):
        """Return the number of levels managed by the GridManager."""
        return len(self.Levels)

    def __getitem__(
        self, index: int | tuple[int, tuple[int, ...]]
    ) -> "GridLevel | Grid":
        """
        Access a specific level or grid in the `LevelsCollection`.

        Parameters
        ----------
        index : int or tuple[int, tuple[int, ...]]
            If an integer is provided, it represents the ID of the level to be accessed.
            If a tuple is provided, it should be in the format `(level_id, grid_indices)`,
            where `level_id` is the ID of the level and `grid_indices` is the tuple of indices
            representing the position of the grid within the level.

        Returns
        -------
        GridLevel or Grid
            If `index` is an integer, the corresponding `GridLevel` is returned.
            If `index` is a tuple, the corresponding `Grid` is returned.

        Raises
        ------
        KeyError
            If the specified level or grid does not exist.

        Examples
        --------
        .. code-block:: python

            gm = GridManager("my_grids.hdf5")

            # Access level 0
            level = gm[0]
            print(level)
            # Output: <GridLevel 0>

            # Access grid at level 0 with indices (2, 0, 2)
            grid = gm[(0, (2, 0, 2))]
            print(grid)
            # Output: <Grid 0-(2, 0, 2)>
        """
        if isinstance(index, int):
            try:
                return self.Levels[index]
            except KeyError:
                raise KeyError(f"{self} has no level {index}.")

        elif isinstance(index, tuple):
            try:
                _level = self.Levels[index[0]]
            except KeyError:
                raise KeyError(f"{self} has no level {index[0]}.")

            try:
                return _level[index[1]]
            except KeyError:
                raise KeyError(f"{self} level {_level} has no grid {index[1]}.")

    def __setitem__(
        self, index: int | tuple[int, tuple[int, ...]], value: "GridLevel | Grid"
    ):
        """
        Add or update a level or grid in the `LevelsCollection`.

        Parameters
        ----------
        index : int or tuple[int, tuple[int, ...]]
            If an integer is provided, it represents the ID of the level to be added or updated.
            If a tuple is provided, it should be in the format `(level_id, grid_indices)`,
            where `level_id` is the ID of the level and `grid_indices` is the tuple of indices
            representing the position of the grid within the level.
        value : GridLevel or Grid
            The `GridLevel` or `Grid` object to be added or updated.

        Raises
        ------
        TypeError
            If the `value` is not an instance of `GridLevel` or `Grid`.
        ValueError
            If the level specified in the index does not exist.

        Examples
        --------
        .. code-block:: python

            gm = GridManager("my_grids.hdf5")

            # Add or update a level
            new_level = GridLevel(...)  # Assume GridLevel is properly instantiated
            gm[0] = new_level

            # Add or update a grid within a level
            new_grid = Grid(...)  # Assume Grid is properly instantiated
            gm[(0, (2, 0, 2))] = new_grid
        """
        if isinstance(index, int):
            self.Levels[index] = value

        elif isinstance(index, tuple):
            _level = self.Levels[index[0]]
            _level[index[1]] = value

    def __delitem__(self, index: int | tuple[int, tuple[int, ...]]):
        """
        Remove a level or grid from the `LevelsCollection`.

        Parameters
        ----------
        index : int or tuple[int, tuple[int, ...]]
            If an integer is provided, it represents the ID of the level to be removed.
            If a tuple is provided, it should be in the format `(level_id, grid_indices)`,
            where `level_id` is the ID of the level and `grid_indices` is the tuple of indices
            representing the position of the grid within the level.

        Raises
        ------
        KeyError
            If the specified level or grid does not exist.

        Examples
        --------
        .. code-block:: python

            gm = GridManager("my_grids.hdf5")

            # Remove a level from the collection
            del gm[0]

            # Remove a grid from a specific level
            del gm[(0, (2, 0, 2))]
        """
        if isinstance(index, int):
            del self.Levels[index]

        elif isinstance(index, tuple):
            _level = self.Levels[index[0]]
            del _level[index[1]]

    def __contains__(self, index: int | tuple[int, tuple[int, ...]]) -> bool:
        """
        Check if a level or grid exists in the `LevelsCollection`.

        Parameters
        ----------
        index : int or tuple[int, tuple[int, ...]]
            If an integer is provided, it represents the ID of the level to check for existence.
            If a tuple is provided, it should be in the format `(level_id, grid_indices)`,
            where `level_id` is the ID of the level and `grid_indices` is the tuple of indices
            representing the position of the grid within the level.

        Returns
        -------
        bool
            `True` if the level or grid exists, otherwise `False`.

        Examples
        --------
        .. code-block:: python

            gm = GridManager("my_grids.hdf5")

            # Check if a level exists
            print(0 in gm)
            # Output: True  # If level 0 exists

            # Check if a grid exists in a level
            print((0, (2, 0, 2)) in gm)
            # Output: True  # If grid at indices (2, 0, 2) exists in level 0
        """
        if isinstance(index, int):
            return index in self.Levels

        elif isinstance(index, tuple):
            return (index[0] in self.Levels) and (index[1] in self.Levels[index[0]])

    @classmethod
    def _get_metadata_descriptors(cls) -> Dict[str, "GridManagerMetadataDescriptor"]:
        """
        Retrieve all metadata descriptors, including those in parent classes.

        Returns
        -------
        dict
            A dictionary of metadata descriptors with the attribute names as keys and the descriptor instances as values.
        """
        if not cls._metadata_cache:
            descriptors = {}
            for base_class in inspect.getmro(cls):
                for name, attr in vars(base_class).items():
                    if isinstance(attr, GridManagerMetadataDescriptor):
                        descriptors[name] = attr
            cls._metadata_cache = descriptors
        return cls._metadata_cache

    @classmethod
    def create(
        cls, path: str | Path, overwrite: bool = False, **kwargs
    ) -> "GridManager":
        """
        Create a new GridManager, always HDF5-backed.

        This method creates a new HDF5 file or overwrites an existing one. Metadata and grid structure
        are initialized from the provided keyword arguments.

        Parameters
        ----------
        path : str or pathlib.Path
            The file path for HDF5 backing. The file will be created or overwritten.
        overwrite : bool, optional
            Whether to overwrite the file if it already exists. Default is False.
        kwargs : optional
            Additional metadata attributes for initialization, such as:
            - `NDIM`: Number of dimensions.
            - `BLOCK_SIZE`: Dimensions of the grid.
            - `BBOX`: Bounding box for the grid domain.

        Returns
        -------
        GridManager
            A new instance of the GridManager.

        Raises
        ------
        FileExistsError
            If the file already exists and `overwrite=False`.

        Examples
        --------

        .. code-block:: python

            gm = GridManager.create("my_grids.hdf5", overwrite=True, NDIM=3, BLOCK_SIZE=[10, 10, 10], BBOX=[[0, 0, 0], [10, 10, 10]])

        """
        # Ensure the file does not already exist or overwrite is allowed.
        path = Path(path)
        if path.exists() and overwrite:
            path.unlink()
            cls.logger.info("[NEW ] Deleted existing %s.", path)
        elif path.exists():
            raise FileExistsError(
                f"File {path} already exists. Use overwrite=True to allow overwrite."
            )

        # Create the HDF5 file and initialize the GridManager.
        with HDF5FileHandler(str(path), mode="w") as handler:
            cls.logger.info("[NEW ] Created %s.", path)
            if "HEADER" not in handler:
                handler.create_group("HEADER")
            if "FIELDS" not in handler:
                handler.create_group("FIELDS")

        return cls(path=path, **kwargs)

    def close(self):
        """
        Close the HDF5 file and ensure resources are released.
        """
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None
            self.logger.debug("[END ] HDF5 file closed.")

    def commit_changes(self):
        """
        Commit changes to the HDF5 file.

        This method flushes any pending changes to the HDF5 file, ensuring data consistency.
        """
        if self._handle is not None:
            self._handle.flush()
            self.logger.debug("[FLSH] Changes committed to HDF5 file.")

    def unload_metadata(self, key: str):
        """
        Unload specific metadata from memory to free up resources.

        Parameters
        ----------
        key : str
            The metadata key to unload.

        Examples
        --------
        >>> gm = GridManager("my_grids.hdf5")
        >>> gm.unload_metadata("BBOX")
        >>> "BBOX" not in gm._meta
        True  # Metadata for BBOX is unloaded
        """
        if key in self._meta:
            self.logger.debug(f"[META] Unloading metadata: {key}")
            self._meta.pop(key, None)

    def reload_metadata(self, key: str):
        """
        Reload specific metadata from the HDF5 file if applicable.

        Parameters
        ----------
        key : str
            The metadata key to reload.

        Examples
        --------
        >>> gm = GridManager("my_grids.hdf5")
        >>> gm.reload_metadata("BBOX")
        >>> gm._meta["BBOX"]  # Metadata for BBOX is reloaded
        array([[ 0.,  0.,  0.],
               [10., 10., 10.]])
        """
        self.logger.debug(f"[META] Reloading metadata: {key}")
        self._meta[key] = self._handle["HEADER"].attrs.get(key)

    def add_level(self, refinement_factor: int) -> "GridLevel":
        """
        Add a new grid level to the LevelsCollection.

        This method adds a new level to the grid hierarchy with the specified refinement factor.
        The refinement factor determines how much finer the grid resolution is compared to the previous level.

        Parameters
        ----------
        refinement_factor : int
            The refinement factor for the new level. For example, a factor of 2 means that each grid
            in the new level has twice the resolution of the grids in the previous level.

        Returns
        -------
        GridLevel
            The newly added GridLevel instance.

        Examples
        --------

        .. code-block:: python

            gm.add_level(refinement_factor=2)

        """
        return self.Levels.add_level(refinement_factor)

    def add_grid(self, level_id: int, indices: NDArray[int]) -> "Grid":
        """
        Add a new grid to a specified level.

        This method adds a new grid at the specified level and grid indices. The grid is created
        at the provided indices within the level's domain.

        Parameters
        ----------
        level_id : int
            The ID of the level to which the grid will be added.
        indices : NDArray[int]
            The position index of the new grid in the level. This is a tuple of integers representing
            the grid's location in the level's grid structure.

        Returns
        -------
        Grid
            The newly added Grid instance.

        Raises
        ------
        ValueError
            If the specified level does not exist.

        Examples
        --------

        .. code-block:: python

            gm.add_grid(level_id=0, indices=[0, 0, 0])

        """
        if level_id not in self.Levels:
            raise ValueError(f"Level {level_id} does not exist in the GridManager.")
        level = self.Levels[level_id]
        return level.add_grid(indices)

    def add_field(
        self, level_id: int, grid_id: int, name: str, dtype: Any, units: str = ""
    ) -> "Field":
        """
        Add a new field to a specified grid within a level.

        Fields represent physical quantities (e.g., density, temperature) that are stored in the grid.
        This method adds a new field to the grid.

        Parameters
        ----------
        level_id : int
            The ID of the level containing the grid.
        grid_id : int
            The ID of the grid to which the field will be added.
        name : str
            The name of the new field (e.g., "density").
        dtype : Any
            The data type of the field (e.g., `numpy.float32`).
        units : str, optional
            The units of the field (e.g., "g/cm^3"). Default is an empty string.

        Returns
        -------
        Field
            The newly added Field instance.

        Raises
        ------
        ValueError
            If the specified level or grid does not exist.

        Examples
        --------

        .. code-block:: python

            gm.add_field(level_id=0, grid_id=0, name="density", dtype=np.float32, units="g/cm^3")

        """
        if level_id not in self.Levels:
            raise ValueError(f"Level {level_id} does not exist in the GridManager.")
        level = self.Levels[level_id]
        if grid_id not in level.Grids:
            raise ValueError(f"Grid {grid_id} does not exist in Level {level_id}.")
        grid = level.Grids[grid_id]
        return grid.Fields.add_field(name=name, dtype=dtype, units=units)

    def remove_level(self, level_id: int):
        """
        Remove a specified level and all associated grids from the GridManager.

        This method deletes the level and all grids within it from both the in-memory structure
        and the HDF5 file.

        Parameters
        ----------
        level_id : int
            The ID of the level to be removed.

        Raises
        ------
        ValueError
            If the specified level does not exist.
        """
        del self.Levels[level_id]

    def remove_grid(self, level_id: int, grid_id: tuple[int, ...]):
        """
        Remove a specified grid from a level in the GridManager.

        This method deletes the grid from both the in-memory structure and the HDF5 file.

        Parameters
        ----------
        level_id : int
            The ID of the level containing the grid.
        grid_id : tuple[int, ...]
            The index of the grid to be removed.

        Raises
        ------
        ValueError
            If the specified level or grid does not exist.
        """
        level = self.Levels[level_id]
        del level.Grids[grid_id]

    def remove_field(self, level_id: int, grid_id: tuple[int, ...], field_name: str):
        """
        Remove a specified field from a grid in the GridManager.

        This method deletes the field from both the in-memory structure and the HDF5 file.

        Parameters
        ----------
        level_id : int
            The ID of the level containing the grid.
        grid_id : tuple[int, ...]
            The index of the grid containing the field.
        field_name : str
            The name of the field to be removed.

        Raises
        ------
        ValueError
            If the specified level, grid, or field does not exist.
        """
        level = self.Levels[level_id]
        grid = level.Grids[grid_id]
        del grid.Fields[field_name]

    def get_level(self, level_id: int) -> "GridLevel":
        """
        Retrieve a specific level by its ID.

        Parameters
        ----------
        level_id : int
            The ID of the level to retrieve.

        Returns
        -------
        GridLevel
            The level corresponding to the specified ID.

        Raises
        ------
        KeyError
            If the level does not exist.
        """
        return self[level_id]  # Uses __getitem__ with an integer to get a level.

    def get_grid(self, level_id: int, grid_indices: tuple[int, ...]) -> "Grid":
        """
        Retrieve a specific grid by its level ID and grid indices.

        Parameters
        ----------
        level_id : int
            The ID of the level containing the grid.
        grid_indices : tuple[int, ...]
            The grid indices within the level.

        Returns
        -------
        Grid
            The grid corresponding to the specified level ID and grid indices.

        Raises
        ------
        KeyError
            If the level or grid does not exist.
        """
        return self[
            (level_id, grid_indices)
        ]  # Uses __getitem__ with a tuple to get a grid.

    def get_field(
        self, level_id: int, grid_indices: tuple[int, ...], field_name: str
    ) -> "Field":
        """
        Retrieve a specific field from a grid.

        Parameters
        ----------
        level_id : int
            The ID of the level containing the grid.
        grid_indices : tuple[int, ...]
            The grid indices within the level.
        field_name : str
            The name of the field to retrieve.

        Returns
        -------
        Field
            The field corresponding to the specified level ID, grid indices, and field name.

        Raises
        ------
        KeyError
            If the level, grid, or field does not exist.
        """
        grid = self[(level_id, grid_indices)]  # Get the grid using __getitem__
        return grid.Fields[field_name]  # Access the field from the grid.

    def add_universal_field(
        self,
        field_name: str,
        units: str = "",
        dtype: np.dtype = "f8",
        overwrite: bool = False,
        **kwargs,
    ):
        """
        Add a field universally to all grids in the manager.

        This method adds a new field to every grid across all levels in the `GridManager`. If the field already
        exists and `overwrite=True`, the existing field is replaced.

        Parameters
        ----------
        field_name : str
            The name of the field to add universally to all grids.
        units : str, optional
            The units for the field. Default is an empty string.
        dtype : np.dtype, optional
            The data type for the field. Default is 'f8'.
        overwrite : bool, optional
            If True, overwrites existing fields with the same name. Default is False.
        register : bool, optional
            If True, registers the field in the `Fields` container after creation. Default is True.
        field_meta : dict, optional
            Metadata to associate with the field during registration. Default is an empty dictionary.
        kwargs : optional
            Additional arguments for field creation. This can be used to provide additional metadata or parameters
            for field creation.

        Notes
        -----
        - The method iterates over all levels and grids, adding the specified field to each grid.
        - The method checks if the field already exists in the `Fields` container. If `overwrite=False` and the field
          exists, a `ValueError` is raised.
        - The `register` parameter controls whether the field should be registered in the `Fields` container after
          creation. If set to `False`, the field will be created but not registered.
        - The `field_meta` parameter allows you to provide additional metadata to the field during registration.

        Examples
        --------
        .. code-block:: python

            gm.add_universal_field(field_name="temperature", units="K", dtype=np.float32, overwrite=True)

        This will add the "temperature" field to all grids across all levels with the specified units and data type.
        """
        # Managing overwrite logic. If we have the field in our Fields, then we
        # universally remove the field and then proceed.
        self.logger.info(
            f"[ADD ] Adding universal field {field_name}... OVERWRITE={overwrite}."
        )
        if overwrite and (field_name in self.Fields):
            self.remove_universal_field(field_name, unregister=True)
        elif (not overwrite) and (field_name in self.Fields):
            raise ValueError(
                f"Cannot add field {field_name} universally because it already exists and overwrite=False."
            )
        else:
            pass

        # Iterate through each level and each grid adding the field as we go.
        register_flag, field_meta = kwargs.pop("register", True), kwargs.pop(
            "field_meta", {}
        )
        with logging_redirect_tqdm(loggers=[self.logger]), tqdmWarningRedirector():
            for level in tqdm(
                self.Levels.values(),
                desc=f"[ADD ] Building field {field_name}",
                position=0,
                leave=False,
            ):
                for grid in tqdm(
                    level.Grids.values(),
                    desc=f"[ADD, LvL={level.index}] Building field {field_name}",
                    position=1,
                    leave=False,
                ):
                    # Add the field if it does not already exist
                    if field_name not in grid.Fields:
                        grid.add_field(
                            name=field_name,
                            units=str(units),
                            dtype=dtype,
                            register=False,
                            **kwargs,
                        )

        # We now want to register the field.
        if register_flag:
            self.Fields.register_field(
                field_name, units=units, dtype=dtype, **field_meta
            )

    def remove_universal_field(self, field_name: str, **kwargs):
        """
        Remove a field universally from all grids in the manager.

        This method removes the specified field from every grid in the `GridManager`. If the field does not exist
        in a particular grid, it is skipped.

        Parameters
        ----------
        field_name : str
            The name of the field to remove universally from all grids.
        kwargs : optional
            Additional keyword arguments to modify behavior. If `unregister=True` is passed, the field is also removed
            from the `Fields` container.

        Notes
        -----
        - The method iterates over all levels and grids, removing the specified field from each grid.
        - If the field does not exist in a grid, it is simply skipped without raising an error.
        - If `unregister=True` is passed, the field is also removed from the global `Fields` container.

        Examples
        --------
        .. code-block:: python

            gm.remove_universal_field("density", unregister=True)

        This will remove the "density" field from all grids across all levels and unregister it from the `Fields` container.
        """
        self.logger.debug(f"[DEL ] Removing universal field {field_name}...")
        with logging_redirect_tqdm(loggers=[self.logger]), tqdmWarningRedirector():
            for level in tqdm(
                self.Levels.values(),
                desc=f"[DEL ] Removing field {field_name}",
                position=0,
                leave=False,
            ):
                for grid in tqdm(
                    level.Grids.values(),
                    desc=f"[DEL, LvL={level.index}] Removing field {field_name}",
                    position=1,
                    leave=False,
                ):
                    # Add the field if it does not already exist
                    if field_name in grid.Fields:
                        del grid.Fields[field_name]

        # Remove the universal field from the FieldIndexContainer
        if kwargs.pop("unregister", True):
            del self.Fields[field_name]

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
        Add a field to all grids across all levels, using a function to generate field values.

        This method creates a field across all grids in the `GridManager` and populates it by applying the given function.
        Optionally, if a geometry handler is provided, grid coordinates can be transformed before applying the function.

        Parameters
        ----------
        function : Callable[[np.ndarray], np.ndarray]
            A function that takes grid coordinates as input and returns the field data for that grid. The function should
            return an array of the same shape as the input coordinates.
        field_name : str
            The name of the field to add.
        dtype : np.dtype, optional
            The data type of the field. Default is 'f8' (double precision floating point).
        units : str, optional
            The units for the field. Default is an empty string.
        geometry : GeometryHandler, optional
            A geometry handler that can transform the Cartesian coordinates of the grid into another coordinate system.
            If provided, the handler is applied to the grid coordinates before the function is applied.
        overwrite : bool, optional
            If True, overwrites any existing fields with the same name. Default is False.
        kwargs : dict, optional
            Additional keyword arguments to pass to the function when computing field values.

        Notes
        -----
        - The field is created and initialized across all grids in two phases:
          1. **Field Creation**: The field is created in every grid.
          2. **Field Population**: The field is populated with values returned by the function. If a geometry handler
             is provided, the coordinates are transformed before applying the function.
        - The function must return field values corresponding to the grid's coordinates. If the `geometry` parameter is
          provided, the grid coordinates are transformed according to the geometry before the function is applied.
        - The method also commits changes to the HDF5 file after processing all levels.

        Examples
        --------
        .. code-block:: python

            def density_profile(coords):
                # Example density function based on spherical distance from origin
                r = np.sqrt(np.sum(coords ** 2, axis=-1))
                return 1 / (r + 1)

            gm.add_field_from_function(density_profile, "density", units="g/cm^3")

        This example adds a "density" field to all grids, where the field values are computed by the `density_profile`
        function. The grid coordinates are used to compute a spherical density distribution.
        """
        # Prepare the final function with optional coordinate transformation.
        # If a geometry is provided, we need to automatically convert the GridManager coordinates
        # into the desired coordinate system, evaluate the function, and then dump the values.
        final_function = (
            function
            if geometry is None
            else lambda grid_coords: function(
                *geometry.build_converter(self.AXES)(grid_coords)
            )
        )

        # PHASE 1: Create the field across all grids
        self.add_universal_field(
            field_name, units=units, dtype=dtype, overwrite=overwrite, **kwargs
        )

        # PHASE 2: Populate the field data in each grid
        with logging_redirect_tqdm(loggers=[self.logger]), tqdmWarningRedirector():
            for level in tqdm(
                self.Levels.values(),
                desc=f"[DUMP] Constructing {field_name} from func.:",
                position=0,
                leave=False,
            ):
                cell_size = level.CELL_SIZE
                for grid in tqdm(
                    level.Grids.values(),
                    desc=f"[DUMP, LvL={level.index}] Constructing {field_name} from func.:",
                    position=1,
                    leave=False,
                ):
                    coords = grid.opt_get_coordinates(
                        grid.BBOX, self.BLOCK_SIZE, cell_size
                    )
                    field_data = final_function(coords)
                    grid.Fields[field_name][:] = field_data

            # Commit changes after processing each level
            self.commit_changes()

    def add_field_from_profile(
        self,
        profile: "Profile",
        field_name: str,
        dtype: np.dtype = "f8",
        units: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Add a field to all grids across all levels using a `Profile` object.

        This method leverages a `Profile` object to generate field values for all grids in the `GridManager`. The
        `Profile` provides a function that generates field data, as well as optional geometry handling and metadata
        such as data type and units.

        Parameters
        ----------
        profile : Profile
            A `Profile` object that defines a callable function to generate the field data. The `Profile` may also
            contain a `geometry_handler` to transform the grid coordinates.
        field_name : str
            The name of the field to add.
        dtype : np.dtype, optional
            The data type for the field. Default is 'f8' (double precision floating point).
        units : str, optional
            The units for the field. Default is an empty string.
        kwargs : dict, optional
            Additional keyword arguments to pass to the `Profile` function during field computation.

        Notes
        -----
        - This method internally calls `add_field_from_function()`, passing the `Profile`'s function and geometry handler.
        - The `Profile` object provides a callable function for generating field data and may optionally provide a
          `geometry_handler` to transform the grid coordinates before applying the function.
        - The method creates the field across all grids in the manager and populates it with values returned by the
          `Profile`'s function.

        Examples
        --------
        .. code-block:: python

            # Assuming we have a Profile object named 'density_profile'
            gm.add_field_from_profile(density_profile, "density", dtype=np.float32, units="g/cm^3")

        This will add the "density" field to all grids in the `GridManager`, using the function defined in the
        `density_profile` object to populate the field values.
        """
        # Use the profile's function, data type, and units
        self.add_field_from_function(
            function=profile,
            field_name=field_name,
            dtype=dtype,
            units=units,
            geometry=profile.geometry_handler,
            **kwargs,
        )


if __name__ == "__main__":
    GridManager.logger.setLevel("INFO")
    q = GridManager.create(
        "test.hdf5", overwrite=True, BBOX=[[0, 0], [1, 1]], BLOCK_SIZE=[10, 10]
    )
    q[0, (0, 0)].refine(10)
