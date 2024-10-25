"""
Base Geometry Module for Cluster Generator
==========================================

This module defines the abstract base class for various geometries used in cluster profiles.
Geometries represent different coordinate systems and transformations between Cartesian
coordinates and their respective systems (e.g., spherical, cylindrical). They also provide
methods for integrating density profiles, computing gradients, and other geometry-specific
operations optimized for the symmetry of the system.

Classes
-------
- :py:class:`BaseGeometry`
    Abstract base class for different geometries used in cluster profiles.

Notes
-----
To extend this module and create a new geometry class, follow these steps:

1. **Subclassing**: Create a new class that inherits from `BaseGeometry`.
2. **Define Expected Parameters**: Override the `_expected_parameters` attribute.
3. **Implement Abstract Methods**: Implement the `from_cartesian`, `to_cartesian`,
   `integrate_profile`, and `compute_gradient` methods.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Type, Union

import h5py
import numpy as np

from cluster_generator.geometry._types import (
    AxisName,
    AxisOrder,
    GenericVolumeElement,
    LameCFunctions,
    ProfileInput,
    ProfileResult,
    SymmetryType,
)


class InvalidGeometryParameterError(ValueError):
    """Exception raised for invalid geometry parameters."""


class GeometryHandler(ABC):
    """
    Abstract base class for handling different geometries used in cluster profiles.

    Attributes
    ----------
    NAME : :py:class:`AxisName`
        Name of the geometry type (e.g., 'Cartesian', 'Spherical'). Defaults to 'Generic'.

    AXES : :py:class:`AxisOrder`
        A tuple representing the default axes for the geometry (e.g., ('x', 'y', 'z')).

    SYMMETRY_AXES : :py:class:`AxisOrder`, optional
        A tuple specifying the symmetry axes if the geometry has symmetry (e.g., cylindrical or spherical symmetry).

    SYMMETRY_TYPE : :py:class:`SymmetryType`, optional
        Defines the type of symmetry (e.g., rotational or reflectional symmetry).

    FREE_AXES : :py:class:`AxisOrder`, optional
        Specifies the axes that are free to vary, not constrained by symmetry.

    _expected_parameters : dict
        A dictionary that holds the default values for the parameters specific to the geometry.

    Examples
    --------
    To subclass and create a specific geometry (e.g., SphericalGeometry):

    .. code-block:: python

        class SphericalGeometry(GeometryHandler):
            NAME = "Spherical"
            AXES = ('r', 'theta', 'phi')
            SYMMETRY_AXES = ('theta', 'phi')
            SYMMETRY_TYPE = "Rotational"

            _expected_parameters = {
                "radius": 1.0
            }

            def from_cartesian(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
                r = np.sqrt(x**2 + y**2 + z**2)
                theta = np.arccos(z / r)
                phi = np.arctan2(y, x)
                return r, theta, phi

            def to_cartesian(self, r: np.ndarray, theta: np.ndarray, phi: np.ndarray):
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
                return x, y, z

            def _get_lame_functions(self):
                # Implementation for spherical Lame coefficients goes here
                pass

            def _get_volume_element_function(self):
                # Implementation for spherical volume element goes here
                pass

    Raises
    ------
    InvalidGeometryParameterError
        Raised when an unexpected geometry parameter is provided.

    Notes
    -----

    The :py:class:`GeometryHandler` class serves a few key purposes:

    1. **Specify a coordinate system**: The geometry handler instructs all of the other classes in ``cluster_generator``
       about what geometry to use. This means that it informs all of the computations done in model generation, etc.
    2. **Provide operations support**: Each coordinate system provides support for various operations of importance
       which are geometry specific.

    """

    # Class attributes. These are used to determine general properties
    # of the geometry being specified.
    NAME: AxisName = "Generic"
    AXES: AxisOrder = ("x", "y", "z")
    SYMMETRY_AXES: AxisOrder = None
    SYMMETRY_TYPE: SymmetryType = None
    FREE_AXES: AxisOrder = None

    # Specifiers for determining the coordinate system correctly.
    # Implementations should place any geometry specifying parameters here with values
    # equal to the default value for that parameter.
    _expected_parameters: Dict[str, Any] = {}

    def __init__(self, **kwargs):
        """
        Initialize the geometry with specified parameters.

        Parameters
        ----------
        **kwargs : dict
            Parameters required to define the geometry. These must match the expected
            specifiers for the geometry type.

        Raises
        ------
        InvalidGeometryParameterError
            If any unexpected parameter is provided that is not in the geometry's
            specifiers.
        """
        # Initialize the geometry parameters with defaults
        self._parameters: Dict[str, Any] = self.__class__._expected_parameters.copy()

        # Validate and update parameters based on provided kwargs
        for key, value in kwargs.items():
            if key in self._parameters:
                self._parameters[key] = value
            else:
                raise InvalidGeometryParameterError(
                    f"Unexpected geometry parameter: '{key}' for {self.__class__.NAME}."
                )

        # Initialize geometry-specific attributes
        self.lame_functions: LameCFunctions = self._get_lame_functions()
        self.volume_element_function: GenericVolumeElement = (
            self._get_volume_element_function()
        )

    def __repr__(self) -> str:
        """Return a string representation of the geometry object."""
        params_str = ", ".join([f"{k}={v}" for k, v in self._parameters.items()])
        return f"<{self.__class__.__name__}({params_str})>"

    def __str__(self) -> str:
        """Return a human-readable string representation of the geometry object."""
        return f"{self.NAME} Geometry with parameters: {self._parameters}"

    def __eq__(self, other: Any) -> bool:
        """Check equality between two geometry objects."""
        if not isinstance(other, GeometryHandler):
            return False
        return self.NAME == other.NAME and all(
            self.parameters[k] == other.parameters.get(k, None) for k in self.parameters
        )

    def __hash__(self) -> int:
        """
        Generate a hash value for the geometry object.

        The hash is computed based on a representative dictionary that includes the class name
        and the parameters defining the geometry.

        Returns
        -------
        int
            The hash value representing the geometry object.
        """
        # Create a representative dictionary with the class name and parameters
        representative_dict = {
            "class_name": self.__class__.__name__,
            "parameters": tuple(
                sorted(self._parameters.items())
            ),  # Sort items to ensure consistent hashing
        }
        # Use the frozenset of the dictionary items to generate a hash
        return hash(frozenset(representative_dict.items()))

    @abstractmethod
    def from_cartesian(
        self, x: ProfileInput, y: ProfileInput, z: ProfileInput
    ) -> Tuple[ProfileResult]:
        """Convert Cartesian coordinates to the specific coordinate system."""
        pass

    @abstractmethod
    def to_cartesian(self, *args: ProfileInput) -> Tuple[ProfileResult]:
        """Convert coordinates from the specific coordinate system to Cartesian coordinates."""
        pass

    @abstractmethod
    def _get_lame_functions(self) -> LameCFunctions:
        """
        Fetches the Lame coefficient functions for this geometry.
        """
        pass

    @abstractmethod
    def _get_volume_element_function(self) -> GenericVolumeElement:
        """Compute the volume element for this geometry."""
        pass

    @classmethod
    def from_hdf5(
        cls, hdf5_file: Union[str, h5py.File], group_path: str
    ) -> "GeometryHandler":
        """
        Load a geometry object from the specified HDF5 file and group path.

        Parameters
        ----------
        hdf5_file : str or h5py.File
            The path to the HDF5 file or an open h5py.File object from which the geometry will be loaded.
        group_path : str
            The path within the HDF5 file where the geometry is stored.

        Returns
        -------
        GeometryHandler
            An instance of the geometry object initialized from the HDF5 file.

        Raises
        ------
        ValueError
            If the class specified in the HDF5 file cannot be found.
        """
        # Managing str vs. h5py.HDF5File type issues.
        oc_flag = isinstance(hdf5_file, str)
        if oc_flag:
            hdf5_file = h5py.File(hdf5_file)

        try:
            # Now load the geometry from HDF5.
            if group_path not in hdf5_file:
                raise IOError(
                    f"Failed to load geometry. Group {group_path} does not exist in {hdf5_file}."
                )

            group = hdf5_file[group_path]

            # Store the class name as an attribute and load the true class.
            try:
                class_name = group.attrs["class_name"]
            except KeyError:
                raise IOError(
                    f"Failed to load geometry. Group {group_path} is missing required attribute class_name."
                    f" This likely indicates that either this is not the correct geometry group or"
                    f" that something went wrong when writing the geometry to HDF5 originally."
                )

            _cls = cls._get_class_by_name(class_name)

            # create the parameters.
            parameters = {k: v for k, v in group.attrs.items() if k != "class_name"}
            return _cls(**parameters)

        finally:
            if oc_flag:
                hdf5_file.close()

    def to_hdf5(
        self, hdf5_file: Union[str, h5py.File], group_path: str, mode="a"
    ) -> None:
        """
        Save the geometry object to the specified HDF5 file and group path.

        Parameters
        ----------
        hdf5_file : str or h5py.File
            The path to the HDF5 file or an open h5py.File object where the geometry will be saved.
        group_path : str
            The path within the HDF5 file where the geometry will be stored.

        Raises
        ------
        TypeError
            If the hdf5_file is neither a string nor an h5py.File object.
        """
        # Type check the hdf5_file and figure out if we need to open / close or not.
        oc_flag = isinstance(hdf5_file, str)
        if oc_flag:
            # We have a string so we need to open the file and mark it for closure afterwards.
            hdf5_file = h5py.File(hdf5_file, mode=mode)

        try:
            # Create the group path if it doesn't exist
            group = hdf5_file.require_group(group_path)

            # Store the class name as an attribute
            group.attrs["class_name"] = self.__class__.__name__

            # Save parameters as attributes or datasets
            for key, value in self._parameters.items():
                group.attrs[key] = value
        finally:
            if oc_flag:
                hdf5_file.close()

    def build_converter(self, grid_axis_order: AxisOrder):
        """
        Build a converter function to map Cartesian grid coordinates into the FREE_AXES
        coordinates of the geometry, using the geometry's `from_cartesian` method. This is
        useful when you have grid coordinates in a particular order (e.g., Cartesian 'x', 'y', 'z')
        but need to transform them into the coordinate system defined by the geometry,
        and return only the coordinates corresponding to the free axes (i.e., the unconstrained
        axes in a system with symmetry).

        Parameters
        ----------
        grid_axis_order : AxisOrder
            A tuple representing the axis order of the grid (e.g., ('x', 'y', 'z')). This specifies
            the current ordering of the axes in the grid from which coordinates are taken.

        Returns
        -------
        function
            A converter function that takes N input arrays (one for each coordinate in the grid_axis_order)
            and returns M arrays, one for each of the free axes of the geometry.
        """
        # Standard Cartesian axes and determine the dimensionality of the geometry
        standard_cart_axes = ["x", "y", "z"]
        ndim = len(self.AXES)
        # Precompute grid-to-standard axis mapping for efficient reordering
        _grid_to_std = np.array(
            [standard_cart_axes.index(ax) for ax in grid_axis_order], dtype=int
        )
        # Identify missing coordinates that need to be filled with zeros
        # missing_indices = [
        #    standard_cart_axes.index(axis)
        #    for axis in standard_cart_axes
        #    if axis not in grid_axis_order
        # ]

        # Precompute free axes indices for optimized extraction from geometry-specific coordinates
        free_axes_indices = np.array(
            [self.AXES.index(axis) for axis in self.FREE_AXES], dtype=int
        )

        # Step 3: Build the converter function
        def converter(grid_coords):
            """
            Convert the Cartesian coordinates from the grid into the free axes coordinates of the geometry.

            Parameters
            ----------
            grid_coords : tuple of np.ndarray
                Cartesian coordinates in the order defined by `grid_axis_order`.

            Returns
            -------
            tuple of np.ndarray
                Coordinates corresponding to the geometry's FREE_AXES.
            """
            # Pre-allocate sorted_coords with zeros upfront, avoiding the need for additional checks
            coord_shape = grid_coords[0].shape
            sorted_coords = np.zeros((ndim, *coord_shape), dtype=grid_coords[0].dtype)
            # Fill the provided grid coordinates into the proper positions
            sorted_coords[_grid_to_std] = grid_coords

            # Convert the Cartesian coordinates to geometry-specific coordinates
            geometry_coords = self.from_cartesian(*sorted_coords)

            # Extract the free axes using numpy.take for optimized multi-axis extraction
            free_axes_coords = np.take(geometry_coords, free_axes_indices, axis=0)
            return free_axes_coords

        return converter

    @staticmethod
    def _get_class_by_name(class_name: str) -> Type["GeometryHandler"]:
        """
        Retrieve a geometry class by its name, searching through the MRO.

        Parameters
        ----------
        class_name : str
            The name of the geometry class.

        Returns
        -------
        Type[GeometryHandler]
            The class type corresponding to the provided class name, or None if not found.
        """

        def find_in_subclasses(base_class):
            """
            Recursively search subclasses for the class with the given name.

            Parameters
            ----------
            base_class : Type[GeometryHandler]
                The class to start the search from.

            Returns
            -------
            Type[GeometryHandler] or None
                The class type if found, otherwise None.
            """
            for subclass in base_class.__subclasses__():
                if subclass.__name__ == class_name:
                    return subclass
                # Recursively search in subclasses of the current subclass
                result = find_in_subclasses(subclass)
                if result is not None:
                    return result
            return None

        # Start the search from the base GeometryHandler class
        return find_in_subclasses(GeometryHandler)

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Get the parameters defining the geometry.

        Returns
        -------
        dict
            A dictionary containing the initialized parameters for the geometry.
        """
        return self._parameters
