"""
Radial Geometry Module for Cluster Generator
============================================

This module defines various geometries with radial symmetry, which reduces
the problem to a single free dimension. These geometries include spherical,
oblate, prolate, and triaxial forms, all characterized by their specific
radial properties and symmetry.

The module provides:
    - Conversion functions between Cartesian and radial coordinate systems.
    - Volume element functions tailored to each geometry.
    - Gradient computation methods that incorporate the Lame coefficient.
    - Support for spline-based gradient computation in radial geometries.

Geometries included:
    - :py:class:`SphericalGeometryHandler`
    - :py:class:`OblateGeometryHandler`
    - :py:class:`ProlateGeometry`
    - :py:class:`TriaxialGeometry`

Examples
--------
Here’s an example usage of the `SphericalGeometryHandler` class to convert
coordinates between Cartesian and spherical systems:

.. code-block:: python

    spherical_handler = SphericalGeometryHandler()

    # Cartesian coordinates
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0])
    z = np.array([1.0, 2.0])

    # Convert to spherical coordinates
    r, theta, phi = spherical_handler.from_cartesian(x, y, z)

    # Convert back to Cartesian coordinates
    x_cart, y_cart, z_cart = spherical_handler.to_cartesian(r, theta, phi)

See Also
--------
- :py:class:`GeometryHandler`
    The base class for all geometry handlers used in the cluster generator.

Raises
------
InvalidGeometryParameterError
    Raised when an unexpected geometry parameter is provided during initialization.

Notes
-----
- These geometries are designed to be used in modeling environments where radial symmetry reduces computational complexity.
- For more complex, non-symmetric geometries, use the `TriaxialGeometry` class, which supports asymmetric ellipsoids.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple

import numpy as np

from cluster_generator.geometry.abc import GeometryHandler
from cluster_generator.geometry._types import (
    AxisName,
    AxisOrder,
    LameCFunction,
    LameCFunctions3D,
    ProfileInput,
    ProfileResult,
    SymmetryType,
    VolumeElement_3D,
)

if TYPE_CHECKING:
    from cluster_generator.grids.grids import Grid, GridLevel


class RadialGeometryHandler(GeometryHandler, ABC):
    """
    Base class for handling radial geometries.

    This class is designed to handle generic radial geometries, providing a basis for
    spherical, cylindrical, or other radial symmetry calculations. It includes methods
    for volume integration over radial shells and ensures that the geometry's symmetry
    and axes are properly handled.

    Attributes
    ----------
    NAME : :py:class:`AxisName`
        Name of the geometry type (e.g., 'GenericRadial'). Defaults to 'GenericRadial'.

    AXES : :py:class:`AxisOrder`
        A tuple representing the default axes for the geometry (e.g., ['r', 'phi', 'theta']).

    SYMMETRY_AXES : :py:class:`AxisOrder`
        Axes of symmetry for the radial geometry.

    SYMMETRY : :py:class:`SymmetryType`
        Symmetry type for the geometry (e.g., spherical symmetry, cylindrical symmetry).

    FREE_AXES : :py:class:`AxisOrder`
        Free axes in the geometry, representing non-symmetric directions (e.g., 'r' for radial symmetry).

    _expected_parameters : Dict[str, Any]
        A dictionary that defines the expected parameters for specific radial geometries.
        This allows customization of geometry-specific properties (e.g., eccentricity).

    Raises
    ------
    InvalidGeometryParameterError
        If an unexpected parameter is provided during initialization.
    """

    NAME: AxisName = "GenericRadial"
    AXES: AxisOrder = ["r", "phi", "theta"]
    SYMMETRY_AXES: AxisOrder = ["phi", "theta"]
    SYMMETRY: SymmetryType = SymmetryType.SPHERICAL
    FREE_AXES: AxisOrder = ["r"]

    _expected_parameters: Dict[str, Any] = {}

    def __init__(self, **kwargs):
        """
        Initialize the radial geometry handler with specified parameters.

        This constructor allows the initialization of radial geometry handlers with
        specific properties or parameters such as eccentricity for oblate or prolate geometries.

        Parameters
        ----------
        **kwargs :
            Parameters specific to the geometry, such as eccentricity or flattening for
            different radial geometries. These parameters are passed as keyword arguments
            to the geometry handler.

        Raises
        ------
        InvalidGeometryParameterError
            If any unexpected parameter is provided that is not expected by the geometry.

        Examples
        --------
        .. code-block:: python

            geometry = RadialGeometryHandler(eccentricity=0.2)  # Initialize with custom parameter
        """
        super().__init__(**kwargs)
        self.shell_volume_function: Callable[
            [ProfileInput], ProfileResult
        ] = self._get_shell_volume_function()

    @abstractmethod
    def _get_shell_volume_function(self) -> Callable[[ProfileInput], ProfileResult]:
        """
        Abstract method to return the appropriate shell volume function for the geometry.

        The shell volume function depends on the specific radial geometry being handled.
        For example, in spherical symmetry, the shell volume scales as `4 * pi * r^2`.

        This method should be implemented in subclasses for each specific geometry.

        Returns
        -------
        Callable[[ProfileInput], ProfileResult]
            A callable function that computes the shell volume for a given radius.

        Notes
        -----
        This method must be implemented in subclasses that define specific radial geometries.
        """
        pass

    @abstractmethod
    def get_volume_within_shell(self, r: ProfileInput):
        pass

    def integrate_shells(
        self, func: Callable[[ProfileInput], ProfileResult], r: ProfileInput
    ) -> ProfileResult:
        """
        Integrate a function over radial shells using quadrature.

        This method performs volume integration over radial shells by integrating a
        provided function `func` over the range `[0, r]`. The volume of each shell
        depends on the specific geometry and is calculated using the `shell_volume_function`.

        Parameters
        ----------
        func : Callable[[ProfileInput], ProfileResult]
            The function to integrate over the radial shells. This function takes
            the radius as input and returns the value to integrate (e.g., a density
            profile or a mass profile).

        r : ProfileInput
            The radius or array of radii at which to perform the integration.

        Returns
        -------
        ProfileResult
            The integrated result over the radial shells.

        Raises
        ------
        InvalidGeometryParameterError
            If the geometry parameters are not valid for the integration.

        Examples
        --------
        .. code-block:: python

            from scipy.integrate import quad

            def density_profile(r):
                return r**-2

            geometry = RadialGeometryHandler()
            mass_profile = geometry.integrate_shells(density_profile, r=[10, 20, 50])
            print(mass_profile)  # Returns the integrated mass profile at the specified radii.
        """
        from scipy.integrate import quad

        # Ensure that radii is an array-like data structure to minimize syntactic logic.
        if not isinstance(r, np.ndarray):
            r = np.array([r])

        # Create an array to store the results of the integration.
        res = np.zeros_like(r)

        # Define the integrand, which combines the function `func` with the shell volume.
        _integrand = lambda _r, sve=self.shell_volume_function: func(_r) * sve(_r)

        # Perform the integration for each radius in `r`.
        for i, _ri in enumerate(r):
            res[i] = quad(_integrand, 0, _ri)[0]

        return res

    def get_radii_from_grid(self, grid: "Grid") -> np.ndarray:
        """
        Retrieve the radial distances for each cell in the given grid.

        This method takes the grid's Cartesian coordinates, converts them to the
        radial coordinates specified by this geometry handler, and extracts the "r" axis.

        Parameters
        ----------
        grid : Grid
            The grid instance containing Cartesian coordinates.

        Returns
        -------
        np.ndarray
            The radial distances for each cell in the grid.

        Raises
        ------
        TypeError
            If the grid does not provide valid Cartesian coordinates.
        ValueError
            If this geometry does not have an "r" axis in its FREE_AXES.
        """
        # Step 1: Ensure this geometry has an "r" axis as a free axis
        if "r" not in self.FREE_AXES:
            raise ValueError("The geometry must have an 'r' axis as a free axis to compute radii.")

        # Step 2: Retrieve Cartesian coordinates from the grid
        cartesian_coords = grid.get_coordinates()
        # Step 3: Convert the Cartesian coordinates to this geometry's radial coordinates
        radial_coords = self.build_converter(grid.grid_manager.AXES)(cartesian_coords)

        # Step 4: Extract the radius component from the radial coordinates
        radius_index = self.FREE_AXES.index("r")
        radii = radial_coords[radius_index]

        return radii

    def get_min_radii_spacing(self, grid_level: "GridLevel") -> float:
        """
        Compute the minimum radial spacing (Delta r) based on the cell size of the provided grid level.

        This method calculates Delta r by taking the Cartesian cell size of the grid level and transforming it
        into the radial coordinate in this geometry. The result is the radial spacing, Delta r, for a single cell.

        Parameters
        ----------
        grid_level : GridLevel
            The grid level instance providing the CELL_SIZE attribute, which represents the Cartesian
            cell size in each spatial dimension.

        Returns
        -------
        float
            The computed Delta r, representing the minimum radial spacing for a cell in this geometry.

        Raises
        ------
        ValueError
            If the geometry does not support a radial coordinate transformation or lacks an "r" axis.

        Notes
        -----
        This method is geometry-dependent and assumes that the geometry can interpret
        the Cartesian cell size to provide a radial distance.
        """
        # Step 1: Ensure this geometry has an "r" axis to compute radial spacing
        if "r" not in self.FREE_AXES:
            raise ValueError("The geometry must have an 'r' axis to compute radial spacing.")

        # Step 2: Extract CELL_SIZE from the grid level, which is the Cartesian spacing in each direction
        cell_size_vector = grid_level.CELL_SIZE
        cell_size_vector = cell_size_vector.reshape((cell_size_vector.size,1))

        # Step 3: Transform the Cartesian cell size point into radial coordinates
        # by treating it as a Cartesian distance from the origin
        # Step 3: Convert the Cartesian coordinates to this geometry's radial coordinates
        radial_coords = self.build_converter(grid_level.grid_manager.AXES)(*cell_size_vector)

        # Step 4: Extract the radius component from the radial coordinates
        radius_index = self.FREE_AXES.index("r")
        d_radii = radial_coords[radius_index]

        return d_radii

    def get_min_max_radii_from_grid(self, grid: "Grid") -> Tuple[float, float]:
        """
        Efficiently compute the minimum and maximum radial distances for the grid based on its bounding box (BBOX).

        Parameters
        ----------
        grid : Grid
            The grid instance containing the bounding box (BBOX) and cell size.

        Returns
        -------
        Tuple[float, float]
            The minimum and maximum radii for the grid.
        """
        if "r" not in self.FREE_AXES:
            raise ValueError("The geometry must have an 'r' axis to compute radial bounds.")

        # Retrieve bounding box coordinates and cell size
        bbox_min, bbox_max = grid.BBOX[0], grid.BBOX[1]
        dimensions = len(bbox_min)
        cell_size = grid.level.CELL_SIZE

        # Step 1: Generate all corner points and adjust them to approximate cell centers
        corners = np.array(np.meshgrid(*zip(bbox_min, bbox_max))).T.reshape(-1, dimensions)
        half_cell_offset = cell_size / 2
        centered_corners = corners + np.where(corners == bbox_min, half_cell_offset, -half_cell_offset)

        # Step 2: Check and add zero-crossing points where applicable, adjusted to cell centers
        zero_crossings = []
        for i in range(dimensions):
            if bbox_min[i] < 0 < bbox_max[i]:
                zero_cross = centered_corners.copy()
                zero_cross[:, i] = 0  # Set to zero for crossing axis
                zero_crossings.append(zero_cross)

        if zero_crossings:
            zero_crossings = np.vstack(zero_crossings)
            all_points = np.vstack([centered_corners, zero_crossings])
        else:
            all_points = centered_corners

        # Step 3: Convert all points to radial distances in one vectorized operation
        radial_coords = self.build_converter(grid.grid_manager.AXES)(all_points.T)
        radii = np.abs(radial_coords[self.FREE_AXES.index("r")])

        # Step 4: Determine the min and max radii
        return radii.min(), radii.max()


class SphericalGeometryHandler(RadialGeometryHandler):
    """
    Class for handling spherical geometry.

    Provides conversion between Cartesian and spherical coordinates and methods for
    computing volume elements and Lame coefficients in spherical geometry.


    Examples
    --------
    Convert Cartesian coordinates to spherical coordinates:

    .. code-block:: python

        spherical_handler = SphericalGeometryHandler()
        r, theta, phi = spherical_handler.from_cartesian(x, y, z)
    """

    NAME: AxisName = "Spherical"
    SYMMETRY_TYPE: SymmetryType = SymmetryType.SPHERICAL
    SYMMETRY_AXES: AxisOrder = ["phi", "theta"]
    FREE_AXES: AxisOrder = ["r"]

    _expected_parameters: Dict[str, Any] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "<SphericalGeometryHandler>"

    def _get_shell_volume_function(self) -> Callable[[ProfileInput], ProfileResult]:
        return lambda r: 4 * np.pi * r**2

    def get_volume_within_shell(self, r: ProfileInput):
        return (4 / 3) * np.pi * r**3

    def _get_volume_element_function(self) -> VolumeElement_3D:
        """
        Return the volume element function for spherical geometry.

        Returns
        -------
        VolumeElement_3D
            The volume element function.
        """
        return lambda r, _, theta: np.sin(theta) * r**2

    @staticmethod
    def _get_lame_r() -> LameCFunction:
        def _lame_r(*args: ProfileInput) -> ProfileResult:
            return np.ones_like(args[0])

        return _lame_r

    @staticmethod
    def _get_lame_phi() -> LameCFunction:
        def _lame_phi(*args: ProfileInput) -> ProfileResult:
            r, theta = args[1], args[2]

            return r * np.sin(theta)

        return _lame_phi

    @staticmethod
    def _get_lame_theta() -> LameCFunction:
        def _lame_theta(*args: ProfileInput) -> ProfileResult:
            r = args[1]

            return r

        return _lame_theta

    def _get_lame_functions(self) -> LameCFunctions3D:
        return (
            self._get_lame_r(),
            self._get_lame_phi(),
            self._get_lame_theta(),
        )

    def from_cartesian(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi).

        Parameters
        ----------
        x, y, z : np.ndarray
            Cartesian coordinates.

        Returns
        -------
        tuple of np.ndarray
            Spherical coordinates (r, theta, phi).

        """
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        phi = np.arctan2(np.sqrt(x**2 + y**2), z)
        return r, theta, phi

    def to_cartesian(
        self, r: np.ndarray, theta: np.ndarray, phi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z).

        Parameters
        ----------
        r, theta, phi : np.ndarray
            Spherical coordinates.

        Returns
        -------
        tuple of np.ndarray
            Cartesian coordinates (x, y, z).

        """
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x = r * sin_phi * cos_theta
        y = r * sin_phi * sin_theta
        z = r * cos_phi

        return x, y, z


class OblateGeometryHandler(RadialGeometryHandler):
    """
    Class for handling oblate spheroidal geometry.


    Notes
    -----
    Oblate spheroidal coordinates are used to model ellipsoids where the z-axis is shorter
    than the x and y axes. This is useful in scenarios where an ellipsoid has rotational symmetry
    about the z-axis but is flattened along this axis.
    """

    NAME: AxisName = "Oblate"
    SYMMETRY_TYPE: SymmetryType = SymmetryType.OBLATE
    SYMMETRY_AXES: AxisOrder = ["z"]
    FREE_AXES: AxisOrder = ["r"]

    _expected_parameters: Dict[str, Any] = {"ecc": 0.0}

    def __init__(self, ecc: float = _expected_parameters["ecc"]):
        super().__init__(ecc=ecc)

    def get_volume_within_shell(self, r: ProfileInput):
        return (4 / 3) * np.sqrt(1 - self.parameters["ecc"] ** 2) * np.pi * r**3

    def _get_shell_volume_function(self) -> Callable[[ProfileInput], ProfileResult]:
        return (
            lambda r, _e=self.parameters["ecc"]: 4
            * np.pi
            * r**2
            * np.sqrt(1 - _e**2)
        )

    def _get_volume_element_function(self) -> VolumeElement_3D:
        return (
            lambda r, _, theta, _e=self.parameters["ecc"]: np.sqrt(1 - _e**2)
            * np.sin(theta)
            * r**2
        )

    def _get_lame_r(self) -> LameCFunction:
        def _lame_r(*args: ProfileInput, e=self.parameters["ecc"]) -> ProfileResult:
            theta = args[2]
            chi = 1 / np.sqrt(1 - e**2)

            t1 = (np.cos(theta) * chi) ** 2
            t2 = (np.sin(theta)) ** 2

            return np.sqrt(t1 + t2)

        return _lame_r

    def _get_lame_theta(self) -> LameCFunction:
        def _lame_theta(*args: ProfileInput, e=self.parameters["ecc"]) -> ProfileResult:
            r, theta = args[1], args[2]
            chi = 1 / np.sqrt(1 - e**2)

            t1 = (np.sin(theta) * chi) ** 2
            t2 = (np.cos(theta)) ** 2

            return r * np.sqrt(t1 + t2)

        return _lame_theta

    @staticmethod
    def _get_lame_phi() -> LameCFunction:
        def _lame_phi(*args: ProfileInput) -> ProfileResult:
            r, theta = args[1], args[2]

            return r * np.sin(theta)

        return _lame_phi

    def _get_lame_functions(self) -> LameCFunctions3D:
        return (
            self._get_lame_r(),
            self._get_lame_phi(),
            self._get_lame_theta(),
        )

    def from_cartesian(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert Cartesian coordinates (x, y, z) to oblate spheroidal coordinates (r, theta, phi).

        Parameters
        ----------
        x, y, z : np.ndarray
            Cartesian coordinates.

        Returns
        -------
        tuple of np.ndarray
            Oblate spheroidal coordinates (r, theta, phi).

        """
        e2 = self.parameters["ecc"] ** 2
        r = np.sqrt(x**2 + y**2 + z**2 / (1 - e2))
        theta = np.arctan2(
            np.sqrt(x**2 + y**2), z * np.sqrt(1 - e2)
        )  # Adjusted theta computation
        phi = np.arctan2(y, x)
        return r, theta, phi

    def to_cartesian(
        self, r: np.ndarray, theta: np.ndarray, phi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert oblate spheroidal coordinates (r, theta, phi) to Cartesian coordinates (x, y, z).

        Parameters
        ----------
        r, theta, phi : np.ndarray
            Oblate spheroidal coordinates.

        Returns
        -------
        tuple of np.ndarray
            Cartesian coordinates (x, y, z).

        """
        e2 = self.parameters["ecc"] ** 2
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta) / np.sqrt(1 - e2)
        return x, y, z


class ProlateGeometryHandler(RadialGeometryHandler):
    """
    Class for handling prolate spheroidal geometry.

    Methods
    -------
    from_cartesian(x, y, z)
        Convert Cartesian coordinates to prolate spheroidal coordinates.
    to_cartesian(r, theta, phi)
        Convert prolate spheroidal coordinates to Cartesian coordinates.
    _get_lame_functions()
        Get Lame coefficient functions for prolate spheroidal geometry.
    _get_volume_element_function()
        Return the function computing the volume element for prolate geometry.

    Notes
    -----
    Prolate spheroidal coordinates are used to model ellipsoids where the z-axis is longer than
    the x and y axes. This is useful in scenarios where an ellipsoid has rotational symmetry
    about the z-axis but is elongated along this axis.
    """

    NAME: AxisName = "Prolate"
    SYMMETRY: SymmetryType = SymmetryType.PROLATE
    SYMMETRY_AXES: AxisOrder = ["z"]
    FREE_AXES: AxisOrder = ["r"]

    _expected_parameters = {"ecc": 0.0}

    def __init__(self, ecc: float = 0.0):
        super().__init__(ecc=ecc)

    def get_volume_within_shell(self, r: ProfileInput):
        return (4 / 3) * (1 - self.parameters["ecc"] ** 2) * np.pi * r**3

    def _get_shell_volume_function(self) -> Callable[[ProfileInput], ProfileResult]:
        return lambda r, _e=self.parameters["ecc"]: 4 * np.pi * r**2 * (1 - _e**2)

    def _get_volume_element_function(self) -> VolumeElement_3D:
        return (
            lambda r, _, theta, _e=self.parameters["ecc"]: (1 - _e**2)
            * np.sin(theta)
            * r**2
        )

    def _get_lame_r(self) -> LameCFunction:
        def _lame_r(*args: ProfileInput, e=self.parameters["ecc"]) -> ProfileResult:
            theta = args[2]
            chi = 1 / np.sqrt(1 - e**2)

            t1 = (np.cos(theta)) ** 2
            t2 = (chi * np.sin(theta)) ** 2

            return np.sqrt(t1 + t2)

        return _lame_r

    def _get_lame_theta(self) -> LameCFunction:
        def _lame_theta(*args: ProfileInput, e=self.parameters["ecc"]) -> ProfileResult:
            r, theta = args[1], args[2]
            chi = 1 / np.sqrt(1 - e**2)

            t1 = (np.cos(theta) * chi) ** 2
            t2 = (np.sin(theta)) ** 2

            return r * np.sqrt(t1 + t2)

        return _lame_theta

    def _get_lame_phi(self) -> LameCFunction:
        def _lame_phi(*args: ProfileInput, e=self.parameters["ecc"]) -> ProfileResult:
            r, theta = args[1], args[2]
            chi = 1 / np.sqrt(1 - e**2)

            return r * chi * np.sin(theta)

        return _lame_phi

    def _get_lame_functions(self) -> LameCFunctions3D:
        return (
            self._get_lame_r(),
            self._get_lame_phi(),
            self._get_lame_theta(),
        )

    def from_cartesian(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert Cartesian coordinates (x, y, z) to prolate spheroidal coordinates (r, theta, phi).

        Parameters
        ----------
        x, y, z : np.ndarray
            Cartesian coordinates.

        Returns
        -------
        tuple of np.ndarray
            Prolate spheroidal coordinates (r, theta, phi).

        """
        e2 = self.parameters["ecc"] ** 2
        r = np.sqrt(x**2 + y**2 + z**2 / (e2 + 1))
        theta = np.arctan2(np.sqrt(x**2 + y**2), z * np.sqrt(e2 + 1))
        phi = np.arctan2(y, x)
        return r, theta, phi

    def to_cartesian(
        self, r: np.ndarray, theta: np.ndarray, phi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert prolate spheroidal coordinates (r, theta, phi) to Cartesian coordinates (x, y, z).

        Parameters
        ----------
        r, theta, phi : np.ndarray
            Prolate spheroidal coordinates.

        Returns
        -------
        tuple of np.ndarray
            Cartesian coordinates (x, y, z).

        """
        e2 = self.parameters["ecc"] ** 2
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta) * np.sqrt(e2 + 1)
        return x, y, z


class TriaxialGeometryHandler(RadialGeometryHandler):
    """
    Class for handling triaxial geometry with two free dimensions and no symmetry.

    Attributes
    ----------
    NAME : AxisName
        The name of the geometry ('Triaxial').

    SYMMETRY : SymmetryType
        The symmetry type of the geometry ('Triaxial').

    Notes
    -----
    Triaxial spheroidal coordinates are useful for geometries where the three axes (x, y, and z) have different lengths,
    resulting in no symmetry along any axis. This can be used to model ellipsoids with unequal axes.


    Examples
    --------
    .. code-block:: python

        triaxial_handler = TriaxialGeometry(ecc1=0.5, ecc2=0.3)
        r, theta, phi = triaxial_handler.from_cartesian(x, y, z)
        x_cart, y_cart, z_cart = triaxial_handler.to_cartesian(r, theta, phi)
    """

    NAME: AxisName = "Triaxial"
    SYMMETRY: SymmetryType = SymmetryType.TRIAXIAL
    SYMMETRY_AXES: AxisOrder = []
    FREE_AXES: AxisOrder = ["r"]

    _expected_parameters = {"ecc1": 0.0, "ecc2": 0.0}

    def __init__(self, ecc1: float = 0.0, ecc2: float = 0.0):
        super().__init__(ecc1=ecc1, ecc2=ecc2)

    def get_volume_within_shell(self, r: ProfileInput):
        return (
            (4 / 3)
            * np.sqrt(1 - self.parameters["ecc2"] ** 2)
            * np.sqrt(1 - self.parameters["ecc1"] ** 2)
            * np.pi
            * r**3
        )

    def _get_shell_volume_function(self) -> Callable[[ProfileInput], ProfileResult]:
        return (
            lambda r, _e1=self.parameters["ecc1"], _e2=self.parameters["ecc2"]: 4
            * np.pi
            * r**2
            * np.sqrt(1 - _e1**2)
            * np.sqrt(1 - _e2**2)
        )

    def _get_volume_element_function(self) -> VolumeElement_3D:
        return (
            lambda r, _, theta, _e1=self.parameters["ecc1"], _e2=self.parameters[
                "ecc2"
            ]: np.sqrt(1 - _e1**2)
            * np.sqrt(1 - _e2**2)
            * np.sin(theta)
            * r**2
        )

    def _get_lame_r(self) -> LameCFunction:
        def _lame_r(
            *args: ProfileInput, e1=self.parameters["ecc1"], e2=self.parameters["ecc2"]
        ) -> ProfileResult:
            phi, theta = args[1], args[2]

            t1 = (np.cos(phi) * np.sin(theta) * (1 / np.sqrt(1 - e1**2))) ** 2
            t2 = (np.sin(phi) * np.sin(theta) * (1 / np.sqrt(1 - e2**2))) ** 2
            t3 = np.cos(theta) ** 2

            return np.sqrt(t1 + t2 + t3)

        return _lame_r

    def _get_lame_phi(self) -> LameCFunction:
        def _lame_phi(
            *args: ProfileInput, e1=self.parameters["ecc1"], e2=self.parameters["ecc2"]
        ) -> ProfileResult:
            r, phi, theta = args[0], args[1], args[2]

            t1 = (np.sin(phi) * np.sin(theta) * (1 / np.sqrt(1 - e1**2))) ** 2
            t2 = (np.cos(phi) * np.sin(theta) * (1 / np.sqrt(1 - e2**2))) ** 2

            return r * np.sqrt(t1 + t2)

        return _lame_phi

    def _get_lame_theta(self) -> LameCFunction:
        def _lame_theta(
            *args: ProfileInput, e1=self.parameters["ecc1"], e2=self.parameters["ecc2"]
        ) -> ProfileResult:
            r, phi, theta = args[0], args[1], args[2]

            t1 = (np.cos(phi) * np.cos(theta) * (1 / np.sqrt(1 - e1**2))) ** 2
            t2 = (np.sin(phi) * np.cos(theta) * (1 / np.sqrt(1 - e2**2))) ** 2
            t3 = np.sin(theta) ** 2

            return r * np.sqrt(t1 + t2 + t3)

        return _lame_theta

    def _get_lame_functions(self) -> LameCFunctions3D:
        return (
            self._get_lame_r(),
            self._get_lame_phi(),
            self._get_lame_theta(),
        )

    def from_cartesian(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert Cartesian coordinates (x, y, z) to triaxial spheroidal coordinates (r, theta, phi).

        Parameters
        ----------
        x, y, z : np.ndarray
            Cartesian coordinates.

        Returns
        -------
        tuple of np.ndarray
            Triaxial spheroidal coordinates (r, theta, phi).


        """
        e1_squared = self.parameters["ecc1"] ** 2
        e2_squared = self.parameters["ecc2"] ** 2
        r = np.sqrt(x**2 + y**2 / (1 - e1_squared) + z**2 / (1 - e2_squared))
        theta = np.arctan2(np.sqrt(x**2 + y**2), z * np.sqrt(1 - e2_squared))
        phi = np.arctan2(y, x)
        return r, theta, phi

    def to_cartesian(
        self, r: np.ndarray, theta: np.ndarray, phi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert triaxial spheroidal coordinates (r, theta, phi) to Cartesian coordinates (x, y, z).

        Parameters
        ----------
        r, theta, phi : np.ndarray
            Triaxial spheroidal coordinates.

        Returns
        -------
        tuple of np.ndarray
            Cartesian coordinates (x, y, z).

        """
        e1_squared = self.parameters["ecc1"] ** 2
        e2_squared = self.parameters["ecc2"] ** 2
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi) * np.sqrt(1 - e1_squared)
        z = r * np.cos(theta) * np.sqrt(1 - e2_squared)
        return x, y, z
