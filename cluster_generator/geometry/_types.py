from enum import Enum
from typing import Callable, Tuple, Type, TypeVar

from numpy.typing import NDArray

_GeometryHandler = TypeVar("_GeometryHandler")
GeometryName = str
AxisName = str
AxisOrder = tuple[AxisName, AxisName, AxisName]
ProfileResult = TypeVar("ProfileResult", float, NDArray[float], int, NDArray[int])
ProfileInput = ProfileResult


class MissingGeometryAttribute:
    """
    Sentinel descriptor that raises a NotImplementedError when accessed.
    Used to enforce abstract class attributes.

    This descriptor is used to enforce that subclasses must define certain
    optional attributes. If the attribute is accessed without being defined,
    a `NotImplementedError` is raised.

    Attributes
    ----------
    _name : str
        Name of the attribute.

    Examples
    --------
    .. code-block:: python

        class Example:
            missing_attribute = MissingOptionalAttribute()

            def __init__(self):
                print(self.missing_attribute)

        # Will raise NotImplementedError:
        e = Example()

    """

    def __set_name__(self, owner: Type["_GeometryHandler"], name: str):
        self._name: str = name

    def __get__(self, instance: "_GeometryHandler", owner: Type["_GeometryHandler"]):
        raise NotImplementedError(f"{owner.__name__} does not define {self._name}.")


class SymmetryType(Enum):
    """Enumeration for various forms of symmetry.

    Notes
    -----
    Generically, a given geometry class (descended from :py:class:`BaseGeometry`) only exists because
    it facilitates a particular symmetry. The actual functionality of each symmetry is encapsulated in the
    geometry class; not in the :py:class:`SymmetryType` class.
    """

    NONE = "none"  # No reduction in dimensions
    SPHERICAL = "spherical"  # Reduces 3D to 1D (radial symmetry)
    AXISYMMETRIC = "axisymmetric"  # Reduces 3D to 2D (cylindrical symmetry)
    TRIAXIAL = "triaxial"  # No reduction, retains all 3 dimensions
    OBLATE = (
        "oblate"  # Special case of triaxial, typically for specific ellipsoidal shapes
    )
    PROLATE = "prolate"  # Similar to oblate, specific shape, no generic reduction

    @property
    def dimension_reduction(self) -> int:
        """Return the reduction in dimensions for each symmetry type.

        Returns
        -------
        int
            The number of dimensions reduced by this symmetry type.
            E.g., 2 means the original dimensionality is reduced by 2.
        """
        reductions = {
            SymmetryType.NONE: 0,
            SymmetryType.SPHERICAL: 2,  # e.g., from 3D to 1D
            SymmetryType.AXISYMMETRIC: 1,  # e.g., from 3D to 2D
            SymmetryType.TRIAXIAL: 0,
            SymmetryType.OBLATE: 1,
            SymmetryType.PROLATE: 1,
        }
        return reductions[self]

    @property
    def grid_axis_orders(self):
        """
        Returns the default grid axis order based on the symmetry type of the geometry.

        This property provides a mapping between the symmetry type of the geometry and the
        corresponding grid axis order. The grid axis order defines which coordinates should
        be used for the grid in different geometries.

        Symmetry types reduce the dimensionality of the problem, so the axis orders
        reflect the dimensional reduction. For example, spherical symmetry reduces
        the 3D problem to 1D by using only the radial ('x') coordinate.

        Returns
        -------
        list[str]
            A list representing the axis order used for the grid based on the geometry's symmetry type.

        Raises
        ------
        KeyError
            If the symmetry type of the geometry is not supported.

        Examples
        --------
        For a spherical symmetry, the method would return:

        >>> geometry.grid_axis_orders
        ['x']

        For an axisymmetric geometry, it would return:

        >>> geometry.grid_axis_orders
        ['x', 'z']

        Notes
        -----
        - SymmetryType.NONE assumes a fully 3D Cartesian geometry using the 'x', 'y', and 'z' axes.
        - SymmetryType.SPHERICAL reduces the problem to 1D, keeping only the radial ('x') axis.
        - SymmetryType.AXISYMMETRIC reduces the problem to 2D, using 'x' and 'z'.
        - SymmetryType.TRIAXIAL assumes full 3D, like SymmetryType.NONE, keeping 'x', 'y', and 'z'.
        - SymmetryType.OBLATE and SymmetryType.PROLATE reduce the problem to 2D using 'x' and 'z'.
        """
        # Define the grid axis orders for different symmetry types
        axis_orders_by_symmetry = {
            SymmetryType.NONE: ["x", "y", "z"],  # Full 3D Cartesian coordinates
            SymmetryType.SPHERICAL: ["x"],  # Radial axis only (1D)
            SymmetryType.AXISYMMETRIC: ["x", "z"],  # Symmetry around an axis (2D)
            SymmetryType.TRIAXIAL: [
                "x",
                "y",
                "z",
            ],  # Full 3D Cartesian (similar to NONE)
            SymmetryType.OBLATE: [
                "x",
                "z",
            ],  # 2D for oblate spheroids (flattened at poles)
            SymmetryType.PROLATE: [
                "x",
                "z",
            ],  # 2D for prolate spheroids (elongated along poles)
        }

        # Return the grid axis order based on the symmetry type of the geometry
        # If the symmetry type is not defined, this will raise a KeyError
        return axis_orders_by_symmetry[self]


LameCFunction = Callable[..., ProfileResult]
LameCFunction1D = Callable[[ProfileInput], ProfileResult]
LameCFunction2D = Callable[[ProfileInput, ProfileInput], ProfileResult]
LameCFunction3D = Callable[[ProfileInput, ProfileInput, ProfileInput], ProfileResult]
LameCFunctions = Tuple[LameCFunction]
LameCFunctions2D = Tuple[
    LameCFunction | LameCFunction2D, LameCFunction | LameCFunction2D
]
LameCFunctions3D = Tuple[
    LameCFunction | LameCFunction3D,
    LameCFunction | LameCFunction3D,
    LameCFunction | LameCFunction3D,
]

GenericVolumeElement = Callable[[ProfileInput, ...], ProfileResult]
VolumeElement_1D = Callable[[ProfileInput], ProfileResult]
VolumeElement_2D = Callable[[ProfileInput, ProfileInput], ProfileResult]
VolumeElement_3D = Callable[[ProfileInput, ProfileInput, ProfileInput], ProfileResult]
