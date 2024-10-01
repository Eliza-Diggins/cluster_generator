"""
Geometry Module for Radial Profiles
===================================

This module provides a set of geometry handlers and utilities designed to support radial profiles
commonly used in astrophysical systems. It includes various geometry classes for handling different
symmetry types (e.g., spherical, cylindrical, ellipsoidal) and methods for calculating volumes
within shells, performing mass integrations, and other geometric operations necessary for working
with radial profiles.

The geometry module abstracts geometry-specific details so that radial profiles can remain agnostic
to the underlying coordinate system, allowing them to be reused in different contexts.

.. rubric:: Features

- Radial geometry handlers for common astrophysical symmetries, such as spherical, cylindrical,
  oblate, and prolate geometries.
- Shell volume and surface area computations for different geometries.
- Integration utilities to perform mass or density calculations across radial shells.
- Error handling and validation to ensure that profile dimensions match the assigned geometry.

.. rubric:: Core Classes

- `RadialGeometryHandler`: Base class for handling radial geometries. All geometry classes derive
  from this and provide implementations of key methods.
- `SphericalGeometryHandler`: Handles spherical symmetry, which is commonly used in galaxy clusters,
  dark matter halos, and other astrophysical systems.
- `CylindricalGeometryHandler`: Supports cylindrical coordinate systems for systems with symmetry
  along a central axis.
- `OblateGeometryHandler`: Specialized handler for flattened, ellipsoidal geometries.
- `ProlateGeometryHandler`: Specialized handler for elongated ellipsoidal geometries.

.. rubric:: Notes

### Radial Geometry Handlers
The `RadialGeometryHandler` and its derived classes (e.g., `SphericalGeometryHandler`,
`CylindricalGeometryHandler`) encapsulate the mathematical formulas and methods for handling radial
symmetry. These classes provide two key functionalities:

1. **Shell Volume Calculation**:
   Every geometry handler implements a method to compute the volume within a radial shell,
   `get_volume_within_shell`, based on the specific geometry. This allows the profile to compute
   the enclosed volume as a function of radius for different symmetries.

   - For **spherical geometries**, the shell volume is calculated as:

     .. math::
        V(r) = \frac{4}{3} \pi r^3

   - For **cylindrical geometries**, the shell volume is calculated by:

     .. math::
        V(r) = 2 \pi r h

     Where \(h\) is the height along the symmetry axis.

   These formulas generalize for other geometries, and the corresponding `GeometryHandler` subclass
   handles the specific calculations.

2. **Integration Over Shells**:
   The `integrate_shells` method is a flexible utility to compute integrals over radial shells.
   It integrates quantities like mass, density, or energy, across shells based on the provided
   profile function and the geometry of the system.

   The method uses **quadrature** from the SciPy library to numerically integrate the desired
   quantity from a starting radius to the desired outer radius. This generalizes across different
   geometries, which are provided by the relevant geometry handler subclass.

### Flexibility of Geometry Abstraction
One of the primary design goals of this module is to allow radial profiles to remain independent
of the specific geometry they are used in. This is achieved through the `GeometryHandler` interface
and its derived classes.

The profile interacts with the geometry only through a set of predefined methods, such as
`get_volume_within_shell` and `integrate_shells`. As long as the geometry handler provides these
methods, any profile can be seamlessly used in different geometries. For example, a profile that
was originally designed for spherical symmetry can be used in a cylindrical or oblate ellipsoidal
geometry by switching the geometry handler at runtime.

This flexibility allows astrophysicists to build reusable components that can be applied to a
variety of different simulations without rewriting the core logic of the profile.

### Error Handling and Validation
The geometry module includes robust error handling to ensure that the profiles and geometry handlers
work together as expected. For instance:

- **Dimensional Validation**: When a profile is initialized with a geometry, the number of dimensions
  in the geometry must match the profile's expected dimensions. If they do not match, an error is
  raised to prevent incorrect calculations.

- **Shell Volume Computation**: If the shell volume computation fails due to invalid geometry
  parameters, such as non-physical values (negative radii, for example), an error is raised with
  a meaningful message indicating the issue.

### Geometrical Symmetries
This module supports a range of geometrical symmetries. Some of the most common geometries used in
astrophysics are:

- **Spherical Symmetry**: Often used for galaxy clusters, dark matter halos, and star clusters. In
  spherical symmetry, the only relevant coordinate is the radial distance from the center of the
  system, and the geometry assumes that the system is spherically symmetric in all directions.

- **Cylindrical Symmetry**: Useful for modeling systems such as accretion disks or galaxies with
  rotational symmetry around a central axis. In this case, both the radial and axial coordinates
  are relevant, and the system is assumed to be symmetric in the azimuthal direction.

- **Oblate and Prolate Symmetry**: These geometries are useful for systems that are either flattened
  (oblate) or elongated (prolate) along one axis. Such symmetries are often used for ellipsoidal
  galaxies or rotating star clusters.

### Integration with Profiles
The geometry module is designed to work seamlessly with the **profiles module**. Each radial profile
(e.g., density profile, mass profile) is assigned a specific geometry handler that governs how
geometrical operations are performed. This means that profiles do not need to be modified when
switching from spherical to cylindrical or oblate geometries. Instead, the geometry handler takes
care of all necessary adjustments.

For example, a `RadialDensityProfile` can be defined in spherical geometry, but it can be seamlessly
converted to cylindrical geometry by simply passing a different geometry handler, without any changes
to the profile's core logic.

.. rubric:: Usage Example

Here is an example demonstrating how to use the geometry handler to compute the volume within a
radial shell for spherical geometry:

.. code-block:: python

    from cluster_generator.geometry.radial import SphericalGeometryHandler

    # Define a spherical geometry handler
    spherical_geometry = SphericalGeometryHandler()

    # Compute the volume within a shell between radii 1 and 10 kpc
    r_min = 1.0
    r_max = 10.0
    volume = spherical_geometry.get_volume_within_shell(r_min, r_max)
    print(f"Volume within shell: {volume} kpc^3")

"""

from .radial import (
    OblateGeometryHandler,
    ProlateGeometryHandler,
    RadialGeometryHandler,
    SphericalGeometryHandler,
)

__all__ = [
    "RadialGeometryHandler",
    "SphericalGeometryHandler",
    "OblateGeometryHandler",
    "ProlateGeometryHandler",
]
