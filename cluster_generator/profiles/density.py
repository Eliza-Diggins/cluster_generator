"""
Radial Density Profiles Module
==============================

This module contains various radial density profiles used to describe
the mass distribution of dark matter halos, galaxies, and star clusters.
Each profile assumes radial symmetry and is designed to fit astrophysical
systems, such as galaxy clusters or dark matter halos, with specific parameterizations.

These profiles are widely used in modeling the distribution of dark matter
and baryonic matter in galaxies, galaxy clusters, and dark matter halos.

References:
-----------
- [1] Navarro, J. F., Frenk, C. S., & White, S. D. M. (1997).
      A Universal Density Profile from Hierarchical Clustering.
      The Astrophysical Journal, 490(2), 493–508. https://doi.org/10.1086/304888
- [2] Ascasibar & Markevitch (2006), ApJ, 650, 102
- [3] Lilley et al. (2018), MNRAS, 479, 4126
- [4] Baltz, E.A., Marshall, P., & Oguri, M. (2009), JCAP, 2009, 015
"""
from abc import ABC
from typing import TYPE_CHECKING

import numpy as np

from cluster_generator.profiles._abc import RadialProfile
from cluster_generator.profiles._types import (
    ProfileInput,
    ProfileLinkDescriptor,
    ProfileParameter,
    ProfileResult,
)

if TYPE_CHECKING:
    from yt.utilities.cosmology import Cosmology


class RadialDensityProfile(RadialProfile, ABC):
    """
    A class representing a radial density profile for 1D systems. This class provides methods
    for integrating mass and rescaling the profile based on a desired mass at a given radius.

    Attributes
    ----------
    NDIMS : int
        The number of dimensions for this profile (default is 1 for radial profiles).
    CLASS_FUNCTIONS : dict or None
        A dictionary mapping different geometries to corresponding profile functions.
    mass : ProfileLinkDescriptor
        A descriptor that links the density profile to the corresponding mass profile, if available.

    Methods
    -------
    integrate_mass(radii: ProfileInput) -> ProfileResult
        Integrates the mass profile over a given range of radii using the geometry handler's method.
    rescale_by_mass(radius: float, target_mass: float) -> "RadialDensityProfile"
        Rescales the density profile such that the total mass enclosed within the specified radius
        equals the target mass.

    See Also
    --------
    ProfileLinkDescriptor : Used to link this class to a corresponding mass profile, if available.

    Examples
    --------
    .. code-block:: python

        profile = RadialDensityProfile(geometry=SphericalGeometryHandler, rho_0=1.0, r_s=10)
        mass_enclosed = profile.integrate_mass(radii=50)
        rescaled_profile = profile.rescale_by_mass(radius=100, target_mass=1e14)
    """

    NDIMS: int = 1
    CLASS_FUNCTIONS = None
    mass = ProfileLinkDescriptor()

    def integrate_mass(self, radii: ProfileInput) -> ProfileResult:
        """
        Integrate the mass profile over a given range of radii.

        Parameters
        ----------
        radii : ProfileInput
            The radial distance(s) over which to integrate the mass.

        Returns
        -------
        ProfileResult
            The integrated mass within the specified radii.

        Examples
        --------
        .. code-block:: python

            profile = RadialDensityProfile(geometry=SphericalGeometryHandler, rho_0=1.0, r_s=10)
            mass_enclosed = profile.integrate_mass(radii=50)  # Calculate the mass enclosed within r=50 kpc.
        """
        return self.geometry_handler.integrate_shells(self._function, radii)

    def rescale_by_mass(
        self, radius: float, target_mass: float
    ) -> "RadialDensityProfile":
        """
        Rescales the density profile so that the total mass enclosed within the specified
        radius equals the target mass.

        This function uses quadrature to integrate the density profile and find the current
        enclosed mass within the radius. It then rescales the density profile by the ratio
        of the target mass to the current mass.

        Parameters
        ----------
        radius : float
            The radius within which the total mass should be rescaled.
        target_mass : float
            The target mass that should be enclosed within the specified radius.

        Returns
        -------
        RadialDensityProfile
            A new instance of the RadialDensityProfile, rescaled to match the target mass.

        Raises
        ------
        ValueError
            If the integrated mass within the specified radius is zero or negative.

        Notes
        -----
        - If a mass profile exists (via the `mass` link), the rescaling will be done using that
          profile instead of integrating the density.
        - The method adjusts only the amplitude of the profile, scaling it by the required factor.

        Examples
        --------
        .. code-block:: python

            profile = RadialDensityProfile(geometry=SphericalGeometryHandler, rho_0=1.0, r_s=10)
            rescaled_profile = profile.rescale_by_mass(radius=100, target_mass=1e14)
        """
        # Check if a linked mass profile exists and use it if available
        try:
            current_mass = self.mass(radius)
        except ValueError:
            # Otherwise, integrate the mass directly using the density profile
            current_mass = self.integrate_mass(radius)[0]

        print(current_mass)

        # If the mass is zero or negative, rescaling is not possible
        if current_mass <= 0:
            raise ValueError(
                f"Cannot rescale: the current enclosed mass within radius {radius} is {current_mass}."
            )

        # Calculate the scaling factor
        scaling_factor = target_mass / current_mass

        # Create a new instance of the profile, rescaled by the factor
        rescaled_parameters = {
            key: val * scaling_factor if key in ["rho_0", "M"] else val
            for key, val in self._parameters.items()
        }

        return self.__class__(geometry=self.geometry_handler, **rescaled_parameters)

    def overdensity(
        self, radii: ProfileInput, z: float | int = 0.0, cosmo: "Cosmology" = None
    ) -> ProfileResult:
        from yt.utilities.cosmology import Cosmology

        if cosmo is None:
            cosmo = Cosmology()

        # Compute the masses at each radius. We can then use this
        # to compute average density within the radius.
        try:
            masses = self.mass(radii)
        except ValueError:
            # Otherwise, integrate the mass directly using the density profile
            masses = self.integrate_mass(radii)

        # Fetch the volumes. This has to go through the geometry handler
        # because different radial geometries will have different volume(r).
        volumes = self.geometry_handler.get_volume_within_shell(radii)
        densities = masses / volumes

        rho_crit = cosmo.critical_density(z).to_value("Msun/kpc**3")
        return densities / rho_crit

    def find_overdensity_radius(
        self,
        delta: float,
        z: float | int = 0.0,
        cosmo: "Cosmology" = None,
        r_guess: float = 1.0,
        r_min: float = 1e-3,
        r_max: float = 1e3,
        r_tol: float = 1e-3,
        a_tol: float = 1e-3,
        max_iter: int = 100,
    ) -> float:
        """
        Iteratively find the radius at which the overdensity equals a specified value.

        Parameters
        ----------
        delta : float
            The desired overdensity relative to the critical density.
        z : float or int, optional
            Redshift at which the overdensity is computed. Default is 0.
        cosmo : Cosmology, optional
            Cosmology object to compute critical density. If not provided, a default is used.
        r_guess : float, optional
            Initial guess for the radius. Default is 1.0 kpc.
        r_min : float, optional
            Minimum radius to consider. Default is 1e-3 kpc.
        r_max : float, optional
            Maximum radius to consider. Default is 1e3 kpc.
        r_tol : float, optional
            Relative tolerance for the radius. Default is 1e-3.
        a_tol : float, optional
            Absolute tolerance for the overdensity. Default is 1e-3.
        max_iter : int, optional
            Maximum number of iterations. Default is 100.

        Returns
        -------
        float
            The radius at which the overdensity equals the target overdensity.

        Raises
        ------
        ValueError
            If the solution does not converge within the given tolerance.
        """
        from scipy.optimize import root_scalar

        def overdensity_difference(r):
            """Helper function to compute the difference between the target and computed overdensity."""
            current_overdensity = self.overdensity(r, z=z, cosmo=cosmo)[
                0
            ]  # Compute overdensity
            return current_overdensity - delta

        # Use a root-finding method (brentq) to find the radius where overdensity equals target
        result = root_scalar(
            overdensity_difference,
            bracket=[r_min, r_max],
            x0=r_guess,
            method="brentq",
            xtol=r_tol,
            rtol=a_tol,
            maxiter=max_iter,
        )

        if not result.converged:
            raise ValueError(
                f"Failed to find radius for target overdensity after {max_iter} iterations"
            )

        return result.root


class NFWDensityProfile(RadialDensityProfile):
    """
    Navarro-Frenk-White (NFW) density profile for modeling the mass distribution
    in dark matter halos [1]_. The profile assumes radial symmetry.

    Use cases:
    ----------
    - Modeling the structure of dark matter halos in both cosmological simulations
      and observational analysis.
    - Used in galaxy cluster studies, weak gravitational lensing, and large-scale structure.

    Attributes
    ----------
    rho_0 : float
        Characteristic density.
    r_s : float
        Scale radius.

    Examples
    --------
    .. code-block:: python

        nfw = NFWDensityProfile(rho_0=0.1, r_s=10.0)
        r_values = np.linspace(0.1, 100, 1000)
        density = nfw(r_values)

    See Also
    --------
    HernquistDensityProfile : Similar profile but with a steeper inner slope.
    CoredNFWDensityProfile : A modification with a central core to flatten the inner region.

    References
    ----------
    .. [1] Navarro, J. F., Frenk, C. S., & White, S. D. M. (1997).
       A Universal Density Profile from Hierarchical Clustering.
       The Astrophysical Journal, 490(2), 493–508. https://doi.org/10.1086/304888
    """

    rho_0 = ProfileParameter(1.0, "Characteristic density [Msun/kpc^3]")
    r_s = ProfileParameter(1.0, "Scale radius [kpc]")

    mass = ProfileLinkDescriptor(
        module="cluster_generator.profiles.mass", profile_class="NFWMassProfile"
    )

    @staticmethod
    def default_principle_function(r, rho_0=1.0, r_s=1.0, **_):
        return rho_0 / (r / r_s * (1 + r / r_s) ** 2)

    CLASS_FUNCTIONS = {"default": default_principle_function}


class HernquistDensityProfile(RadialDensityProfile):
    """
    Hernquist profile for modeling the density distribution in elliptical galaxies and bulges.

    Use cases:
    ----------
    - Commonly used for modeling elliptical galaxies and stellar bulges.
    - It provides a steeper density drop-off at large radii compared to the NFW profile.

    Attributes
    ----------
    rho_0 : float
        Characteristic density.
    r_s : float
        Scale radius.

    Examples
    --------
    .. code-block:: python

        hernquist = HernquistDensityProfile(rho_0=0.5, r_s=5.0)
        r_values = np.linspace(0.1, 50, 500)
        density = hernquist(r_values)

    See Also
    --------
    NFWDensityProfile : The NFW profile provides a shallower central slope.
    DehnenDensityProfile : A generalization of the Hernquist profile with tunable inner slopes.

    References
    ----------
    Hernquist, L. (1990). "An Analytical Model for Spherical Galaxies and Bulges."
    The Astrophysical Journal, 356, 359–364.
    """

    rho_0 = ProfileParameter(1.0, "Characteristic density [Msun/kpc^3]")
    r_s = ProfileParameter(1.0, "Scale radius [kpc]")

    mass = ProfileLinkDescriptor(
        module="cluster_generator.profiles.mass", profile_class="HernquistMassProfile"
    )

    @staticmethod
    def default_principle_function(
        r: float, rho_0: float = 1.0, r_s: float = 1.0, **_
    ) -> float:
        return rho_0 / ((r / r_s) * (1 + r / r_s) ** 3)

    CLASS_FUNCTIONS = {"default": default_principle_function}


class EinastoDensityProfile(RadialDensityProfile):
    """
    Einasto profile for modeling dark matter halos.

    Use cases:
    ----------
    - Used in simulations to model dark matter halos.
    - Provides better fits to simulated dark matter profiles than the NFW profile.

    Attributes
    ----------
    rho_0 : float
        Characteristic density.
    r_s : float
        Scale radius.
    alpha : float
        Shape parameter controlling the curvature of the profile.

    Examples
    --------
    .. code-block:: python

        einasto = EinastoDensityProfile(rho_0=1.0, r_s=10.0, alpha=0.18)
        r_values = np.linspace(0.1, 100, 1000)
        density = einasto(r_values)

    See Also
    --------
    NFWDensityProfile : The NFW profile is commonly used for dark matter halos but has a different inner slope behavior.

    References
    ----------
    Einasto, J. (1965). "On Galactic Models. II. Galactic Models with a Halo."
    Trudy Astrofizicheskogo Instituta Alma-Ata, 5, 87–100.
    """

    rho_0 = ProfileParameter(1.0, "Characteristic density [Msun/kpc^3]")
    r_s = ProfileParameter(1.0, "Scale radius [kpc]")
    alpha = ProfileParameter(0.18, "Shape parameter")

    @staticmethod
    def default_principle_function(
        r: float, rho_0: float = 1.0, r_s: float = 1.0, alpha: float = 0.18, **_
    ) -> float:
        return rho_0 * np.exp(-2 * alpha * ((r / r_s) ** alpha - 1))

    CLASS_FUNCTIONS = {"default": default_principle_function}


class SingularIsothermalDensityProfile(RadialDensityProfile):
    """
    Singular isothermal profile for systems in thermal equilibrium.

    Use cases:
    ----------
    - Often used in modeling large-scale self-gravitating systems in equilibrium.
    - Applicable in modeling galactic halos and cosmological structures.

    Attributes
    ----------
    rho_0 : float
        Characteristic density.

    Examples
    --------
    .. code-block:: python

        singular_isothermal = SingularIsothermalDensityProfile(rho_0=0.1)
        r_values = np.linspace(0.1, 50, 500)
        density = singular_isothermal(r_values)

    See Also
    --------
    CoredIsothermalDensityProfile : A cored version of the isothermal profile.
    NFWDensityProfile : Another commonly used profile for dark matter halos.

    Notes
    -----
    The singular version of the profile is often used in modeling large-scale
    structures like galaxy clusters and cosmological halos where a central
    core is not required.

    """

    rho_0 = ProfileParameter(1.0, "Characteristic density [Msun/kpc^3]")

    @staticmethod
    def default_principle_function(r: float, rho_0: float = 1.0, **_) -> float:
        return rho_0 / r**2  # Singular version

    CLASS_FUNCTIONS = {"default": default_principle_function}


class CoredIsothermalDensityProfile(RadialDensityProfile):
    """
    Cored isothermal profile for systems in thermal equilibrium.

    Use cases:
    ----------
    - Suitable for small-scale self-gravitating systems where the central region needs to be flattened.
    - Useful for modeling systems like dwarf galaxies and globular clusters.

    Attributes
    ----------
    rho_0 : float
        Characteristic density.
    r_c : float
        Core radius.

    Examples
    --------
    .. code-block:: python

        cored_isothermal = CoredIsothermalDensityProfile(rho_0=0.1, r_c=2.0)
        r_values = np.linspace(0.1, 50, 500)
        density = cored_isothermal(r_values)

    See Also
    --------
    SingularIsothermalDensityProfile : The singular version of the isothermal profile.
    NFWDensityProfile : Another commonly used profile for dark matter halos.

    Notes
    -----
    The cored version of the isothermal profile is suitable for small-scale systems,
    where the density in the central region is flattened to avoid singularities.
    This makes it ideal for modeling systems like dwarf galaxies and globular clusters.

    """

    rho_0 = ProfileParameter(1.0, "Characteristic density [Msun/kpc^3]")
    r_c = ProfileParameter(1.0, "Core radius [kpc]")

    @staticmethod
    def default_principle_function(
        r: float, rho_0: float = 1.0, r_c: float = 1.0, **_
    ) -> float:
        return rho_0 / (1 + (r / r_c) ** 2)  # Cored version

    CLASS_FUNCTIONS = {"default": default_principle_function}


class PlummerDensityProfile(RadialDensityProfile):
    """
    Plummer profile for modeling star clusters and stellar bulges.

    Use cases:
    ----------
    - Widely used for modeling globular clusters and stellar bulges.
    - Provides a softened inner core, avoiding singularities at small radii.

    Attributes
    ----------
    M : float
        Total mass.
    r_s : float
        Plummer radius.

    Examples
    --------
    .. code-block:: python

        plummer = PlummerDensityProfile(M=1e5, r_s=1.0)
        r_values = np.linspace(0.1, 10, 100)
        density = plummer(r_values)

    See Also
    --------
    KingDensityProfile : Often used for globular clusters with a truncation radius.

    References
    ----------
    Plummer, H. C. (1911). "On the problem of distribution in globular star clusters."
    Monthly Notices of the Royal Astronomical Society, 71, 460-470.
    """

    M = ProfileParameter(1.0, "Total mass [Msun]")
    r_s = ProfileParameter(1.0, "Plummer radius [kpc]")

    mass = ProfileLinkDescriptor(
        module="cluster_generator.profiles.mass", profile_class="PlummerMassProfile"
    )

    @staticmethod
    def default_principle_function(
        r: float, M: float = 1.0, r_s: float = 1.0, **_
    ) -> float:
        return (3 * M) / (4 * np.pi * r_s**3) * (1 + (r / r_s) ** 2) ** (-5 / 2)

    CLASS_FUNCTIONS = {"default": default_principle_function}


class DehnenDensityProfile(RadialDensityProfile):
    """
    Dehnen profile, a generalization of the Hernquist profile.

    Use cases:
    ----------
    - Useful for modeling galaxies and dark matter halos with customizable inner slope behavior.
    - Suitable for systems requiring more flexible parameterization than Hernquist.

    Attributes
    ----------
    M : float
        Total mass.
    r_s : float
        Scale radius.
    gamma : float
        Inner slope parameter.

    Examples
    --------
    .. code-block:: python

        dehnen = DehnenDensityProfile(M=1e10, r_s=2.0, gamma=1.5)
        r_values = np.linspace(0.1, 50, 500)
        density = dehnen(r_values)

    See Also
    --------
    HernquistDensityProfile : A related profile with a fixed inner slope.

    References
    ----------
    Dehnen, W. (1993). "A Family of Potential-Density Pairs for Spherical Galaxies and Bulges."
    Monthly Notices of the Royal Astronomical Society, 265(1), 250–256.
    """

    M = ProfileParameter(1.0, "Total mass [Msun]")
    r_s = ProfileParameter(1.0, "Scale radius [kpc]")
    gamma = ProfileParameter(1.0, "Inner slope parameter")

    mass = ProfileLinkDescriptor(
        module="cluster_generator.profiles.mass", profile_class="DehnenMassProfile"
    )

    @staticmethod
    def default_principle_function(
        r: float, M: float = 1.0, r_s: float = 1.0, gamma: float = 1.0, **_
    ) -> float:
        return (
            ((3 - gamma) * M)
            / (4 * np.pi * r_s**3)
            * (r / r_s) ** (-gamma)
            * (1 + r / r_s) ** (gamma - 4)
        )

    CLASS_FUNCTIONS = {"default": default_principle_function}


class JaffeDensityProfile(RadialDensityProfile):
    """
    Jaffe profile for modeling elliptical galaxies.

    Use cases:
    ----------
    - Commonly used in dynamical studies of elliptical galaxies.
    - It has a similar structure to the Hernquist profile but with a different outer slope.

    Attributes
    ----------
    rho_0 : float
        Characteristic density.
    r_s : float
        Scale radius.

    Examples
    --------
    .. code-block:: python

        jaffe = JaffeDensityProfile(rho_0=0.1, r_s=10.0)
        r_values = np.linspace(0.1, 100, 1000)
        density = jaffe(r_values)

    See Also
    --------
    HernquistDensityProfile : Another profile for elliptical galaxies, but with different asymptotic behavior.
    DehnenDensityProfile : A more flexible profile with an adjustable inner slope.

    References
    ----------
    Jaffe, W. (1983). "A simple model for the distribution of light in spherical galaxies."
    Monthly Notices of the Royal Astronomical Society, 202, 995–999.
    """

    rho_0 = ProfileParameter(1.0, "Characteristic density [Msun/kpc^3]")
    r_s = ProfileParameter(1.0, "Scale radius [kpc]")

    mass = ProfileLinkDescriptor(
        module="cluster_generator.profiles.mass", profile_class="JaffeMassProfile"
    )

    @staticmethod
    def default_principle_function(
        r: float, rho_0: float = 1.0, r_s: float = 1.0, **_
    ) -> float:
        return rho_0 / ((r / r_s) * (1 + r / r_s) ** 2)

    CLASS_FUNCTIONS = {"default": default_principle_function}


class KingDensityProfile(RadialDensityProfile):
    """
    King profile for modeling the density distribution in globular clusters.

    Use cases:
    ----------
    - Widely used to model globular clusters with a central core and a truncation radius.
    - Useful in observational studies where a truncation at large radii is observed.

    Attributes
    ----------
    rho_0 : float
        Central density.
    r_c : float
        Core radius.
    r_t : float
        Tidal radius.

    Examples
    --------
    .. code-block:: python

        king = KingDensityProfile(rho_0=0.5, r_c=2.0, r_t=20.0)
        r_values = np.linspace(0.1, 50, 500)
        density = king(r_values)

    See Also
    --------
    PlummerDensityProfile : Another profile commonly used for globular clusters but without a truncation radius.

    References
    ----------
    King, I. R. (1962). "The structure of star clusters. I. an empirical density law."
    The Astronomical Journal, 67, 471–485.
    """

    rho_0 = ProfileParameter(1.0, "Central density [Msun/kpc^3]")
    r_c = ProfileParameter(1.0, "Core radius [kpc]")
    r_t = ProfileParameter(1.0, "Tidal radius [kpc]")

    @staticmethod
    def default_principle_function(
        r: float, rho_0: float = 1.0, r_c: float = 1.0, r_t: float = 1.0, **_
    ) -> float:
        return rho_0 * (
            (1 + (r / r_c) ** 2) ** (-3 / 2) - (1 + (r_t / r_c) ** 2) ** (-3 / 2)
        )

    CLASS_FUNCTIONS = {"default": default_principle_function}


class BurkertDensityProfile(RadialDensityProfile):
    """
    Burkert profile for modeling the density of dark matter halos in dwarf galaxies.

    Use cases:
    ----------
    - Primarily used to model dark matter distribution in dwarf galaxies.
    - The profile is well-suited for cored dark matter halos, especially in smaller galaxies.

    Attributes
    ----------
    rho_0 : float
        Central density.
    r_s : float
        Scale radius.

    Examples
    --------
    .. code-block:: python

        burkert = BurkertDensityProfile(rho_0=0.1, r_s=3.0)
        r_values = np.linspace(0.1, 50, 500)
        density = burkert(r_values)

    See Also
    --------
    NFWDensityProfile : Often used for larger dark matter halos without the central core flattening seen in the Burkert profile.

    References
    ----------
    Burkert, A. (1995). "The structure of dark matter halos in dwarf galaxies."
    The Astrophysical Journal Letters, 447, L25–L28.
    """

    rho_0 = ProfileParameter(1.0, "Central density [Msun/kpc^3]")
    r_s = ProfileParameter(1.0, "Scale radius [kpc]")

    @staticmethod
    def default_principle_function(
        r: float, rho_0: float = 1.0, r_s: float = 1.0, **_
    ) -> float:
        return rho_0 / ((1 + r / r_s) * (1 + (r / r_s) ** 2))

    CLASS_FUNCTIONS = {"default": default_principle_function}


class MooreDensityProfile(RadialDensityProfile):
    """
    Moore profile for dark matter halos with a steeper inner slope compared to the NFW profile.

    Use cases:
    ----------
    - Primarily used in simulations of dark matter halos with a steeper central density profile.
    - Appropriate for dark matter-dominated systems that require a sharper inner profile.

    Attributes
    ----------
    rho_0 : float
        Central density.
    r_s : float
        Scale radius.

    Examples
    --------
    .. code-block:: python

        moore = MooreDensityProfile(rho_0=0.2, r_s=5.0)
        r_values = np.linspace(0.1, 50, 500)
        density = moore(r_values)

    See Also
    --------
    NFWDensityProfile : A related profile but with a shallower central slope.
    CoredNFWDensityProfile : A modification of the NFW profile with a central core.

    References
    ----------
    Moore, B., et al. (1999). "Dark Matter Substructure within Galactic Halos."
    The Astrophysical Journal, 524, L19–L22.
    """

    rho_0 = ProfileParameter(1.0, "Central density [Msun/kpc^3]")
    r_s = ProfileParameter(1.0, "Scale radius [kpc]")

    @staticmethod
    def default_principle_function(
        r: float, rho_0: float = 1.0, r_s: float = 1.0, **_
    ) -> float:
        return rho_0 / ((r / r_s) ** (3 / 2) * (1 + r / r_s) ** (3 / 2))

    mass = ProfileLinkDescriptor(
        module="cluster_generator.profiles.mass", profile_class="MooreMassProfile"
    )

    CLASS_FUNCTIONS = {"default": default_principle_function}


class CoredNFWDensityProfile(RadialDensityProfile):
    """
    Cored NFW profile, a modification of the NFW profile with a central core.

    Use cases:
    ----------
    - Suitable for dark matter halos with a flattened core, often used in dwarf galaxies or low-surface brightness galaxies.
    - Useful when modeling systems that require a suppression of the inner cusp.

    Attributes
    ----------
    rho_0 : float
        Central density.
    r_s : float
        Scale radius.

    Examples
    --------
    .. code-block:: python

        cored_nfw = CoredNFWDensityProfile(rho_0=0.1, r_s=10.0)
        r_values = np.linspace(0.1, 100, 1000)
        density = cored_nfw(r_values)

    See Also
    --------
    NFWDensityProfile : The original NFW profile without a central core.
    BurkertDensityProfile : Another core-modified profile used in dwarf galaxies.

    References
    ----------
    Navarro, J. F., et al. (2004). "The inner structure of LambdaCDM halos III: Universality and Asymptotic Slopes."
    Monthly Notices of the Royal Astronomical Society, 349, 1039–1051.
    """

    rho_0 = ProfileParameter(1.0, "Central density [Msun/kpc^3]")
    r_s = ProfileParameter(1.0, "Scale radius [kpc]")

    mass = ProfileLinkDescriptor(
        module="cluster_generator.profiles.mass", profile_class="CoredNFWMassProfile"
    )

    @staticmethod
    def default_principle_function(
        r: float, rho_0: float = 1.0, r_s: float = 1.0, **_
    ) -> float:
        return rho_0 / ((1 + (r / r_s) ** 2) * (1 + r / r_s) ** 2)

    CLASS_FUNCTIONS = {"default": default_principle_function}


class VikhlininDensityProfile(RadialDensityProfile):
    """
    Vikhlinin density profile for galaxy clusters.

    Use cases:
    ----------
    - Primarily used for modeling X-ray galaxy clusters.
    - Accounts for the complex structure of galaxy clusters with multiple transitions in density slopes.

    Attributes
    ----------
    rho_0 : float
        Central density.
    r_c : float
        Core radius.
    r_s : float
        Scale radius.
    alpha : float
        Inner slope parameter.
    beta : float
        Middle slope parameter.
    epsilon : float
        Outer slope parameter.
    gamma : float
        Width of the outer transition.

    Examples
    --------
    .. code-block:: python

        vikhlinin = VikhlininDensityProfile(rho_0=1e-2, r_c=50.0, r_s=500.0, alpha=0.8, beta=2.0, epsilon=1.0, gamma=3.0)
        r_values = np.linspace(0.1, 1000, 500)
        density = vikhlinin(r_values)

    See Also
    --------
    AM06DensityProfile : Another cluster density profile with a different parameterization for cluster cores.

    References
    ----------
    Vikhlinin, A., et al. (2006). "Chandra Sample of Nearby Relaxed Galaxy Clusters:
    Mass, Gas Fraction, and Mass-Temperature Relation." The Astrophysical Journal, 640, 691–709.
    """

    rho_0 = ProfileParameter(1.0, "Central density [Msun/kpc^3]")
    r_c = ProfileParameter(1.0, "Core radius [kpc]")
    r_s = ProfileParameter(1.0, "Scale radius [kpc]")
    alpha = ProfileParameter(1.0, "Inner slope parameter")
    beta = ProfileParameter(1.0, "Middle slope parameter")
    epsilon = ProfileParameter(1.0, "Outer slope parameter")
    gamma = ProfileParameter(3.0, "Width of the outer transition")

    @staticmethod
    def default_principle_function(
        r, rho_0=1.0, r_c=1.0, r_s=1.0, alpha=1.0, beta=1.0, epsilon=1.0, gamma=3.0, **_
    ):
        return (
            rho_0
            * (r / r_c) ** (-0.5 * alpha)
            * (1 + (r / r_c) ** 2) ** (-1.5 * beta + 0.25 * alpha)
            * (1 + (r / r_s) ** gamma) ** (-0.5 * epsilon / gamma)
        )

    CLASS_FUNCTIONS = {"default": default_principle_function}


class AM06DensityProfile(RadialDensityProfile):
    """
    Ascasibar & Markevitch (2006) density profile for galaxy clusters.

    Use cases:
    ----------
    - Often used in X-ray studies of galaxy clusters, focusing on the central core structure.
    - Designed to capture the steep slope and flattening in galaxy cluster cores.

    Attributes
    ----------
    rho_0 : float
        Central density.
    a : float
        Scale radius.
    a_c : float
        Core radius.
    c : float
        Core scaling factor.
    alpha : float
        Shape parameter.
    beta : float
        Shape parameter.

    Examples
    --------
    .. code-block:: python

        am06 = AM06DensityProfile(rho_0=1e-3, a=200.0, a_c=50.0, c=2.0, alpha=0.8, beta=2.0)
        r_values = np.linspace(1, 1000, 500)
        density = am06(r_values)

    See Also
    --------
    VikhlininDensityProfile : A similar profile with more detailed slope transitions for galaxy clusters.
    NFWDensityProfile : For simpler dark matter halos without a core structure.

    References
    ----------
    Ascasibar, Y., & Markevitch, M. (2006). "The nature of cold fronts in cluster mergers."
    The Astrophysical Journal, 650, 102–127.
    """

    rho_0 = ProfileParameter(1.0, "Central density [Msun/kpc^3]")
    a = ProfileParameter(1.0, "Scale radius [kpc]")
    a_c = ProfileParameter(1.0, "Core radius [kpc]")
    c = ProfileParameter(1.0, "Core scaling factor")
    alpha = ProfileParameter(1.0, "Shape parameter")
    beta = ProfileParameter(1.0, "Shape parameter")

    @staticmethod
    def default_principle_function(
        r, rho_0=1.0, a=1.0, a_c=1.0, c=1.0, alpha=1.0, beta=1.0, **_
    ):
        return (
            rho_0 * (1 + r / a_c) * (1 + r / (a_c * c)) ** alpha * (1 + r / a) ** beta
        )

    CLASS_FUNCTIONS = {"default": default_principle_function}


class SNFWDensityProfile(RadialDensityProfile):
    """
    Super-NFW (SNFW) density profile.

    Use cases:
    ----------
    - A variant of the Navarro-Frenk-White profile used for more steeply falling outer regions of dark matter halos.
    - Suitable for systems requiring a sharper decline than the standard NFW profile, especially at large radii.

    Attributes
    ----------
    M : float
        Total mass.
    a : float
        Scale radius.

    Examples
    --------
    .. code-block:: python

        snfw = SNFWDensityProfile(M=1e12, a=20.0)
        r_values = np.linspace(0.1, 200, 500)
        density = snfw(r_values)

    See Also
    --------
    TNFWDensityProfile : A truncated version of the NFW profile with a similar steepening behavior.
    NFWDensityProfile : The base NFW profile, useful for larger halos without the steeper decline.

    References
    ----------
    Lilley, E. J., et al. (2018). "The growth of galaxies in the Illustris simulation: angular momentum content,
    gas fractions, and star formation efficiency." Monthly Notices of the Royal Astronomical Society, 479, 4126.
    """

    M = ProfileParameter(1.0, "Total mass [Msun]")
    a = ProfileParameter(1.0, "Scale radius [kpc]")

    @staticmethod
    def default_principle_function(r, M=1.0, a=1.0, **_):
        return 3.0 * M / (16.0 * np.pi * a**3) / ((r / a) * (1.0 + r / a) ** 2.5)

    CLASS_FUNCTIONS = {"default": default_principle_function}


class TNFWDensityProfile(RadialDensityProfile):
    """
    Truncated NFW (tNFW) density profile.

    Use cases:
    ----------
    - Used for dark matter halos with a truncation radius, often in tidal-stripped or isolated environments.
    - Ideal for modeling systems where the outer density profile declines faster than a standard NFW halo.

    Attributes
    ----------
    rho_0 : float
        Scale density.
    r_s : float
        Scale radius.
    r_t : float
        Truncation radius.

    Examples
    --------
    .. code-block:: python

        tnfw = TNFWDensityProfile(rho_s=1e-2, r_s=10.0, r_t=100.0)
        r_values = np.linspace(1, 200, 500)
        density = tnfw(r_values)

    See Also
    --------
    SNFWDensityProfile : A super NFW profile that models steep declines in the density without a truncation radius.
    NFWDensityProfile : A non-truncated version of the NFW profile for halos without such sharp outer declines.

    References
    ----------
    Baltz, E. A., Marshall, P., & Oguri, M. (2009). "Precision cosmology and dark energy from strong lens time delays."
    Journal of Cosmology and Astroparticle Physics, 2009, 015.
    """

    rho_0 = ProfileParameter(1.0, "Scale density [Msun/kpc^3]")
    r_s = ProfileParameter(1.0, "Scale radius [kpc]")
    r_t = ProfileParameter(1.0, "Truncation radius [kpc]")

    @staticmethod
    def default_principle_function(r, rho_0=1.0, r_s=1.0, r_t=1.0, **_):
        return rho_0 / ((r / r_s) * (1 + r / r_s) ** 2) / (1 + (r / r_t) ** 2)

    CLASS_FUNCTIONS = {"default": default_principle_function}
