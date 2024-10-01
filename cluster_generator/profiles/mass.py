"""
Radial Mass Profiles Module
===========================

This module contains various radial mass profiles used to describe the mass distribution
of dark matter halos, galaxies, and star clusters. Each profile assumes radial symmetry and
is designed to fit astrophysical systems such as galaxy clusters, elliptical galaxies, and
dark matter halos. These profiles are widely used in modeling the distribution of matter
in astrophysical simulations and observational studies.

References:
-----------
- [1] Navarro, J. F., Frenk, C. S., & White, S. D. M. (1997). A Universal Density Profile from Hierarchical Clustering.
      The Astrophysical Journal, 490(2), 493–508. https://doi.org/10.1086/304888
- [2] Hernquist, L. (1990). "An Analytical Model for Spherical Galaxies and Bulges." The Astrophysical Journal, 356, 359–364.
- [3] Isothermal models in galactic dynamics, Binney & Tremaine, 2008, "Galactic Dynamics".
- [4] Plummer, H. C. (1911). "On the problem of distribution in globular star clusters."
      Monthly Notices of the Royal Astronomical Society, 71, 460-470.
"""
from abc import ABC

import numpy as np

from cluster_generator.profiles._abc import ProfileParameter, RadialProfile
from cluster_generator.profiles._types import ProfileLinkDescriptor


def _obl_correction(eccentricity):
    """
    Oblate correction factor for adjusting spherical profiles to account for oblateness.

    Parameters
    ----------
    eccentricity : float
        Eccentricity of the object (eccentricity of 0 corresponds to a sphere).

    Returns
    -------
    float
        Correction factor for oblateness.
    """
    return np.sqrt(1 - eccentricity**2)


def _prl_correction(eccentricity):
    """
    Prolate correction factor for adjusting spherical profiles to account for prolateness.

    Parameters
    ----------
    eccentricity : float
        Eccentricity of the object (eccentricity of 0 corresponds to a sphere).

    Returns
    -------
    float
        Correction factor for prolateness.
    """
    return 1 - eccentricity**2


def _tri_correction(eccentricity1, eccentricity2):
    """
    Triaxial correction factor for adjusting spherical profiles to account for triaxial shapes.

    Parameters
    ----------
    eccentricity1 : float
        Eccentricity along the first axis.
    eccentricity2 : float
        Eccentricity along the second axis.

    Returns
    -------
    float
        Correction factor for triaxiality.
    """
    return np.sqrt(1 - eccentricity2**2) * np.sqrt(1 - eccentricity1**2)


class RadialMassProfile(RadialProfile, ABC):
    """
    Base class for radial mass profiles. This class defines the structure that all specific
    mass profiles inherit from.

    Notes
    -----
    This abstract class should not be instantiated directly. Instead, use one of the subclasses
    like :py:class:`NFWMassProfile`, :py:class:`HernquistMassProfile`, etc.
    """

    pass


def _nfw_mass_kernel(r, rho_0, r_s):
    """
    Kernel function for the Navarro-Frenk-White mass profile.

    Parameters
    ----------
    r : float
        Radial distance.
    rho_0 : float
        Characteristic density.
    r_s : float
        Scale radius.

    Returns
    -------
    float
        The mass at radius `r` for the NFW profile.
    """
    return (4 * np.pi * rho_0 * r_s**3) * (np.log((r_s + r) / r_s) - (r / (r_s + r)))


class NFWMassProfile(RadialMassProfile):
    """
    Navarro-Frenk-White (NFW) mass profile for modeling the mass distribution in dark matter halos [1]_.

    Equation:
    ---------
    .. math:: M(r) = 4 \pi \rho_0 r_s^3 \left( \log \left( \frac{r_s + r}{r_s} \right) - \frac{r}{r_s + r} \right)

    Attributes
    ----------
    rho_0 : float
        Characteristic density [Msun/kpc^3].
    r_s : float
        Scale radius [kpc].

    See Also
    --------
    :py:class:`NFWDensityProfile` : Corresponding density profile for the NFW mass profile.

    Examples
    --------
    .. code-block:: python

        nfw = NFWMassProfile(rho_0=1.0, r_s=10.0)
        r_values = np.linspace(0.1, 100, 1000)
        mass = nfw(r_values)

    References
    ----------
    .. [1] Navarro, J. F., Frenk, C. S., & White, S. D. M. (1997). A Universal Density Profile from Hierarchical Clustering.
           The Astrophysical Journal, 490(2), 493–508.
    """

    rho_0 = ProfileParameter(1.0, "Characteristic density [Msun/kpc^3]")
    r_s = ProfileParameter(1.0, "Scale radius [kpc]")

    CLASS_FUNCTIONS = {
        "Spherical": _nfw_mass_kernel,
        "Oblate": lambda r, rho_0=1, r_s=1, ecc=None: _nfw_mass_kernel(r, rho_0, r_s)
        * _obl_correction(ecc),
        "Prolate": lambda r, rho_0=1, r_s=1, ecc=None: _nfw_mass_kernel(r, rho_0, r_s)
        * _prl_correction(ecc),
        "Triaxial": lambda r, rho_0=1, r_s=1, ecc1=None, ecc2=None: _nfw_mass_kernel(
            r, rho_0, r_s
        )
        * _tri_correction(ecc1, ecc2),
        "default": _nfw_mass_kernel,
    }

    density = ProfileLinkDescriptor(module="density", profile_class="NFWDensityProfile")


def _hernquist_mass_kernel(r, rho_0, r_s):
    """
    Kernel function for the Hernquist mass profile.

    Parameters
    ----------
    r : float
        Radial distance.
    rho_0 : float
        Characteristic density.
    r_s : float
        Scale radius.

    Returns
    -------
    float
        The mass at radius `r` for the Hernquist profile.
    """
    return 2 * np.pi * rho_0 * r_s * r**2 / (r / r_s + 1) ** 2


class HernquistMassProfile(RadialMassProfile):
    """
    Hernquist mass profile for modeling elliptical galaxies and bulges [2]_.

    Equation:
    ---------
    .. math:: M(r) = 4 \pi \rho_0 r_s^2 \frac{r^2}{(r + r_s)^2}

    Attributes
    ----------
    rho_0 : float
        Characteristic density [Msun/kpc^3].
    r_s : float
        Scale radius [kpc].

    See Also
    --------
    :py:class:`HernquistDensityProfile` : Corresponding density profile for the Hernquist mass profile.

    Examples
    --------
    .. code-block:: python

        hernquist = HernquistMassProfile(rho_0=0.5, r_s=5.0)
        r_values = np.linspace(0.1, 50, 500)
        mass = hernquist(r_values)

    References
    ----------
    .. [2] Hernquist, L. (1990). "An Analytical Model for Spherical Galaxies and Bulges." The Astrophysical Journal, 356, 359–364.
    """

    rho_0 = ProfileParameter(1.0, "Characteristic density [Msun/kpc^3]")
    r_s = ProfileParameter(1.0, "Scale radius [kpc]")

    CLASS_FUNCTIONS = {
        "Spherical": _hernquist_mass_kernel,
        "Oblate": lambda r, rho_0=1, r_s=1, ecc=None: _hernquist_mass_kernel(
            r, rho_0, r_s
        )
        * _obl_correction(ecc),
        "Prolate": lambda r, rho_0=1, r_s=1, ecc=None: _hernquist_mass_kernel(
            r, rho_0, r_s
        )
        * _prl_correction(ecc),
        "Triaxial": lambda r, rho_0=1, r_s=1, ecc1=None, ecc2=None: _hernquist_mass_kernel(
            r, rho_0, r_s
        )
        * _tri_correction(ecc1, ecc2),
        "default": _hernquist_mass_kernel,
    }

    density = ProfileLinkDescriptor(
        module="density", profile_class="HernquistDensityProfile"
    )


def _isothermal_mass_kernel(r, rho_0, r_c):
    """
    Kernel function for the Isothermal mass profile.

    Parameters
    ----------
    r : float
        Radial distance.
    rho_0 : float
        Characteristic density.
    r_c : float
        Core radius (optional).

    Returns
    -------
    float
        The mass at radius `r` for the Isothermal profile.
    """
    if r_c is not None:
        return 4 * np.pi * rho_0 * r_c**2 * (r - r_c * np.arctan(r / r_c))
    else:
        return 4 * np.pi * rho_0 * r


class IsothermalMassProfile(RadialMassProfile):
    """
    Isothermal mass profile for self-gravitating systems in equilibrium [3]_.

    Equation:
    ---------
    .. math:: M(r) = 4 \pi \rho_0 r_c^2 (r - r_c \arctan(r / r_c))

    Attributes
    ----------
    rho_0 : float
        Characteristic density [Msun/kpc^3].
    r_c : float
        Core radius [kpc].

    See Also
    --------
    :py:class:`IsothermalDensityProfile` : Corresponding density profile for the Isothermal mass profile.

    Examples
    --------
    .. code-block:: python

        isothermal = IsothermalMassProfile(rho_0=0.1, r_c=2.0)
        r_values = np.linspace(0.1, 50, 500)
        mass = isothermal(r_values)

    References
    ----------
    .. [3] Binney, J., & Tremaine, S. (2008). "Galactic Dynamics." Princeton University Press.
    """

    rho_0 = ProfileParameter(1.0, "Characteristic density [Msun/kpc^3]")
    r_c = ProfileParameter(1.0, "Core radius [kpc]")

    CLASS_FUNCTIONS = {
        "Spherical": _isothermal_mass_kernel,
        "Oblate": lambda r, rho_0=1, r_c=1, ecc=None: _isothermal_mass_kernel(
            r, rho_0, r_c
        )
        * _obl_correction(ecc),
        "Prolate": lambda r, rho_0=1, r_c=1, ecc=None: _isothermal_mass_kernel(
            r, rho_0, r_c
        )
        * _prl_correction(ecc),
        "Triaxial": lambda r, rho_0=1, r_c=1, ecc1=None, ecc2=None: _isothermal_mass_kernel(
            r, rho_0, r_c
        )
        * _tri_correction(ecc1, ecc2),
        "default": _isothermal_mass_kernel,
    }

    density = ProfileLinkDescriptor(
        module="density", profile_class="IsothermalDensityProfile"
    )


def _plummer_mass_kernel(r, M, r_s):
    """
    Kernel function for the Plummer mass profile.

    Parameters
    ----------
    r : float
        Radial distance.
    M : float
        Total mass.
    r_s : float
        Plummer radius.

    Returns
    -------
    float
        The mass at radius `r` for the Plummer profile.
    """
    return M * r**3 / (r_s**3 * (1 + (r / r_s) ** 2) ** (3 / 2))


class PlummerMassProfile(RadialMassProfile):
    """
    Plummer mass profile for modeling star clusters and stellar bulges [4]_.

    Equation:
    ---------
    .. math:: M(r) = \frac{M r^3}{r_s^3 \left( 1 + \left( \frac{r}{r_s} \right)^2 \right)^{3/2}}

    Attributes
    ----------
    M : float
        Total mass [Msun].
    r_s : float
        Plummer radius [kpc].

    See Also
    --------
    :py:class:`PlummerDensityProfile` : Corresponding density profile for the Plummer mass profile.

    Examples
    --------
    .. code-block:: python

        plummer = PlummerMassProfile(M=1e5, r_s=1.0)
        r_values = np.linspace(0.1, 10, 100)
        mass = plummer(r_values)

    References
    ----------
    .. [4] Plummer, H. C. (1911). "On the problem of distribution in globular star clusters."
           Monthly Notices of the Royal Astronomical Society, 71, 460-470.
    """

    M = ProfileParameter(1.0, "Total mass [Msun]")
    r_s = ProfileParameter(1.0, "Plummer radius [kpc]")

    CLASS_FUNCTIONS = {
        "Spherical": _plummer_mass_kernel,
        "Oblate": lambda r, M=1, r_s=1, ecc=None: _plummer_mass_kernel(r, M, r_s)
        * _obl_correction(ecc),
        "Prolate": lambda r, M=1, r_s=1, ecc=None: _plummer_mass_kernel(r, M, r_s)
        * _prl_correction(ecc),
        "Triaxial": lambda r, M=1, r_s=1, ecc1=None, ecc2=None: _plummer_mass_kernel(
            r, M, r_s
        )
        * _tri_correction(ecc1, ecc2),
        "default": _plummer_mass_kernel,
    }

    density = ProfileLinkDescriptor(
        module="density", profile_class="PlummerDensityProfile"
    )


def _dehnen_mass_kernel(r, M, r_s, gamma):
    """
    Kernel function for the Dehnen mass profile.

    Parameters
    ----------
    r : float
        Radial distance.
    M : float
        Total mass.
    r_s : float
        Scale radius.
    gamma : float
        Inner slope parameter.

    Returns
    -------
    float
        The mass at radius `r` for the Dehnen profile.
    """
    return M * (r / (r + r_s)) ** (3 - gamma)


class DehnenMassProfile(RadialMassProfile):
    """
    Dehnen mass profile, a generalization of the Hernquist profile with adjustable inner slope [5]_.

    Equation:
    ---------
    .. math:: M(r) = M \left( \frac{r}{r + r_s} \right)^{3 - \gamma}

    Attributes
    ----------
    M : float
        Total mass [Msun].
    r_s : float
        Scale radius [kpc].
    gamma : float
        Inner slope parameter.

    See Also
    --------
    :py:class:`DehnenDensityProfile` : Corresponding density profile for the Dehnen mass profile.

    Examples
    --------
    .. code-block:: python

        dehnen = DehnenMassProfile(M=1e10, r_s=2.0, gamma=1.5)
        r_values = np.linspace(0.1, 50, 500)
        mass = dehnen(r_values)

    References
    ----------
    .. [5] Dehnen, W. (1993). "A Family of Potential-Density Pairs for Spherical Galaxies and Bulges."
           Monthly Notices of the Royal Astronomical Society, 265(1), 250–256.
    """

    M = ProfileParameter(1.0, "Total mass [Msun]")
    r_s = ProfileParameter(1.0, "Scale radius [kpc]")
    gamma = ProfileParameter(1.0, "Inner slope parameter")

    CLASS_FUNCTIONS = {
        "Spherical": _dehnen_mass_kernel,
        "Oblate": lambda r, M=1, r_s=1, gamma=1, ecc=None: _dehnen_mass_kernel(
            r, M, r_s, gamma
        )
        * _obl_correction(ecc),
        "Prolate": lambda r, M=1, r_s=1, gamma=1, ecc=None: _dehnen_mass_kernel(
            r, M, r_s, gamma
        )
        * _prl_correction(ecc),
        "Triaxial": lambda r, M=1, r_s=1, gamma=1, ecc1=None, ecc2=None: _dehnen_mass_kernel(
            r, M, r_s, gamma
        )
        * _tri_correction(ecc1, ecc2),
        "default": _dehnen_mass_kernel,
    }

    density = ProfileLinkDescriptor(
        module="density", profile_class="DehnenDensityProfile"
    )


def _jaffe_mass_kernel(r, rho_0, r_s):
    """
    Kernel function for the Jaffe mass profile.

    Parameters
    ----------
    r : float
        Radial distance.
    rho_0 : float
        Characteristic density.
    r_s : float
        Scale radius.

    Returns
    -------
    float
        The mass at radius `r` for the Jaffe profile.
    """
    return 4 * np.pi * rho_0 * r_s**2 * (r / (r + r_s))


class JaffeMassProfile(RadialMassProfile):
    """
    Jaffe mass profile for modeling elliptical galaxies [6]_.

    Equation:
    ---------
    .. math:: M(r) = 4 \pi \rho_0 r_s^2 \frac{r}{r + r_s}

    Attributes
    ----------
    rho_0 : float
        Characteristic density [Msun/kpc^3].
    r_s : float
        Scale radius [kpc].

    See Also
    --------
    :py:class:`JaffeDensityProfile` : Corresponding density profile for the Jaffe mass profile.

    Examples
    --------
    .. code-block:: python

        jaffe = JaffeMassProfile(rho_0=0.1, r_s=10.0)
        r_values = np.linspace(0.1, 100, 1000)
        mass = jaffe(r_values)

    References
    ----------
    .. [6] Jaffe, W. (1983). "A simple model for the distribution of light in spherical galaxies."
           Monthly Notices of the Royal Astronomical Society, 202, 995–999.
    """

    rho_0 = ProfileParameter(1.0, "Characteristic density [Msun/kpc^3]")
    r_s = ProfileParameter(1.0, "Scale radius [kpc]")

    CLASS_FUNCTIONS = {
        "Spherical": _jaffe_mass_kernel,
        "Oblate": lambda r, rho_0=1, r_s=1, ecc=None: _jaffe_mass_kernel(r, rho_0, r_s)
        * _obl_correction(ecc),
        "Prolate": lambda r, rho_0=1, r_s=1, ecc=None: _jaffe_mass_kernel(r, rho_0, r_s)
        * _prl_correction(ecc),
        "Triaxial": lambda r, rho_0=1, r_s=1, ecc1=None, ecc2=None: _jaffe_mass_kernel(
            r, rho_0, r_s
        )
        * _tri_correction(ecc1, ecc2),
        "default": _jaffe_mass_kernel,
    }

    density = ProfileLinkDescriptor(
        module="density", profile_class="JaffeDensityProfile"
    )


def _moore_mass_kernel(r, rho_0, r_s):
    """
    Kernel function for the Moore mass profile.

    Parameters
    ----------
    r : float
        Radial distance.
    rho_0 : float
        Central density.
    r_s : float
        Scale radius.

    Returns
    -------
    float
        The mass at radius `r` for the Moore profile.
    """
    return 4 * np.pi * rho_0 * r_s**3 * (np.log((r + r_s) / r_s) - r / (r + r_s))


class MooreMassProfile(RadialMassProfile):
    """
    Moore mass profile for modeling dark matter halos with a steep inner slope [7]_.

    Equation:
    ---------
    .. math:: M(r) = 4 \pi \rho_0 r_s^3 \left( \log \left( \frac{r + r_s}{r_s} \right) - \frac{r}{r + r_s} \right)

    Attributes
    ----------
    rho_0 : float
        Central density [Msun/kpc^3].
    r_s : float
        Scale radius [kpc].

    See Also
    --------
    :py:class:`MooreDensityProfile` : Corresponding density profile for the Moore mass profile.

    Examples
    --------
    .. code-block:: python

        moore = MooreMassProfile(rho_0=0.2, r_s=5.0)
        r_values = np.linspace(0.1, 50, 500)
        mass = moore(r_values)

    References
    ----------
    .. [7] Moore, B., et al. (1999). "Dark Matter Substructure within Galactic Halos."
           The Astrophysical Journal, 524, L19–L22.
    """

    rho_0 = ProfileParameter(1.0, "Central density [Msun/kpc^3]")
    r_s = ProfileParameter(1.0, "Scale radius [kpc]")

    CLASS_FUNCTIONS = {
        "Spherical": _moore_mass_kernel,
        "Oblate": lambda r, rho_0=1, r_s=1, ecc=None: _moore_mass_kernel(r, rho_0, r_s)
        * _obl_correction(ecc),
        "Prolate": lambda r, rho_0=1, r_s=1, ecc=None: _moore_mass_kernel(r, rho_0, r_s)
        * _prl_correction(ecc),
        "Triaxial": lambda r, rho_0=1, r_s=1, ecc1=None, ecc2=None: _moore_mass_kernel(
            r, rho_0, r_s
        )
        * _tri_correction(ecc1, ecc2),
        "default": _moore_mass_kernel,
    }

    density = ProfileLinkDescriptor(
        module="density", profile_class="MooreDensityProfile"
    )


def _cored_nfw_mass_kernel(r, rho_0, r_s):
    """
    Kernel function for the Cored NFW mass profile.

    Parameters
    ----------
    r : float
        Radial distance.
    rho_0 : float
        Central density.
    r_s : float
        Scale radius.

    Returns
    -------
    float
        The mass at radius `r` for the Cored NFW profile.
    """
    return 4 * np.pi * rho_0 * r_s**3 * (np.log((r + r_s) / r_s) - r / (r + r_s))


class CoredNFWMassProfile(RadialMassProfile):
    """
    Cored Navarro-Frenk-White (NFW) mass profile, a modification of the NFW profile with a central core [1]_.

    This profile assumes the same asymptotic behavior as the NFW profile at large radii, but introduces
    a core that flattens the central cusp, making it more appropriate for systems where the inner slope
    is not as steep.

    Equation:
    ---------
    .. math:: M(r) = 4 \pi \rho_0 r_s^3 \left( \log \left( \frac{r + r_s}{r_s} \right) - \frac{r}{r + r_s} \right)

    Attributes
    ----------
    rho_0 : float
        Central density [Msun/kpc^3].
    r_s : float
        Scale radius [kpc].

    See Also
    --------
    :py:class:`NFWMassProfile` : The original NFW mass profile without a central core.
    :py:class:`CoredNFWDensityProfile` : Corresponding density profile for the Cored NFW mass profile.

    Examples
    --------
    .. code-block:: python

        cored_nfw = CoredNFWMassProfile(rho_0=0.1, r_s=10.0)
        r_values = np.linspace(0.1, 100, 1000)
        mass = cored_nfw(r_values)

    References
    ----------
    .. [1] Navarro, J. F., et al. (2004). "The inner structure of LambdaCDM halos III: Universality and Asymptotic Slopes."
           Monthly Notices of the Royal Astronomical Society, 349, 1039–1051.
    """

    rho_0 = ProfileParameter(1.0, "Central density [Msun/kpc^3]")
    r_s = ProfileParameter(1.0, "Scale radius [kpc]")

    CLASS_FUNCTIONS = {
        "Spherical": _cored_nfw_mass_kernel,
        "Oblate": lambda r, rho_0=1, r_s=1, ecc=None: _cored_nfw_mass_kernel(
            r, rho_0, r_s
        )
        * _obl_correction(ecc),
        "Prolate": lambda r, rho_0=1, r_s=1, ecc=None: _cored_nfw_mass_kernel(
            r, rho_0, r_s
        )
        * _prl_correction(ecc),
        "Triaxial": lambda r, rho_0=1, r_s=1, ecc1=None, ecc2=None: _cored_nfw_mass_kernel(
            r, rho_0, r_s
        )
        * _tri_correction(ecc1, ecc2),
        "default": _cored_nfw_mass_kernel,
    }

    density = ProfileLinkDescriptor(
        module="density", profile_class="CoredNFWDensityProfile"
    )
