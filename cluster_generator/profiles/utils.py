from typing import TYPE_CHECKING

import numpy as np
from unyt import unyt_quantity

if TYPE_CHECKING:
    from yt.utilities.cosmology import Cosmology


def find_overdensity_radius(
    mass: float | unyt_quantity, delta: float, z: float = 0.0, cosmo: "Cosmology" = None
):
    """
    Calculate the radius corresponding to a given mass and overdensity.

    This function determines the radius at which the enclosed mass equals
    a given mass for a specified overdensity, such as the commonly used
    virial overdensities (e.g., 200 times the critical density of the universe).

    Parameters
    ----------
    mass : float
        The total enclosed mass (in Msun) within the desired radius.
    delta : float
        The overdensity factor, typically 200 for r200 (200 times the critical density),
        or any other overdensity factor for which the radius is to be computed.
    z : float, optional
        The redshift at which the halo formation occurs. Default is 0.0 (present day).
    cosmo : yt ``Cosmology`` object, optional
        A cosmology object that provides methods to calculate the critical density
        and other cosmological parameters. If not provided, a default cosmology from
        yt will be used.

    Returns
    -------
    radius : float
        The radius (in kpc) corresponding to the given enclosed mass and overdensity.

    Notes
    -----
    The radius is computed using the following formula:

    .. math::

        r_{\Delta} = \left( \frac{3 M}{4 \pi \Delta \rho_{\text{crit}}(z)} \right)^{1/3}

    where:
    - :math:`M` is the enclosed mass.
    - :math:`\Delta` is the overdensity factor.
    - :math:`\rho_{\text{crit}}(z)` is the critical density of the universe at redshift :math:`z`.

    """
    from yt.utilities.cosmology import Cosmology

    # Perform the required unit coercions to make sure the output
    # is in kpc.
    if not hasattr(mass, "units"):
        mass = unyt_quantity(mass, "Msun")

    # If no cosmology is provided, use the default cosmology
    if cosmo is None:
        cosmo = Cosmology()

    # Calculate the critical density at the given redshift
    rho_crit = cosmo.critical_density(z).to_value("Msun/kpc**3")

    # Compute the radius corresponding to the mass and overdensity
    radius = (3.0 * mass / (4.0 * np.pi * delta * rho_crit)) ** (1.0 / 3.0)

    return radius


def find_radius_mass(m_r, delta, z=0.0, cosmo=None):
    """
    Given a mass profile and an overdensity, find the radius
    and mass (e.g. M200, r200)

    Parameters
    ----------
    m_r : RadialProfile
        The mass profile.
    delta : float
        The overdensity to compute the mass and radius for.
    z : float, optional
        The redshift of the halo formation. Default: 0.0
    cosmo : yt ``Cosmology`` object
        The cosmology to be used when computing the critical
        density. If not supplied, a default one from yt will
        be used.
    """
    from scipy.optimize import bisect
    from yt.utilities.cosmology import Cosmology

    if cosmo is None:
        cosmo = Cosmology()
    rho_crit = cosmo.critical_density(z).to_value("Msun/kpc**3")
    f = lambda r: 3.0 * m_r(r) / (4.0 * np.pi * r**3) - delta * rho_crit
    r_delta = bisect(f, 0.01, 10000.0)
    return r_delta, m_r(r_delta)
