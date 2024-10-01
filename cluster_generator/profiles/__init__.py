"""
Cluster Generator Profiles Module
=================================

This module provides a collection of radial density, temperature, entropy, and generic profiles
used in astrophysical simulations and analysis. These profiles are widely applicable to describe
the mass distribution of dark matter halos, galaxies, star clusters, and galaxy clusters.

Each profile is implemented using a flexible system, allowing for dynamic geometry handling
and parameterization. The profiles are designed to be easily rescaled and converted between
different physical representations (e.g., from concentration parameters to characteristic densities).

Quick Links
-----------
- **Radial Density Profiles**:
    - `NFWDensityProfile`: Navarro-Frenk-White (NFW) profile, commonly used for dark matter halos.
    - `HernquistDensityProfile`: Hernquist profile, frequently applied to elliptical galaxies and stellar bulges.
    - `EinastoDensityProfile`: Einasto profile, useful for describing dark matter halo mass distributions.
    - `SingularIsothermalDensityProfile`: Singular isothermal model for large-scale equilibrium systems.
    - `CoredIsothermalDensityProfile`: Isothermal model with a core for small-scale systems like globular clusters.
    - `PlummerDensityProfile`: Plummer profile for star clusters and bulges, avoiding central singularities.
    - `DehnenDensityProfile`: Dehnen profile, a flexible generalization of the Hernquist profile.
    - `JaffeDensityProfile`: Jaffe profile, used to model elliptical galaxies.
    - `KingDensityProfile`: King profile, commonly used for globular clusters with a truncation radius.
    - `BurkertDensityProfile`: Burkert profile, used for cored dark matter halos in dwarf galaxies.
    - `MooreDensityProfile`: Moore profile, providing a steeper inner slope than the NFW profile.
    - `CoredNFWDensityProfile`: NFW profile modified with a central core.
    - `VikhlininDensityProfile`: Vikhlinin profile, used for modeling galaxy clusters in X-ray studies.
    - `AM06DensityProfile`: Ascasibar & Markevitch profile, designed for galaxy clusters.
    - `SNFWDensityProfile`: Super-NFW profile for steep outer declines in dark matter halos.
    - `TNFWDensityProfile`: Truncated NFW profile for halos with a truncation radius.

- **Temperature Profiles**:
    - `VikhlininTemperatureProfile`: Vikhlinin temperature profile for galaxy clusters.
    - `AM06TemperatureProfile`: Temperature profile from Ascasibar & Markevitch (2006).
    - `UniversalPressureTemperatureProfile`: Universal pressure profile-based temperature model.
    - `IsothermalTemperatureProfile`: Simple isothermal temperature profile.
    - `CoolingFlowTemperatureProfile`: Profile describing cooling flows in galaxy clusters.
    - `DoubleBetaTemperatureProfile`: Temperature model with two beta components.
    - `BetaModelTemperatureProfile`: Standard beta-model for temperature distribution in clusters.

- **Entropy Profiles**:
    - `BaselineEntropyProfile`: Voit et al. (2005) baseline entropy profile for galaxy clusters.
    - `BrokenEntropyProfile`: A broken power-law model for entropy in clusters.
    - `WalkerEntropyProfile`: Entropy profile used to describe galaxy clusters.

- **General Profiles**:
    - `ConstantProfile`: Generic constant profile for 1D, 2D, or 3D systems.
    - `RadialPowerLawProfile`: Radial power-law profile for systems with power-law behavior.

Conversion Methods
------------------
Many profiles support conversion between different parameterizations, such as using concentration
parameters or virial mass instead of characteristic densities. These conversions are implemented
through methods within each profile class.

Usage Examples
--------------
All profile classes follow a similar pattern for instantiation and evaluation:

.. code-block:: python

    from cluster_generator.profiles.density import NFWDensityProfile

    # Create a Navarro-Frenk-White density profile
    nfw = NFWDensityProfile(rho_0=0.1, r_s=10.0)

    # Evaluate the profile at specific radial distances
    r_values = np.linspace(0.1, 100, 1000)
    density = nfw(r_values)

    # Rescale the profile by mass within a given radius
    rescaled_nfw = nfw.rescale_by_mass(radius=100, target_mass=1e14)

This module allows for flexible profile creation, modification, and evaluation, providing the tools needed to model a wide range of astrophysical systems with radial symmetry.

"""

# Import necessary classes and functions into the module's namespace
from .density import (
    AM06DensityProfile,
    BurkertDensityProfile,
    CoredIsothermalDensityProfile,
    CoredNFWDensityProfile,
    DehnenDensityProfile,
    EinastoDensityProfile,
    HernquistDensityProfile,
    JaffeDensityProfile,
    KingDensityProfile,
    MooreDensityProfile,
    NFWDensityProfile,
    PlummerDensityProfile,
    SingularIsothermalDensityProfile,
    SNFWDensityProfile,
    TNFWDensityProfile,
    VikhlininDensityProfile,
)
from .general import ConstantProfile, RadialPowerLawProfile
from .temperature import (
    AM06TemperatureProfile,
    BaselineEntropyProfile,
    BetaModelTemperatureProfile,
    BrokenEntropyProfile,
    CoolingFlowTemperatureProfile,
    DoubleBetaTemperatureProfile,
    IsothermalTemperatureProfile,
    UniversalPressureTemperatureProfile,
    VikhlininTemperatureProfile,
    WalkerEntropyProfile,
)

# Add any other necessary imports or utility functions here
