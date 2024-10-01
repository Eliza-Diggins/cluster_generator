"""
Temperature and Entropy Profiles for Galaxy Clusters
====================================================

This module contains classes for various temperature and entropy profiles
used in modeling galaxy clusters. These profiles are commonly employed in
astrophysical contexts, particularly for studying the thermodynamic properties
of clusters and the distribution of temperature, entropy, and pressure.

The profiles included in this module range from standard models such as
the isothermal and beta models to more complex profiles like the Vikhlinin
and polytropic models. Each profile is implemented as a callable class,
allowing for flexible use and extension.


References
----------
- Vikhlinin, A., Kravtsov, A., Forman, W., et al. 2006, ApJ, 640, 691
- Ascasibar, Y., & Markevitch, M. 2006, ApJ, 650, 102
- Voit, G.M., Kay, S.T., & Bryan, G.L. 2005, MNRAS, 364, 909
"""

import numpy as np

from cluster_generator.profiles._abc import RadialProfile
from cluster_generator.profiles._types import ProfileParameter


class RadialTemperatureProfile(RadialProfile):
    """Base class for all radial temperature profiles."""

    pass


class RadialEntropyProfile(RadialProfile):
    """Base class for all radial entropy profiles."""

    pass


class VikhlininTemperatureProfile(RadialTemperatureProfile):
    """
    Vikhlinin temperature profile for galaxy clusters.

    This profile is derived from Vikhlinin et al. (2006) and models
    the temperature structure of galaxy clusters.

    Parameters
    ----------
    T_0 : float
        The scale temperature of the profile in keV.
    a : float
        The inner logarithmic slope.
    b : float
        The width of the transition region.
    c : float
        The outer logarithmic slope.
    r_t : float
        The scale radius in kpc.
    T_min : float
        The minimum temperature in keV.
    r_cool : float
        The cooling radius in kpc.
    a_cool : float
        The logarithmic slope in the cooling region.

    Equation
    --------
    .. math::

        T(r) = T_0 \left( \frac{r}{r_t} \right)^{-a} \left(1 + \left(\frac{r}{r_t}\right)^b \right)^{-\frac{c}{b}}
               \times \frac{x + T_\mathrm{min}/T_0}{x + 1}
        where :math:`x = \left( \frac{r}{r_\mathrm{cool}} \right)^{a_\mathrm{cool}}`

    References
    ----------
    Vikhlinin, A., Kravtsov, A., Forman, W., et al. 2006, ApJ, 640, 691

    Examples
    --------
    .. code-block:: python

        profile = VikhlininTemperatureProfile(T_0=5, a=0.1, b=0.5, c=1.2, r_t=200, T_min=2, r_cool=10, a_cool=0.2)
        temperature = profile(50)  # Evaluate at r = 50 kpc
    """

    T_0 = ProfileParameter(5.0, "Scale temperature [keV]")
    a = ProfileParameter(0.1, "Inner slope")
    b = ProfileParameter(0.5, "Width of transition")
    c = ProfileParameter(1.2, "Outer slope")
    r_t = ProfileParameter(200.0, "Scale radius [kpc]")
    T_min = ProfileParameter(2.0, "Minimum temperature [keV]")
    r_cool = ProfileParameter(10.0, "Cooling radius [kpc]")
    a_cool = ProfileParameter(0.2, "Cooling slope")

    @staticmethod
    def default_principle_function(r, T_0, a, b, c, r_t, T_min, r_cool, a_cool):
        x = (r / r_cool) ** a_cool
        t = (r / r_t) ** (-a) / ((1.0 + (r / r_t) ** b) ** (c / b))
        return T_0 * t * (x + T_min / T_0) / (x + 1)

    CLASS_FUNCTIONS = {"default": default_principle_function}


class AM06TemperatureProfile(RadialTemperatureProfile):
    """
    Ascasibar & Markevitch (2006) temperature profile for galaxy clusters.

    Parameters
    ----------
    T_0 : float
        The scale temperature of the profile in keV.
    a : float
        The scale radius in kpc.
    a_c : float
        The cooling radius in kpc.
    c : float
        The scale of the temperature drop of the cool core.

    Equation
    --------
    .. math::

        T(r) = T_0 \frac{1}{1 + r / a} \left( \frac{c + r/a_c}{1 + r/a_c} \right)

    References
    ----------
    Ascasibar, Y., & Markevitch, M. 2006, ApJ, 650, 102

    Examples
    --------
    .. code-block:: python

        profile = AM06TemperatureProfile(T_0=4, a=300, a_c=50, c=0.2)
        temperature = profile(100)  # Evaluate at r = 100 kpc
    """

    T_0 = ProfileParameter(4.0, "Scale temperature [keV]")
    a = ProfileParameter(300.0, "Scale radius [kpc]")
    a_c = ProfileParameter(50.0, "Cooling radius [kpc]")
    c = ProfileParameter(0.2, "Cooling core drop")

    @staticmethod
    def default_principle_function(r, T_0, a, a_c, c):
        return T_0 / (1.0 + r / a) * (c + r / a_c) / (1.0 + r / a_c)

    CLASS_FUNCTIONS = {"default": default_principle_function}


class UniversalPressureTemperatureProfile(RadialTemperatureProfile):
    """
    Universal pressure profile temperature model for galaxy clusters.

    Parameters
    ----------
    T_0 : float
        The central temperature in keV.
    r_s : float
        The scale radius in kpc.

    Equation
    --------
    .. math::

        T(r) = T_0 \left(1 + \frac{r}{r_s}\right)^{-1.5}

    Examples
    --------
    .. code-block:: python

        profile = UniversalPressureTemperatureProfile(T_0=5, r_s=300)
        temperature = profile(100)  # Evaluate at r = 100 kpc
    """

    T_0 = ProfileParameter(5.0, "Central temperature [keV]")
    r_s = ProfileParameter(300.0, "Scale radius [kpc]")

    @staticmethod
    def default_principle_function(r, T_0, r_s):
        return T_0 * (1 + r / r_s) ** -1.5

    CLASS_FUNCTIONS = {"default": default_principle_function}


class IsothermalTemperatureProfile(RadialTemperatureProfile):
    """
    Isothermal temperature profile for galaxy clusters.

    Parameters
    ----------
    T_0 : float
        The constant temperature in keV.

    Equation
    --------
    .. math::

        T(r) = T_0

    Examples
    --------
    .. code-block:: python

        profile = IsothermalTemperatureProfile(T_0=5)
        temperature = profile(100)  # Evaluate at r = 100 kpc
    """

    T_0 = ProfileParameter(5.0, "Constant temperature [keV]")

    @staticmethod
    def default_principle_function(r, T_0):
        return T_0

    CLASS_FUNCTIONS = {"default": default_principle_function}


class CoolingFlowTemperatureProfile(RadialTemperatureProfile):
    """
    Cooling flow temperature profile for galaxy clusters.

    Parameters
    ----------
    T_0 : float
        The central temperature in keV.
    r_c : float
        The core radius in kpc.
    a : float
        The slope parameter.

    Equation
    --------
    .. math::

        T(r) = T_0 \left( \frac{r}{r_c} \right)^{-a}

    Examples
    --------
    .. code-block:: python

        profile = CoolingFlowTemperatureProfile(T_0=5, r_c=100, a=0.8)
        temperature = profile(50)  # Evaluate at r = 50 kpc
    """

    T_0 = ProfileParameter(5.0, "Central temperature [keV]")
    r_c = ProfileParameter(100.0, "Core radius [kpc]")
    a = ProfileParameter(0.8, "Slope parameter")

    @staticmethod
    def default_principle_function(r, T_0, r_c, a):
        return T_0 * (r / r_c) ** -a

    CLASS_FUNCTIONS = {"default": default_principle_function}


class DoubleBetaTemperatureProfile(RadialTemperatureProfile):
    """
    Double beta-model temperature profile for galaxy clusters.

    Parameters
    ----------
    T_0 : float
        The temperature for the first beta component in keV.
    r_c : float
        The core radius in kpc.
    beta_1 : float
        The slope of the first beta component.
    T_1 : float
        The temperature for the second beta component in keV.
    beta_2 : float
        The slope of the second beta component.

    Equation
    --------
    .. math::

        T(r) = T_0 \left(1 + \left( \frac{r}{r_c} \right)^2 \right)^{-\beta_1}
             + T_1 \left(1 + \left( \frac{r}{r_c} \right)^2 \right)^{-\beta_2}

    Examples
    --------
    .. code-block:: python

        profile = DoubleBetaTemperatureProfile(T_0=5, r_c=100, beta_1=0.8, T_1=3, beta_2=1.2)
        temperature = profile(50)  # Evaluate at r = 50 kpc
    """

    T_0 = ProfileParameter(5.0, "First temperature [keV]")
    r_c = ProfileParameter(100.0, "Core radius [kpc]")
    beta_1 = ProfileParameter(0.8, "First beta slope")
    T_1 = ProfileParameter(3.0, "Second temperature [keV]")
    beta_2 = ProfileParameter(1.2, "Second beta slope")

    @staticmethod
    def default_principle_function(r, T_0, r_c, beta_1, T_1, beta_2):
        return (
            T_0 * (1 + (r / r_c) ** 2) ** -beta_1
            + T_1 * (1 + (r / r_c) ** 2) ** -beta_2
        )

    CLASS_FUNCTIONS = {"default": default_principle_function}


class BetaModelTemperatureProfile(RadialTemperatureProfile):
    """
    Standard beta-model temperature profile for galaxy clusters.

    Parameters
    ----------
    T_0 : float
        The central temperature in keV.
    r_c : float
        The core radius in kpc.
    beta : float
        The slope parameter.

    Equation
    --------
    .. math::

        T(r) = T_0 \left(1 + \left( \frac{r}{r_c} \right)^2 \right)^{-\beta}

    Examples
    --------
    .. code-block:: python

        profile = BetaModelTemperatureProfile(T_0=5, r_c=100, beta=0.8)
        temperature = profile(50)  # Evaluate at r = 50 kpc
    """

    T_0 = ProfileParameter(5.0, "Central temperature [keV]")
    r_c = ProfileParameter(100.0, "Core radius [kpc]")
    beta = ProfileParameter(0.8, "Slope parameter")

    @staticmethod
    def default_principle_function(r, T_0, r_c, beta):
        return T_0 * (1 + (r / r_c) ** 2) ** -beta

    CLASS_FUNCTIONS = {"default": default_principle_function}


class BaselineEntropyProfile(RadialEntropyProfile):
    """
    The baseline entropy profile for galaxy clusters (Voit et al. 2005).

    Parameters
    ----------
    K_0 : float
        The central entropy floor in keV*cm^2.
    K_200 : float
        The entropy at the radius r_200 in keV*cm^2.
    r_200 : float
        The virial radius in kpc.
    alpha : float
        The logarithmic slope of the profile.

    Equation
    --------
    .. math::

        K(r) = K_0 + K_{200} \left( \frac{r}{r_{200}} \right)^{\alpha}

    References
    ----------
    Voit, G.M., Kay, S.T., & Bryan, G.L. 2005, MNRAS, 364, 909

    Examples
    --------
    .. code-block:: python

        profile = BaselineEntropyProfile(K_0=10, K_200=200, r_200=1000, alpha=1.1)
        entropy = profile(500)  # Evaluate at r = 500 kpc
    """

    K_0 = ProfileParameter(10.0, "Central entropy [keV cm^2]")
    K_200 = ProfileParameter(200.0, "Entropy at r_200 [keV cm^2]")
    r_200 = ProfileParameter(1000.0, "Virial radius [kpc]")
    alpha = ProfileParameter(1.1, "Logarithmic slope")

    @staticmethod
    def default_principle_function(r, K_0, K_200, r_200, alpha):
        return K_0 + K_200 * (r / r_200) ** alpha

    CLASS_FUNCTIONS = {"default": default_principle_function}


class BrokenEntropyProfile(RadialEntropyProfile):
    """
    A broken entropy profile model for galaxy clusters.

    Parameters
    ----------
    r_s : float
        The scale radius in kpc.
    K_scale : float
        The entropy scaling factor.
    alpha : float
        The slope at small radii.
    K_0 : float, optional
        A core entropy floor. Default is 0.0.

    Equation
    --------
    .. math::

        K(r) = K_\mathrm{scale} \left[ K_0 + \left( \frac{r}{r_s} \right)^{\alpha} \left(1 + \left( \frac{r}{r_s} \right)^5 \right)^{0.2 (1.1 - \alpha)} \right]

    Examples
    --------
    .. code-block:: python

        profile = BrokenEntropyProfile(r_s=300, K_scale=200, alpha=1.1, K_0=20)
        entropy = profile(150)  # Evaluate at r = 150 kpc
    """

    r_s = ProfileParameter(300.0, "Scale radius [kpc]")
    K_scale = ProfileParameter(200.0, "Entropy scaling factor")
    alpha = ProfileParameter(1.1, "Slope at small radii")
    K_0 = ProfileParameter(0.0, "Core entropy floor [keV cm^2]")

    @staticmethod
    def default_principle_function(r, r_s, K_scale, alpha, K_0=0.0):
        x = r / r_s
        ret = (x**alpha) * (1.0 + x**5) ** (0.2 * (1.1 - alpha))
        return K_scale * (K_0 + ret)

    CLASS_FUNCTIONS = {"default": default_principle_function}


class WalkerEntropyProfile(RadialEntropyProfile):
    """
    Walker entropy profile for galaxy clusters.

    Parameters
    ----------
    r_200 : float
        The virial radius in kpc.
    A : float
        The normalization constant for the profile.
    B : float
        A parameter controlling the cutoff in entropy.
    K_scale : float
        The entropy scaling factor.
    alpha : float, optional
        The slope at small radii. Default is 1.1.

    Equation
    --------
    .. math::

        K(r) = K_\mathrm{scale} \left( A \left( \frac{r}{r_{200}} \right)^{\alpha} \exp \left( - \left( \frac{r}{B} \right)^2 \right) \right)

    Examples
    --------
    .. code-block:: python

        profile = WalkerEntropyProfile(r_200=1000, A=0.5, B=0.2, K_scale=100)
        entropy = profile(200)  # Evaluate at r = 200 kpc
    """

    r_200 = ProfileParameter(1000.0, "Virial radius [kpc]")
    A = ProfileParameter(0.5, "Normalization constant")
    B = ProfileParameter(0.2, "Cutoff parameter")
    K_scale = ProfileParameter(100.0, "Entropy scaling factor")
    alpha = ProfileParameter(1.1, "Slope at small radii")

    @staticmethod
    def default_principle_function(r, r_200, A, B, K_scale, alpha=1.1):
        x = r / r_200
        return K_scale * (A * x**alpha) * np.exp(-((x / B) ** 2))

    CLASS_FUNCTIONS = {"default": default_principle_function}
