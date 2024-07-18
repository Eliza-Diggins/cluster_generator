"""Radial profiles for representing physical properties of galaxy clusters.

Notes
-----

In addition to the built-in :py:class:`RadialProfile` instances, you can also create your own simply by initializing the :py:class:`RadialProfile`
class on a function representing your preferred radial profile.
"""
from numbers import Number
from typing import Any, Callable, Literal

import numpy as np
import yt.utilities.cosmology
from numpy.typing import ArrayLike

from cluster_generator.utilities.types import NumericInput, Registry, Self
from cluster_generator.utilities.utils import enforce_style  # back-compat.

NumericCallable = Callable[[NumericInput], NumericInput]

_nfw_factor = lambda conc: 1.0 / (np.log(conc + 1.0) - conc / (1.0 + conc))


class ProfileRegistry(Registry):
    """Registry of valid built-in :py:class:`RadialProfile` generators."""

    @property
    def types(self) -> list[str]:
        """The available types of profile in this registry."""
        _t = []
        for _, v in self._mapping.items():
            if hasattr(v, "type"):
                _t.append(v.type)

        return list(set(list(_t)))


DEFAULT_PROFILE_REGISTRY: ProfileRegistry = ProfileRegistry()
""" ProfileRegistry: The default registry of radial profiles."""


class RadialProfile:
    """Class representation of a single radial profile."""

    def __init__(self, profile: Self | NumericCallable):
        """Initialize a :py:class:`RadialProfile` instance.

        Parameters
        ----------
        profile: callable
            The function (or existing :py:class:`RadialProfile`) to create the radial profile.
        """
        if isinstance(profile, RadialProfile):
            self.profile: NumericCallable = profile.profile
        else:
            self.profile: NumericCallable = profile
            """ callable: The underlying function of radius for this radial profile."""

    def __call__(self, r: NumericInput) -> NumericInput:
        return self.profile(r)

    def _do_op(
        self,
        other: Any,
        op: Callable[[NumericInput, NumericInput], NumericInput],
    ) -> NumericCallable:
        if hasattr(other, "profile"):
            p = lambda r: op(self.profile(r), other.profile(r))
        else:
            p = lambda r: op(self.profile(r), other)
        return p

    def __add__(self, other: Self | NumericCallable) -> Self:
        p = self._do_op(other, np.add)
        return RadialProfile(p)

    def __mul__(self, other: Self | NumericCallable) -> Self:
        p = self._do_op(other, np.multiply)
        return RadialProfile(p)

    __radd__ = __add__
    __rmul__ = __mul__

    def __pow__(self, power: int | float | complex) -> Self:
        p = lambda r: self.profile(r) ** power
        return RadialProfile(p)

    def add_core(self, r_core: float, alpha: float) -> Self:
        r"""Add a small core with radius ``r_core`` to the profile by multiplying it by
        :math:`1-exp(-(r/r_{\rm{core}})^{\alpha})`.

        Parameters
        ----------
        r_core : float
            The core radius in kpc.
        alpha : float
            The power-low index inside the exponential.
        """

        def _core(r):
            x = r / r_core
            ret = 1.0 - np.exp(-(x**alpha))
            return self.profile(r) * ret

        return RadialProfile(_core)

    def cutoff(self, r_cut: float, k: float = 5) -> Self:
        """Truncate the profile with a particular sharpness dictated by ``k`` at radius
        ``r_cut``.

        Parameters
        ----------
        r_cut: float
            The cutoff point.
        k: float
            The truncation sharpness parameter.

        Returns
        -------
        """

        def _cutoff(r):
            x = r / r_cut
            step = 1.0 / (1.0 + np.exp(-2 * k * (x - 1)))
            p = self.profile(r) * (1.0 - step)
            return p

        return RadialProfile(_cutoff)

    @classmethod
    def from_array(cls, r: ArrayLike, f_r: ArrayLike) -> Self:
        """Generate a callable radial profile using an array of radii and an array of
        values.

        Parameters
        ----------
        r : array-like
            Array of radii in kpc.
        f_r : array-like
            Array of profile values in the appropriate units.
        """
        from scipy.interpolate import UnivariateSpline

        f = UnivariateSpline(r, f_r)
        return cls(f)

    @enforce_style
    def plot(
        self,
        rmin: Number,
        rmax: Number,
        num_points: int = 1000,
        fig: Any = None,
        ax: Any = None,
        scale: Literal["log", "linear"] = "log",
        **kwargs
    ) -> tuple[Any, Any]:
        """Make a quick plot of a profile using Matplotlib.

        Parameters
        ----------
        rmin : float
            The minimum radius of the plot in kpc.
        rmax : float
            The maximum radius of the plot in kpc.
        num_points : integer, optional
            The number of logspaced points between rmin
            and rmax to use when making the plot. Default: 1000
        scale: str, optional
            The display scale to utilize. Either ``'log'`` or ``'linear'``
        fig : :class:`~matplotlib.figure.Figure`, optional
            A Figure instance to plot in. Default: None, one will be
            created if not provided.
        ax : :class:`~matplotlib.axes.Axes`, optional
            An Axes instance to plot in. Default: None, one will be
            created if not provided.

        Return
        ------
        fig: :py:class:`plt.Figure`
            The figure.
        ax: :py:class:`plt.Axes`
            The axes.
        """
        import matplotlib.pyplot as plt

        # setup the figure if not provided.
        if fig is None:
            fig = plt.figure(figsize=(10, 10))
        if ax is None:
            ax = fig.add_subplot(111)

        # Construct abscissa
        rr = np.logspace(np.log10(rmin), np.log10(rmax), num_points, endpoint=True)

        # plotting
        ax.plot(rr, self(rr), **kwargs)
        ax.set_xlabel("Radius (kpc)")

        if scale == "log":
            ax.set_yscale("log")
            ax.set_xscale("log")

        return fig, ax


@DEFAULT_PROFILE_REGISTRY.autoregister(type="generic")
def constant_profile(const: Number) -> RadialProfile:
    """A constant profile.

    Parameters
    ----------
    const : float
        The value of the constant.
    """
    p = lambda r: const
    return RadialProfile(
        np.vectorize(p)
    )  # Ensure that r array-like -> f(r) array-like.


@DEFAULT_PROFILE_REGISTRY.autoregister(type="generic")
def power_law_profile(A: float, r_s: float, alpha: float) -> RadialProfile:
    """A profile which is a power-law with radius, scaled so that it has a certain value
    ``A`` at a scale radius ``r_s``. Can be used as a density, temperature, mass, or
    entropy profile (or whatever else one may need).

    Parameters
    ----------
    A : float
        Scale value of the profile at r = r_s.
    r_s : float
        Scale radius in kpc.
    alpha : float
        Power-law index of the profile.
    """
    p = lambda r: A * (r / r_s) ** alpha
    return RadialProfile(p)


@DEFAULT_PROFILE_REGISTRY.autoregister(type="density")
def beta_model_profile(rho_c: float, r_c: float, beta: float) -> RadialProfile:
    """A beta-model density profile (Cavaliere A., Fusco-Femiano R., 1976, A&A, 49,
    137).

    Parameters
    ----------
    rho_c : float
        The core density in Msun/kpc**3.
    r_c : float
        The core radius in kpc.
    beta : float
        The beta parameter.
    """
    p = lambda r: rho_c * ((1 + (r / r_c) ** 2) ** (-1.5 * beta))
    return RadialProfile(p)


@DEFAULT_PROFILE_REGISTRY.autoregister(type="density")
def hernquist_density_profile(M_0: float, a: float) -> RadialProfile:
    """A Hernquist density profile (Hernquist, L. 1990, ApJ, 356, 359).

    Parameters
    ----------
    M_0 : float
        The total mass in Msun.
    a : float
        The scale radius in kpc.
    """
    p = lambda r: M_0 / (2.0 * np.pi * a**3) / ((r / a) * (1.0 + r / a) ** 3)
    return RadialProfile(p)


@DEFAULT_PROFILE_REGISTRY.autoregister(type="density")
def cored_hernquist_density_profile(M_0: float, a: float, b: float) -> RadialProfile:
    """A Hernquist density profile (Hernquist, L. 1990, ApJ, 356, 359) with a core
    radius.

    Parameters
    ----------
    M_0 : float
        The total mass in Msun.
    a : float
        The scale radius in kpc.
    b : float
        The core radius in kpc.
    """
    p = (
        lambda r: M_0
        * b
        / (2.0 * np.pi * a**3)
        / ((1.0 + b * r / a) * (1.0 + r / a) ** 3)
    )
    return RadialProfile(p)


@DEFAULT_PROFILE_REGISTRY.autoregister(type="mass")
def hernquist_mass_profile(M_0: float, a: float) -> RadialProfile:
    """A Hernquist mass profile (Hernquist, L. 1990, ApJ, 356, 359).

    Parameters
    ----------
    M_0 : float
        The total mass in Msun.
    a : float
        The scale radius in kpc.
    """
    p = lambda r: M_0 * r**2 / (r + a) ** 2
    return RadialProfile(p)


def convert_nfw_to_hernquist(M_200: float, r_200: float, conc: float) -> RadialProfile:
    """Given M200, r200, and a concentration parameter for an NFW profile, return the
    Hernquist mass and scale radius parameters.

    Parameters
    ----------
    M_200 : float
        The mass of the halo at r200 in Msun.
    r_200 : float
        The radius corresponding to the overdensity of 200 times the
        critical density of the universe in kpc.
    conc : float
        The concentration parameter r200/r_s for the NFW profile.
    """
    a = r_200 / (np.sqrt(0.5 * conc * conc * _nfw_factor(conc)) - 1.0)
    M0 = M_200 * (r_200 + a) ** 2 / r_200**2
    return M0, a


@DEFAULT_PROFILE_REGISTRY.autoregister(type="density")
def nfw_density_profile(rho_s: float, r_s: float) -> RadialProfile:
    """An NFW density profile (Navarro, J.F., Frenk, C.S., & White, S.D.M. 1996, ApJ,
    462, 563).

    Parameters
    ----------
    rho_s : float
        The scale density in Msun/kpc**3.
    r_s : float
        The scale radius in kpc.
    """
    p = lambda r: rho_s / ((r / r_s) * (1.0 + r / r_s) ** 2)
    return RadialProfile(p)


@DEFAULT_PROFILE_REGISTRY.autoregister(type="mass")
def nfw_mass_profile(rho_s: float, r_s: float) -> RadialProfile:
    """An NFW mass profile (Navarro, J.F., Frenk, C.S., & White, S.D.M. 1996, ApJ, 462,
    563).

    Parameters
    ----------
    rho_s : float
        The scale density in Msun/kpc**3.
    r_s : float
        The scale radius in kpc.
    """

    def _nfw(r):
        x = r / r_s
        return 4 * np.pi * rho_s * r_s**3 * (np.log(1 + x) - x / (1 + x))

    return RadialProfile(_nfw)


def nfw_scale_density(
    conc: float,
    z: float = 0.0,
    delta: float = 200.0,
    cosmo: yt.utilities.cosmology.Cosmology = None,
) -> RadialProfile:
    """Compute a scale density parameter for an NFW profile given a concentration
    parameter, and optionally a redshift, overdensity, and cosmology.

    Parameters
    ----------
    conc : float
        The concentration parameter for the halo, which should
        correspond the selected overdensity (which has a default
        of 200).
    z : float, optional
        The redshift of the halo formation. Default: 0.0
    delta : float, optional
        The overdensity parameter for which the concentration
        is defined. Default: 200.0
    cosmo : yt Cosmology object
        The cosmology to be used when computing the critical
        density. If not supplied, a default one from yt will
        be used.
    """
    from yt.utilities.cosmology import Cosmology

    if cosmo is None:
        cosmo = Cosmology()
    rho_crit = cosmo.critical_density(z).to_value("Msun/kpc**3")
    rho_s = delta * rho_crit * conc**3 * _nfw_factor(conc) / 3.0
    return rho_s


@DEFAULT_PROFILE_REGISTRY.autoregister(type="density")
def tnfw_density_profile(rho_s: float, r_s: float, r_t: float) -> RadialProfile:
    """A truncated NFW (tNFW) density profile (Baltz, E.A., Marshall, P., & Oguri, M.
    2009, JCAP, 2009, 015).

    Parameters
    ----------
    rho_s : float
        The scale density in Msun/kpc**3.
    r_s : float
        The scale radius in kpc.
    r_t : float
        The truncation radius in kpc.
    """

    def _tnfw(r):
        profile = rho_s / ((r / r_s) * (1 + r / r_s) ** 2)
        profile /= 1 + (r / r_t) ** 2
        return profile

    return RadialProfile(_tnfw)


@DEFAULT_PROFILE_REGISTRY.autoregister(type="mass")
def tnfw_mass_profile(rho_s: float, r_s: float, r_t: float) -> RadialProfile:
    """A truncated NFW (tNFW) mass profile (Baltz, E.A., Marshall, P., & Oguri, M. 2009,
    JCAP, 2009, 015).

    Parameters
    ----------
    rho_s : float
        The scale density in Msun/kpc**3.
    r_s : float
        The scale radius in kpc.
    r_t : float
        The truncation radius in kpc.
    """
    from sympy import Symbol, integrate, lambdify

    xx = Symbol("x")
    aa = Symbol("a")
    yy = Symbol("y")
    f = integrate(xx**2 / (xx * (1 + xx) ** 2) / (1 + (xx / aa) ** 2), (xx, 0, yy))
    fl = lambdify((yy, aa), f, modules="numpy")

    def _tnfw(r):
        x = r / r_s
        a = r_t / r_s
        return 4 * np.pi * rho_s * r_s**3 * fl(x, a).astype("float64")

    return RadialProfile(_tnfw)


@DEFAULT_PROFILE_REGISTRY.autoregister(type="density")
def snfw_density_profile(M: float, a: float) -> RadialProfile:
    """A "super-NFW" density profile (Lilley, E. J., Wyn Evans, N., & Sanders, J.L.
    2018, MNRAS).

    Parameters
    ----------
    M : float
        The total mass in Msun.
    a : float
        The scale radius in kpc.
    """

    def _snfw(r):
        x = r / a
        return 3.0 * M / (16.0 * np.pi * a**3) / (x * (1.0 + x) ** 2.5)

    return RadialProfile(_snfw)


@DEFAULT_PROFILE_REGISTRY.autoregister(type="mass")
def snfw_mass_profile(M: float, a: float) -> RadialProfile:
    """A "super-NFW" mass profile (Lilley, E. J., Wyn Evans, N., & Sanders, J.L. 2018,
    MNRAS).

    Parameters
    ----------
    M : float
        The total mass in Msun.
    a : float
        The scale radius in kpc.
    """

    def _snfw(r):
        x = r / a
        return M * (1.0 - (2.0 + 3.0 * x) / (2.0 * (1.0 + x) ** 1.5))

    return RadialProfile(_snfw)


def snfw_total_mass(mass: float, radius: float, a: float) -> RadialProfile:
    """Find the total mass parameter for the super-NFW model by inputting a reference
    mass and radius (say, M200c and R200c), along with the scale radius.

    Parameters
    ----------
    mass : float
        The input mass in Msun.
    radius : float
        The input radius that the input ``mass`` corresponds to in kpc.
    a : float
        The scale radius in kpc.
    """
    mp = snfw_mass_profile(1.0, a)
    return mass / mp(radius)


@DEFAULT_PROFILE_REGISTRY.autoregister(type="density")
def cored_snfw_density_profile(M: float, a: float, r_c: float) -> RadialProfile:
    """A cored "super-NFW" density profile (Lilley, E. J., Wyn Evans, N., & Sanders,
    J.L. 2018, MNRAS).

    Parameters
    ----------
    M : float
        The total mass in Msun.
    a : float
        The scale radius in kpc.
    r_c : float
        The core radius in kpc.
    """
    b = a / r_c

    def _snfw(r):
        x = r / a
        return (
            3.0 * M * b / (16.0 * np.pi * a**3) / ((1.0 + b * x) * (1.0 + x) ** 2.5)
        )

    return RadialProfile(_snfw)


@DEFAULT_PROFILE_REGISTRY.autoregister(type="mass")
def cored_snfw_mass_profile(M: float, a: float, r_c: float) -> RadialProfile:
    """A cored "super-NFW" mass profile (Lilley, E. J., Wyn Evans, N., & Sanders, J.L.
    2018, MNRAS).

    Parameters
    ----------
    M : float
        The total mass in Msun.
    a : float
        The scale radius in kpc.
    r_c : float
        The core radius in kpc.
    """
    b = a / r_c

    def _snfw(r):
        x = r / a
        y = np.complex128(np.sqrt(x + 1.0))
        d = np.sqrt(np.complex128(b / (1.0 - b)))
        e = b * (b - 1.0) ** 2
        ret = (1.0 - 1.0 / y) * (b - 2.0) / (b - 1.0) ** 2
        ret += (1.0 / y**3 - 1.0) / (3.0 * (b - 1.0))
        ret += d * (np.arctan(y * d) - np.arctan(d)) / e
        return 1.5 * M * b * ret.astype("float64")

    return RadialProfile(_snfw)


def snfw_conc(conc_nfw: float) -> float:
    """Given an NFW concentration parameter, calculate the corresponding sNFW
    concentration parameter. This comes from Equation 31 of (Lilley, E. J., Wyn Evans,
    N., & Sanders, J.L. 2018, MNRAS).

    Parameters
    ----------
    conc_nfw : float
        NFW concentration for r200c.
    """
    return 0.76 * conc_nfw + 1.36


def cored_snfw_total_mass(
    mass: float, radius: float, a: float, r_c: float
) -> RadialProfile:
    """Find the total mass parameter for the cored super-NFW model by inputting a
    reference mass and radius (say, M200c and R200c), along with the scale radius.

    Parameters
    ----------
    mass : float
        The input mass in Msun.
    radius : float
        The input radius that the input ``mass`` corresponds to in kpc.
    a : float
        The scale radius in kpc.
    r_c : float
        The core radius in kpc.
    """
    mp = cored_snfw_mass_profile(1.0, a, r_c)
    return mass / mp(radius)


_dn = lambda n: 3.0 * n - 1.0 / 3.0 + 8.0 / (1215.0 * n) + 184.0 / (229635.0 * n * n)


@DEFAULT_PROFILE_REGISTRY.autoregister(type="density")
def einasto_density_profile(M: float, r_s: float, n: int) -> RadialProfile:
    """A density profile where the logarithmic slope is a power-law. The form here is
    that given in Section 2 of Retana-Montenegro et al. 2012, A&A, 540, A70.

    Parameters
    ----------
    M : float
        The total mass of the profile in M.
    r_s : float
        The scale radius in kpc.
    n : int
        The inverse power-law index.
    """
    from scipy.special import gamma

    alpha = 1.0 / n
    h = r_s / _dn(n) ** n
    rho_0 = M / (4.0 * np.pi * h**3 * n * gamma(3.0 * n))

    def _einasto(r):
        s = r / h
        return rho_0 * np.exp(-(s**alpha))

    return RadialProfile(_einasto)


@DEFAULT_PROFILE_REGISTRY.autoregister(type="mass")
def einasto_mass_profile(M: float, r_s: float, n: int) -> RadialProfile:
    """A mass profile where the logarithmic slope is a power-law. The form here is that
    given in Section 2 of Retana-Montenegro et al. 2012, A&A, 540, A70.

    Parameters
    ----------
    M : float
        The total mass of the profile in M.
    r_s : float
        The scale radius in kpc.
    n : int
        The inverse power-law index.
    """
    from scipy.special import gammaincc

    alpha = 1.0 / n
    h = r_s / _dn(n) ** n

    def _einasto(r):
        s = r / h
        return M * (1.0 - gammaincc(3.0 * n, s**alpha))

    return RadialProfile(_einasto)


@DEFAULT_PROFILE_REGISTRY.autoregister(type="density")
def am06_density_profile(
    rho_0: float, a: float, a_c: float, c: float, n: int
) -> RadialProfile:
    """The density profile for galaxy clusters suggested by Ascasibar, Y., & Markevitch,
    M. 2006, ApJ, 650, 102. Works best in concert with the ``am06_temperature_profile``.

    Parameters
    ----------
    rho_0 : float
        The scale density of the profile in Msun/kpc**3.
    a : float
        The scale radius in kpc.
    a_c : float
        The scale radius of the cool core in kpc.
    c : float
        The scale of the temperature drop of the cool core.
    n : int
    """
    alpha = -1.0 - n * (c - 1.0) / (c - a / a_c)
    beta = 1.0 - n * (1.0 - a / a_c) / (c - a / a_c)
    p = (
        lambda r: rho_0
        * (1.0 + r / a_c)
        * (1.0 + r / a_c / c) ** alpha
        * (1.0 + r / a) ** beta
    )
    return RadialProfile(p)


@DEFAULT_PROFILE_REGISTRY.autoregister(type="density")
def vikhlinin_density_profile(
    rho_0: float,
    r_c: float,
    r_s: float,
    alpha: float,
    beta: float,
    epsilon: float,
    gamma: float = None,
) -> RadialProfile:
    """A modified beta-model density profile for galaxy clusters from Vikhlinin, A.,
    Kravtsov, A., Forman, W., et al. 2006, ApJ, 640, 691.

    Parameters
    ----------
    rho_0 : float
        The scale density in Msun/kpc**3.
    r_c : float
        The core radius in kpc.
    r_s : float
        The scale radius in kpc.
    alpha : float
        The inner logarithmic slope parameter.
    beta : float
        The middle logarithmic slope parameter.
    epsilon : float
        The outer logarithmic slope parameter.
    gamma : float
        This parameter controls the width of the outer
        transition. If None, it will be gamma = 3 by default.
    """
    if gamma is None:
        gamma = 3.0
    profile = (
        lambda r: rho_0
        * (r / r_c) ** (-0.5 * alpha)
        * (1.0 + (r / r_c) ** 2) ** (-1.5 * beta + 0.25 * alpha)
        * (1.0 + (r / r_s) ** gamma) ** (-0.5 * epsilon / gamma)
    )
    return RadialProfile(profile)


@DEFAULT_PROFILE_REGISTRY.autoregister(type="temperature")
def vikhlinin_temperature_profile(
    T_0: float,
    a: float,
    b: float,
    c: float,
    r_t: float,
    T_min: float,
    r_cool: float,
    a_cool: float,
) -> RadialProfile:
    """A temperature profile for galaxy clusters from Vikhlinin, A., Kravtsov, A.,
    Forman, W., et al. 2006, ApJ, 640, 691.

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
        The scale radius kpc.
    T_min : float
        The minimum temperature in keV.
    r_cool : float
        The cooling radius in kpc.
    a_cool : float
        The logarithmic slope in the cooling region.
    """

    def _temp(r):
        x = (r / r_cool) ** a_cool
        t = (r / r_t) ** (-a) / ((1.0 + (r / r_t) ** b) ** (c / b))
        return T_0 * t * (x + T_min / T_0) / (x + 1)

    return RadialProfile(_temp)


@DEFAULT_PROFILE_REGISTRY.autoregister(type="temperature")
def am06_temperature_profile(
    T_0: float, a: float, a_c: float, c: float
) -> RadialProfile:
    """The temperature profile for galaxy clusters suggested by Ascasibar, Y., &
    Markevitch, M. 2006, ApJ, 650, 102. Works best in concert with the
    ``am06_density_profile``.

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
    """
    p = lambda r: T_0 / (1.0 + r / a) * (c + r / a_c) / (1.0 + r / a_c)
    return RadialProfile(p)


@DEFAULT_PROFILE_REGISTRY.autoregister(type="entropy")
def baseline_entropy_profile(
    K_0: float, K_200: float, r_200: float, alpha: float
) -> RadialProfile:
    """The baseline entropy profile for galaxy clusters (Voit, G.M., Kay, S.T., & Bryan,
    G.L. 2005, MNRAS, 364, 909).

    Parameters
    ----------
    K_0 : float
        The central entropy floor in keV*cm**2.
    K_200 : float
        The entropy at the radius r_200 in keV*cm**2.
    r_200 : float
        The virial radius in kpc.
    alpha : float
        The logarithmic slope of the profile.
    """
    p = lambda r: K_0 + K_200 * (r / r_200) ** alpha
    return RadialProfile(p)


@DEFAULT_PROFILE_REGISTRY.autoregister(type="entropy")
def broken_entropy_profile(
    r_s: float, K_scale: float, alpha: float, K_0: float = 0.0
) -> RadialProfile:
    def _entr(r):
        x = r / r_s
        ret = (x**alpha) * (1.0 + x**5) ** (0.2 * (1.1 - alpha))
        return K_scale * (K_0 + ret)

    return RadialProfile(_entr)


@DEFAULT_PROFILE_REGISTRY.autoregister(type="entropy")
def walker_entropy_profile(
    r_200: float, A: float, B: float, K_scale: float, alpha: float = 1.1
) -> RadialProfile:
    def _entr(r):
        x = r / r_200
        return K_scale * (A * x**alpha) * np.exp(-((x / B) ** 2))

    return RadialProfile(_entr)


def rescale_profile_by_mass(profile, mass, radius):
    """Rescale a density ``profile`` by a total ``mass`` within some ``radius``.

    Parameters
    ----------
    profile : ``RadialProfile`` object
        The profile object to rescale.
    mass : float
        The input mass in Msun.
    radius : float
        The input radius that the input ``mass`` corresponds to in kpc.

    Examples
    --------
    >>> rho_0 = 1.0
    >>> a = 600.0
    >>> a_c = 60.0
    >>> c = 0.17
    >>> alpha = -2.0
    >>> gas_density = am06_density_profile(rho_0, a, a_c, c, alpha)
    >>> M200 = 1.0e14
    >>> r200 = 900.0
    >>> gas_density = rescale_profile_by_mass(gas_density, M200, r200)
    """
    from scipy.integrate import quad

    mass_int = lambda r: profile(r) * r * r
    rescale = mass / (4.0 * np.pi * quad(mass_int, 0.0, radius)[0])
    return rescale * profile


def find_overdensity_radius(m, delta, z=0.0, cosmo=None):
    """Given a mass value and an overdensity, find the radius that corresponds to that
    enclosed mass.

    Parameters
    ----------
    m : float
        The enclosed mass.
    delta : float
        The overdensity to compute the radius for.
    z : float, optional
        The redshift of the halo formation. Default: 0.0
    cosmo : yt ``Cosmology`` object
        The cosmology to be used when computing the critical
        density. If not supplied, a default one from yt will
        be used.
    """
    from yt.utilities.cosmology import Cosmology

    if cosmo is None:
        cosmo = Cosmology()
    rho_crit = cosmo.critical_density(z).to_value("Msun/kpc**3")
    return (3.0 * m / (4.0 * np.pi * delta * rho_crit)) ** (1.0 / 3.0)


def find_radius_mass(m_r, delta, z=0.0, cosmo=None):
    """Given a mass profile and an overdensity, find the radius and mass (e.g. M200,
    r200)

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
