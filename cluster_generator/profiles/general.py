"""Generic profiles for various astrophysical applications.

Examples
--------
.. code-block:: python

    # Example usage of ConstantProfile
    const_profile = ConstantProfile(value=10, NDIM=1)
    result = const_profile(50)  # Evaluate the profile at r = 50 for NDIM = 1

    # Example usage of RadialPowerLawProfile
    powerlaw_profile = RadialPowerLawProfile(A=5.0, r_s=2.0, alpha=-2.0)
    result = powerlaw_profile(10)  # Evaluate the profile at r = 10
"""

import numpy as np

from cluster_generator.profiles._abc import Profile, ProfileParameter, RadialProfile


class ConstantProfile(Profile):
    """
    Constant profile for NDIM = 1, 2, or 3 dimensions.

    This profile returns a constant value at all radial distances `r` for a specified number
    of dimensions (1D, 2D, or 3D). It can be used to represent quantities like constant density
    or temperature over a volume.

    Parameters
    ----------
    value : float
        The constant value for the profile.
    NDIM : int, optional
        The number of dimensions (1, 2, or 3). Default is 3.

    Examples
    --------
    .. code-block:: python

        profile_1d = ConstantProfile(value=10, NDIM=1)
        profile_2d = ConstantProfile(value=20, NDIM=2)
        profile_3d = ConstantProfile(value=30, NDIM=3)

        r = 50
        result_1d = profile_1d(r)  # Evaluate the profile at r = 50 for NDIM = 1
        result_2d = profile_2d(r)  # Evaluate the profile at r = 50 for NDIM = 2
        result_3d = profile_3d(r)  # Evaluate the profile at r = 50 for NDIM = 3

    Notes
    -----
    This profile can be applied to any physical quantity that is constant over a region,
    such as temperature, density, or pressure. The profile remains invariant regardless of
    the radial distance from the origin.
    """

    value = ProfileParameter(1.0, "The constant value of the profile")
    NDIM = ProfileParameter(3, "The number of dimensions (1, 2, or 3)")

    @staticmethod
    def default_principle_function(r, value=1.0, NDIM=None):
        """
        Returns the constant value across the given radial distances.

        Parameters
        ----------
        r : array-like
            Radial distances at which the profile is evaluated.
        value : float, optional
            The constant value to return. Default is 1.0.
        NDIM : int, optional
            The number of dimensions (1, 2, or 3). Default is None.

        Returns
        -------
        array-like
            An array of the same shape as `r`, with each element equal to `value`.
        """
        return value * np.ones_like(r)

    CLASS_FUNCTIONS = {"default": default_principle_function}

    def _validate_geometry_handler(self):
        """
        Validate the geometry handler to ensure dimensional compatibility.

        Raises an error if the dimensions do not match the expected `NDIMS`.

        Raises
        ------
        AttributeError
            If the geometry handler has a different number of dimensions than expected.
        """
        geometry_dimensions = len(self.geometry_handler.FREE_AXES)

        if geometry_dimensions != self.NDIMS:
            raise AttributeError(
                f"The geometry handler {self.geometry_handler} has {geometry_dimensions} dimensions, "
                f"but {self.__class__.__name__} expected {self.NDIMS}."
            )


class RadialPowerLawProfile(RadialProfile):
    """
    Radial power-law profile.

    This profile follows a power-law relationship with the radial distance `r`. It is typically used
    to model astrophysical quantities like density or mass distributions, where the profile decays or
    grows according to a power law.

    The profile is represented as:

    .. math::
        f(r) = A \\left( \\frac{r}{r_s} \\right)^{\\alpha}

    where:
    - `A` is the amplitude of the profile,
    - `r_s` is the scale radius, and
    - `alpha` is the power-law index.

    Parameters
    ----------
    A : float
        The amplitude of the profile.
    r_s : float
        The scale radius.
    alpha : float
        The power-law index.

    Examples
    --------
    .. code-block:: python

        profile = RadialPowerLawProfile(A=5.0, r_s=2.0, alpha=-2.0)
        result = profile(10)  # Evaluate the profile at r = 10

    Notes
    -----
    This profile is suitable for a wide variety of astrophysical modeling tasks where radial symmetry
    is assumed, such as modeling the density distribution of gas or dark matter.
    """

    A = ProfileParameter(1.0, "Amplitude of the profile")
    r_s = ProfileParameter(1.0, "Scale radius")
    alpha = ProfileParameter(-2.0, "Power-law index")

    @staticmethod
    def default_principle_function(r, A=1.0, r_s=1.0, alpha=-2.0):
        """
        Returns the radial power-law profile evaluated at the given radial distances.

        Parameters
        ----------
        r : array-like
            Radial distances at which the profile is evaluated.
        A : float, optional
            The amplitude of the profile. Default is 1.0.
        r_s : float, optional
            The scale radius. Default is 1.0.
        alpha : float, optional
            The power-law index. Default is -2.0.

        Returns
        -------
        array-like
            The radial power-law profile evaluated at the given radial distances.
        """
        return A * (r / r_s) ** alpha

    CLASS_FUNCTIONS = {"default": default_principle_function}
