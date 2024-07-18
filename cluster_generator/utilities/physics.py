"""Utilities module for physics routines and constants."""
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from unyt import Unit
from unyt import physical_constants as pc
from unyt import unyt_quantity

from cluster_generator.utilities.config import cgparams

if TYPE_CHECKING:
    from numbers import Number


mp: unyt_quantity = (pc.mp).to("Msun")
#: :py:class:`unyt.unyt_quantity`: Proton mass in solar masses.
G: unyt_quantity = (pc.G).to("kpc**3/Msun/Myr**2")
#: :py:class:`unyt.unyt_quantity`: Newton's gravitational constant in galactic units.
kboltz: unyt_quantity = (pc.kboltz).to("Msun*kpc**2/Myr**2/K")
#: :py:class:`unyt.unyt_quantity`: Boltzmann's constant
kpc_to_cm: float = (1.0 * Unit("kpc")).to_value("cm")
# float: The conversion of 1 kpc to centimeters.
X_H: float = cgparams.config.physics.hydrogen_abundance
""" float: The cosmological hydrogen abundance.

The adopted value for :math:`X_H` may be changed in the ``cluster_generator`` configuration. Default is 0.76.
"""
mu: float = 1.0 / (2.0 * X_H + 0.75 * (1.0 - X_H))
r""" float: The mean molecular mass given the cosmological hydrogen abundance :math:`X_H` and ignoring metals.

.. math::

    \mu = \frac{1}{\sum_j (j+1)X_j/A_j}

"""
mue: float = 1.0 / (X_H + 0.5 * (1.0 - X_H))
r""" float: The mean molecular mass per free electron in a fully ionized primordial plasma

.. math::

    \mu_e = \frac{\rho}{m_p n_e} = \frac{1}{\sum_j j X_j/A_j}

"""


def rotate(
    vector: NDArray["Number"],
    axis: NDArray["Number"] | str | int,
    angle: "Number",
    rotation_matrix: NDArray["Number"] = None,
) -> tuple[NDArray["Number"], NDArray["Number"]]:
    """Rotate the vector ``vector`` about the axis ``axis`` by ``angle`` degrees.

    Parameters
    ----------
    vector: array
        The vector to rotate.
    axis: array or str or int
        The axis to rotate about. If an array is provided, then it is assumed to be the components of a vector along the
        rotation axis. If a letter or number is provided, it will be interpretted as the name of the intended axis (``0,1,2``) or (``x,y,z``).
    angle: float
        The angle by which to rotate. By convention, rotation is positive when it corresponds to CCW rotation about
        the axis.
    rotation_matrix: array, optional
        The rotation matrix to use.

        .. hint ::

            If provided, this will save the overhead of having to compute it from scratch. Useful for serial calculations.

    Returns
    -------
    array
        The rotation matrix.

        .. hint ::

            In cases where speed may be critical, the rotation matrix should simply be retained from the first calculation and
            then used as just ``rot_matrix@vector``. This will then skip the overhead of regenerating the rotation matrix each time.

    array
        The rotated vector.

    Examples
    --------

    Let's rotate the :math:`x` axis 90 degrees around the :math:`z` axis.

    >>> axis = [0,0,1]
    >>> vector = [1,0,0]
    >>> m,v = rotate(vector,axis,np.pi/2)
    >>> v
    array([[6.123234e-17],
           [1.000000e+00],
           [0.000000e+00]])

    You can also specify the axis as follows:

    >>> axis = 'z'
    >>> m,v = rotate(vector,axis,np.pi/2)
    >>> v
    array([[6.123234e-17],
           [1.000000e+00],
           [0.000000e+00]])

    or also

    >>> axis = 2
    >>> m,v = rotate(vector,axis,np.pi/2)
    >>> v
    array([[6.123234e-17],
           [1.000000e+00],
           [0.000000e+00]])

    The axis also doesn't have to normalized:

    >>> axis = [1,1,1]
    >>> vector = [0,0,1]
    >>> m,v = rotate(vector,axis,np.pi)
    >>> v
    array([[ 0.66666667],
           [ 0.66666667],
           [-0.33333333]])
    """
    if rotation_matrix is not None:
        return rotation_matrix, rotation_matrix @ vector

    # Managing dimensions and axis.
    if isinstance(axis, str):
        axis = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]
    elif isinstance(axis, int):
        axis = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}[axis]

    if len(vector) == 2:  # NDIM = 2, we add axes to everything.
        vector, axis = np.append(vector, 0), np.append(axis, 0)
        _axis_extended = True
    else:
        _axis_extended = False

    vector, axis = np.reshape(vector, (3, 1)), np.reshape(axis, (3, 1))

    # Constructing the rotation matrix #
    c, s = np.cos(angle), np.sin(angle)
    axis = axis / np.linalg.norm(axis)
    axss = axis**2

    rtm = np.array(
        [
            [
                c + axss[0] * (1 - c),
                axis[0] * axis[1] * (1 - c) - axis[2] * s,
                axis[0] * axis[2] * (1 - c) + axis[1] * s,
            ],
            [
                axis[0] * axis[1] * (1 - c) + axis[2] * s,
                c + axss[1] * (1 - c),
                axis[1] * axis[2] * (1 - c) - axis[0] * s,
            ],
            [
                axis[0] * axis[2] * (1 - c) - axis[1] * s,
                axis[1] * axis[2] * (1 - c) + axis[0] * s,
                c + axss[2] * (1 - c),
            ],
        ]
    ).reshape(3, 3)

    if _axis_extended:
        rtm = rtm[:-1, :-1]
        vector = vector[:-1]

    return rtm, rtm @ vector
