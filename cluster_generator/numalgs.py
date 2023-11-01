"""
Numerical algorithms for use in the backend of CG.
"""
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline
from unyt import unyt_array

from cluster_generator.utils import integrate, mp, mu, mylog


def identify_domain_borders(array, domain=None):
    """
    Identify the edges of the domains specified in the array.

    Parameters
    ----------
    array: :py:class:`numpy.ndarray`
        Array (1D) containing ``1`` indicating truth and ``2`` indicating false from which to obtain the boundaries.
    domain: :py:class:`numpy.ndarray`, optional
        The domain of consideration (x-values) to mark the boundaries instead of using indices.

    Returns
    -------
    list
        List of 2-tuples containing the boundary indices (if ``domain==None``) or the boundary positions if domain is specified.

    """
    boundaries = (
        np.concatenate([[-1], array[:-1]]) + array + np.concatenate([array[1:], [-1]])
    )
    if domain is None:
        ind = np.indices(array).reshape((array.size,))
        vals = ind[np.where(boundaries == 1)]
    else:
        vals = domain[np.where(boundaries == 1)]

    return vals.reshape(len(vals) // 2, 2)


def _check_non_positive(array, domain=None):
    o = np.zeros(array.size)
    o[np.where(array < 0)] = 1
    return identify_domain_borders(o, domain=domain)


def find_holes(x, y, rtol=1e-3, dy=None):
    """
    Locates holes (points of non-monotonicity) in the profile defined by ``y`` over ``x``.

    Parameters
    ----------
    x: array_like
        The domain array on which to base the hole identification process.
    y: array_like
        The profile to find non-monotonicities in.

        .. warning::

            If the profile is not "nearly monotone increasing" (i.e. ``y[0] >=y[-1]``), then this method will fail.
            If your profile is the wrong way around, simply pass ``y[::-1]`` instead.

    rtol: :obj:`float`, optional
        The hole identification tolerance. Default is ``1e-3``.
    dy: :obj:`callable`, optional
        The derivative function for the array. If it is not provided, a central difference scheme will be used to generate
        the secant slopes.

    Returns
    -------
    n: int
        The number of identified holes.
    h: :py:class:`numpy.ndarray`
        Size ``(3,n,2)`` array containing the following:

        - hx: Array of size ``(n,2)`` with the minimum and maximum ``x`` for each hole respectively.
        - hy: Array of size ``(n,2)`` with the left and right ``y`` value on each side of every hole.
        - hi: Array of size ``(n,2)`` with the left and right indices of the hole.

    Notes
    -----
    To locate holes in the profile, this algorithm takes the cumulative maximum of the profile and compares it with the
    true profile. In locations where the true profile falls below the cumulative maximum, a hole is identified.
    """
    _x, _y = x[:], y[:]

    if dy is None:
        secants = np.gradient(_y, _x)
    else:
        secants = dy(_x)

    holes = np.zeros(_x.size)
    ymax = np.maximum.accumulate(_y)
    holes[~np.isclose(_y, ymax, rtol=rtol)] = 1
    holes[secants <= -1e-8] = 1

    # construct boundaries of holes
    _hb = (
        np.concatenate([[holes[0]], holes])[:-1]
        + holes
        + np.concatenate([holes, [holes[-1]]])[1:]
    )
    ind = np.indices(_x.shape).reshape((_x.size,))

    hx, hy, hi = _x[np.where(_hb == 1)], _y[np.where(_hb == 1)], ind[np.where(_hb == 1)]

    if holes[0] == 1:
        hx, hy, hi = (
            np.concatenate([[_x[0]], hx]),
            np.concatenate([[hy[0]], hy]),
            np.concatenate([[0], hi]),
        )
    if holes[-1] == 1:
        hx, hy, hi = (
            np.concatenate([hx, [_x[-1]]]),
            np.concatenate([hy, [hy[-1]]]),
            np.concatenate([hi, [ind[-1]]]),
        )

    return len(hx) // 2, np.array(
        [
            hx.reshape(len(hx) // 2, 2),
            hy.reshape(len(hy) // 2, 2),
            hi.reshape(len(hi) // 2, 2),
        ]
    )


def monotone_interpolation(x, y, buffer=10, rtol=1e-3):
    """
    Monotone interpolation scheme based on piecewise cubic spline methods. Holes are "patched" in such a way that
    the profile is monotone, continuously differentiable (however not 2x continuously differentiable).

    Parameters
    ----------
    x: array-like
        The domain over which interpolation should occur.
    y: array-like
        The relevant profile to force into monotonicity.
    buffer: :obj:`int`, optional
        The forward step buffer. Default is ``10``. Value must be an integer larger than 0.
    rtol: :obj:`float`, optional
        The relative tolerance to enforce on hole identification.

    Returns
    -------
    x: :py:class:`numpy.ndarray`
        The corresponding, newly interpolated, domain.
    y: :py:class:`numpy.ndarray`
        The interpolated solution array.

    Notes
    -----
    Methodology was developed by Eliza Diggins (University of Utah) based on the work of [1]_ and [2]_.

    .. [1] Frisch, F. N. and Carlson, R. E. 1980SJNA...17..238F
    .. [2] Steffen, M. 1990A&A...239..443S

    """
    from scipy.interpolate import CubicHermiteSpline

    if y[-1] > y[0]:
        monotonicity = 1
        _x, _y = x[:], y[:]
    elif y[0] > y[-1]:
        monotonicity = -1
        _x, _y = x[:], y[::-1]
    else:
        mylog.warning(
            "Attempted to find holes in profile with no distinct monotonicity."
        )
        return None

    nholes, holes = find_holes(_x, _y, rtol=rtol)
    derivatives = np.gradient(_y, _x, edge_order=2)

    while nholes > 0:
        # carry out the interpolation over the hole.
        hxx, hyy, hii = holes[:, 0, :]

        # building the interpolant information
        hii[1] = hii[1] + np.min(
            np.concatenate(
                [[buffer, len(_x) - 1 - hii[1]], (holes[2, 1:, 0] - hii[1]).ravel()]
            )
        )
        hii = np.array(hii, dtype="int")
        hyy = [_y[hii[0]], np.amax([_y[hii[1]], _y[hii[0]]])]
        hxx = [_x[hii[0]], _x[hii[1]]]

        if hii[1] == len(_x) - 1:
            print(np.amax(_y))
            _y[hii[0] : hii[1] + 1] = _y[hii[0]]
            print(_y[-10:], hxx, hyy, hii)
            input()
        else:
            xb, yb = hxx[1] - (hyy[1] - hyy[0]) / (2 * derivatives[hii[1]]), (1 / 2) * (
                hyy[0] + hyy[-1]
            )
            s = [(yb - hyy[0]) / (xb - hxx[0]), (hyy[1] - yb) / (hxx[1] - xb)]
            p = (s[0] * (hxx[1] - xb) + (s[1] * (xb - hxx[0]))) / (hxx[1] - hxx[0])
            xs = [hxx[0], xb, _x[hii[-1]]]
            ys = [hyy[0], yb, _y[hii[-1]]]
            dys = [0.0, np.amin([2 * s[0], 2 * s[1], p]), derivatives[hii[1]]]

            cinterp = CubicHermiteSpline(xs, ys, dys)
            _y[hii[0] : hii[1]] = cinterp(_x[hii[0] : hii[1]])

        nholes, holes = find_holes(_x, _y, rtol=rtol)

    if monotonicity == -1:
        _x, _y = _x[:], _y[::-1]

    return _x, _y


def positive_interpolation(x, y, correction_parameter, buffer=10, rtol=1e-3, maxit=10):
    """
    A positive interpolation scheme, which fills "holes" which drop below 0 with 2 piecewise monotone cubics with zero slope at
    the hole center.

    Parameters
    ----------
    x: array-like
        The domain over which interpolation should occur.
    y: array-like
        The relevant profile to force into monotonicity.
    correction_parameter: float
        The correction parameter is a float from 0 to 1 which determines the degree of monotonicity to insist on. If ``correction_parameter == 1``, then
        monotone interpolation will be carried out. If ``correction_parameter == 0``, then the minimum of the function will be at zero over the hole.
    buffer: :obj:`int`, optional
        The step buffer. Default is ``10``. Value must be an integer larger than 0.
    rtol: :obj:`float`, optional
        The relative tolerance to enforce on hole identification.
    maxit: :obj:`int`, optional
        The maximum number of allowed iterations during which an interpolation interval can be found. Default is 10.

    Returns
    -------
    x: :py:class:`numpy.ndarray`
        The corresponding, newly interpolated, domain.
    y: :py:class:`numpy.ndarray`
        The interpolated solution array.

    Notes
    -----
    Methodology was developed by Eliza Diggins (University of Utah) based on the work of [1]_ and [2]_.

    .. [1] Frisch, F. N. and Carlson, R. E. 1980SJNA...17..238F
    .. [2] Steffen, M. 1990A&A...239..443S

    """
    from scipy.interpolate import CubicHermiteSpline

    if correction_parameter == 1:
        return monotone_interpolation(x, y, buffer=buffer, rtol=rtol)

    if y[-1] > y[0]:
        monotonicity = 1
        _x, _y = x[:].copy(), y[:].copy()
    elif y[0] > y[-1]:
        monotonicity = -1
        _x, _y = x[:].copy(), y[::-1].copy()
    else:
        monotonicity = 0
        _x, _y = x[:].copy(), y[:].copy()

    nholes, holes = find_holes(_x, _y, rtol=rtol)
    derivatives = np.gradient(_y, _x, edge_order=2)

    for hid in range(nholes):
        hole = holes[:, hid, :]
        hxx, hyy, hii = hole

        if hii[0] == 0:
            hyy[0] = np.amax([0, hyy[1] / 2])
            derivatives[0] = 0
        n = 0
        check = False
        while (n < maxit) and not check:
            if (monotonicity == 0) or n > 0:
                hii[1] = hii[1] + np.min(
                    np.concatenate(
                        [
                            [buffer, len(_x) - 1 - hii[1]],
                            (holes[2, hid + 1 :, 0] - hii[1]).ravel(),
                        ]
                    )
                )
                hii[0] = hii[0] - np.min(
                    np.concatenate(
                        [[buffer, hii[0]], hii[0] - (holes[2, :hid, 1]).ravel()]
                    )
                )
            else:
                hii[1] = hii[1] + np.min(
                    np.concatenate(
                        [
                            [buffer, len(_x) - 1 - hii[1]],
                            (holes[2, hid + 1 :, 0] - hii[1]).ravel(),
                        ]
                    )
                )
            hii = np.array(hii, dtype="int")
            hyy = [np.amax([_y[hii[0]], 0]), np.amax([_y[hii[1]], 0])]
            hxx = [_x[hii[0]], _x[hii[1]]]

            xb, yb = np.mean(hxx), np.amax([np.mean(hyy), 0]) * correction_parameter
            s = [(yb - hyy[0]) / (xb - hxx[0]), (hyy[1] - yb) / (hxx[1] - xb)]

            if (np.abs(derivatives[hii[0]]) < np.abs(3 * s[0])) and (
                np.abs(derivatives[hii[-1]]) < np.abs(3 * s[1])
            ):
                check = True
            else:
                n += 1

        xs = [hxx[0], xb, hxx[-1]]
        ys = [hyy[0], yb, hyy[-1]]
        dys = [derivatives[hii[0]], 0, derivatives[hii[1]]]

        cinterp = CubicHermiteSpline(xs, ys, dys)
        _y[hii[0] : hii[1]] = cinterp(_x[hii[0] : hii[1]])

    if n >= maxit:
        raise ValueError(
            f"Failed to find a viable interpolation domain within {maxit} iterations."
        )

    if monotonicity == -1:
        _x, _y = _x[:], _y[::-1]

    return _x, _y


def solve_temperature(r, potential_gradient, density):
    """
    Solves the temperature equation from the potential gradient and the gas density.

    Parameters
    ----------
    r: unyt_array
        The radius profile
    potential_gradient: unyt_array
        The potential gradient profile.
    density: unyt_array
        The gas density profile.

    Returns
    -------
    unyt_array:
        The computed temperature profile.

    """
    g = potential_gradient.in_units("kpc/Myr**2").v
    d = density.in_units("Msun/kpc**3").v
    rr = r.in_units("kpc").d
    g_r = InterpolatedUnivariateSpline(rr, g)
    d_r = InterpolatedUnivariateSpline(rr, d)

    dPdr_int = lambda r: d_r(r) * g_r(r)
    P = -integrate(dPdr_int, rr)
    dPdr_int2 = lambda r: d_r(r) * g[-1] * (rr[-1] / r) ** 2
    P -= quad(dPdr_int2, rr[-1], np.inf, limit=100)[0]
    pressure = unyt_array(P, "Msun/kpc/Myr**2")
    temp = pressure * mu * mp / density
    temp.convert_to_units("keV")
    return temp


def extrap_power_law(x0, x1, alpha, f=None, df=None, x=None, y=None, sign=1):
    r"""
    Determines an extrapolation function which gives the provided data (either ``f`` or ``y``) a particular asymptotic power-law
    behavior which maintaining :math:`C^1[\mathbb{R}]` behavior at ``x_0``, and which is bounded (above, ``sign==1``) or (below, ``sign==-1``)
    by the equivalent power-law passing through ``(x1,y1)``. If a callable ``f`` and its derivative ``df`` are provided, then a :py:class:`radial_profiles.RadialProfile` is returned
    which is piecewise defined to give the correct asymptotic behavior. If array data is provided (``x`` and ``y``), then
    the arrays ``x`` and ``y`` will be returned with the corresponding alterations.

    Parameters
    ----------
    x0: float
        The extrapolation point which dictates the position at which the function transitions to the extrapolation function.
        ``x0 < x1`` is required.
    x1: float
        The vertex position. The asymptotic power law :math:`g(x) = y_1(x/x_1)^\alpha` passes through this point
        and is equivalent to the output function for sufficiently large :math:`x`.
    alpha: float
        The corresponding power law index to match to during extrapolation.
    f: callable, optional
        Either ``f`` or ``x`` and ``y`` must be specified. A callable function which provides the pre-asymptotic adjustment
        behavior of the profile.
    df: callable, optional
        If ``f`` is provided, ``df`` must also be provided. Provides the derivative of ``f`` at each ``x``.
    x: array-like, optional
        The ``x`` values of the function domain.
    y: array-like, optional
        The ``y`` values of the function domain.
    sign: int, optional
        (default ``1``) The sign of the extrapolation behavior. If ``1``, then the extrapolation behavior is bounded entirely
        above the corresponding power law through ``x_1,y_1``. Otherwise may be ``-1`` and the extrapolation behavior will
        be entirely below the corresponding power law.

    Returns
    -------
    callable or :py:class:`numpy.ndarray`

    Raises
    ------
    ValueError:
        If :math:`\omega \le 0`.

    Notes
    -----

    This algorithm is designed to take some function :math:`f(x)` which has some (presumably undesireable) behavior beyond
    some :math:`x_1` and map it (for :math:`x>x_1`) to a new function :math:`\tilde{f}(x) = \xi(x)E_{\pm}(x)` such that for some :math:`x_0<x_1`,
    :math:`\tilde{f}(x_0) = f(x_0)` and :math:`\tilde{f}'(x_0) = f'(x_0)` and :math:`\xi(x) = y_1(x/x_1)^\alpha`. Here, :math:`E_\pm` is the extrapolation function
    and is (depending on user provided ``sign``) one of

    .. math::

        E_{\pm}(x;x_0) = \frac{\left(\frac{x}{x_0}\right)^\omega}{\left(\frac{x}{x_0}\right)^\omega \mp \gamma}.

    Because at :math:`x=x_0`,

    .. math::

        \xi(x_0)E_{\pm}(x_0,x_0) = y_1\left(\frac{x_0}{x_1}\right)^\alpha \frac{1}{1\pm \gamma} = f(x_0) \implies \gamma = \mp \left(\frac{y_1}{y_0}\left(\frac{x_0}{x_1}\right)^\alpha - 1\right)

    and

    .. math::

        \tilde{f}' = \frac{\xi(x_0)}{x_0(1\mp \gamma)^2}\left[\alpha(1\pm \gamma) \mp \gamma \omega \right] = y_0'

    yielding

    .. math::

        \omega = \frac{\mp 1}{\gamma}\left[\frac{x_0y_0'(1\mp \gamma)^2}{\xi(x_0)} - \alpha(1\mp \gamma)\right]

    For valid results, :math:`\omega > 0` is required.

    Examples
    --------

    In some cases, radial profiles are not well suited for large :math:`r`. Take for example the :py:class:`radial_profiles.vikhlinin_temperature_profile` for A133;
    it falls off far too quickly at large radii (beyond the regime where it was fit to data). Thus, to use such a profile, some correction must be done.

    .. plot::
        :include-source:

        from cluster_generator.radial_profiles import vikhlinin_temperature_profile
        from cluster_generator.numalgs import extrap_power_law
        import numpy as np
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1,1)
        T = vikhlinin_temperature_profile(3.61,0.12,5.00,10.0,1420,0.9747,57,3.88)
        x = np.geomspace(1,5000,5000)
        y = T(x)

        ax.semilogx(x,y,"k:")

        xn,yn = extrap_power_law(700,800,-2,x=x,y=y)

        ax.semilogx(xn,yn,"r--")

        plt.show()

    """
    assert x0 < x1, "x0 < x1 for extrapolated power law algorithm to be viable."

    if (f is not None) and (df is not None):
        mode = "functional"
    elif (x is not None) and (y is not None):
        mode = "pointwise"
    else:
        raise ValueError(
            "Could not find either (f and df) or (x and y) for computation."
        )

    assert sign in [1, -1], "The sign parameter must be either +1 or -1."

    # Determine necessary data
    if mode == "functional":
        y0, y1, dy0 = f(x0), f(x1), df(x0)
    else:
        inds = np.indices(x.shape).reshape((x.size,))
        x0, x1 = (
            x[np.where(np.abs(x - x0) == np.amin(np.abs(x - x0)))],
            x[np.where(np.abs(x - x1) == np.amin(np.abs(x - x1)))],
        )
        i0, i1 = inds[np.where(x == x0)][0], inds[np.where(x == x1)][0]
        y0, y1 = y[i0], y[i1]
        dy0 = (y0 - y[i0 + 1]) / (x0 - x[i0 + 1])

    gamma = -sign * ((y1 / y0) * (x0 / x1) ** (alpha) - 1)
    omega = (-sign / gamma) * (
        ((x0 * dy0 * (1 - sign * gamma) ** 2) / (y1 * (x0 / x1) ** alpha))
        - (alpha * (1 - sign * gamma))
    )

    f_interp = lambda x: (y1 * (x / x1) ** alpha) * (
        ((x / x0) ** omega) / ((x / x0) ** omega - sign * gamma)
    )

    if mode == "pointwise":
        x_, y_ = x.copy(), y.copy()
        y_[i0:] = f_interp(x_[i0:])
        return x_, y_
    else:
        return lambda t, b=x0, q=f: np.piecewise(
            x, [x < b, x >= b], [q(t), f_interp(t)]
        )


def _closest_factors(val):
    assert isinstance(val, int), "Value must be integer."

    a, b, i = 1, val, 0

    while a < b:
        i += 1
        if val % i == 0:
            a = i
            b = val // a

    return (a, b)
