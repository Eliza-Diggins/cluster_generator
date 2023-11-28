"""
Module containing the core gravitational classes for Cluster Generator.
"""
from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from unyt import unyt_array

from cluster_generator.utils import G, cgparams, ensure_ytquantity, integrate


class Gravity(ABC):
    """
    Abstract class structure for gravitational theory classes.

    .. warning::

        This is an archetypal class and cannot be used by cluster models. If you are trying to use a 'default' gravity,
        see :py:class:`gravity.Newtonian`.

    .. rubric:: Gravity Theories Inherited From This Scaffold

    .. py:currentmodule:: gravity

    .. autosummary::
        :template: class.rst

        Classical
        MONDian
        EMOND
        AQUAL
        QUMOND
        Newtonian

    """

    _method_requirements = {}

    @property
    @abstractmethod
    def name(self):
        """Name of the gravitational theory. This is the name that is recognized as the string name."""
        pass

    @property
    @abstractmethod
    def type(self):
        """The type of gravitational theory represented by this class."""
        pass

    @abstractmethod
    def get_virialization_method(self):
        """The preferred virialization method for the given system."""
        pass

    @abstractmethod
    def compute_potential(self, *args, **kwargs):
        """
        Computes the potential from provided fields.

        Returns
        -------
        :py:class:`unyt.unyt_array`
            The resulting potential array.

        """
        pass

    @abstractmethod
    def compute_gravitational_field(self, *args, **kwargs):
        """
        Computes the gravitational field from fields.

        Returns
        -------
        :py:class:`unyt.unyt_array`
            The resulting field array.

        """
        pass

    @abstractmethod
    def compute_dynamical_mass(self, *args, **kwargs):
        """
        Computes the dynamical mass from provided fields.

        Returns
        -------
        :py:class:`unyt.unyt_array`
            The resulting mass array.

        """
        pass

    @classmethod
    def _get_subclasses(cls):
        # --> we actually want to recursively obtain subclasses
        _out = []
        if len(cls.__subclasses__()) == 0:
            _out.append(cls)

        else:
            for subclass in cls.__subclasses__():
                _out += subclass._get_subclasses()

        return _out

    @classmethod
    def _subclasses(cls):
        return {k.name: k for k in cls._get_subclasses()}

    @classmethod
    def _determine_best_method(cls, fields, method):
        for k, v in cls._method_requirements[method].items():
            if all(i in fields for i in v):
                return k

        raise ValueError("No valid method was found.")

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __class_getitem__(cls, item):
        return cls._subclasses()[item]


class Classical(Gravity, ABC):
    """
    Abstract class representation for classical and semi-classical gravity representations.

    .. warning::

        This is an archetypal class and cannot be used by cluster models. If you are trying to use a "default" gravity,
        see :py:class:`gravity.Newtonian`.

    .. rubric:: Gravity Theories Inherited From This Scaffold

    .. py:currentmodule:: gravity

    .. autosummary::
        :template: class.rst

        Newtonian
    """

    #: The name of the gravity class.
    name = "classical"
    #: The type of the gravity class.
    type = "classical"

    _virialization_method = cgparams["gravity"]["default_virialization"]

    @classmethod
    def get_virialization_method(cls):
        """
        Returns the default virialization method connected with this :py:class:`gravity.Gravity`-like class.

        .. note::

            To change the virialization method of a :py:class:`model.ClusterModel`, use the :py:meth:`model.ClusterModel.set_virialization_method` method.

        Returns
        -------
        str
            The virialization method.

        """
        return cls._virialization_method


class Mondian(Gravity, ABC):
    """
    Abstract class structure for MONDian gravity representations.

    .. warning::

        This is an archetypal class and cannot be used by cluster models. If you are trying to use a 'default' gravity,
        see :py:class:`gravity.AQUAL`.

    .. rubric:: Gravity Theories Inherited From This Scaffold

    .. py:currentmodule:: gravity

    .. autosummary::
        :template: class.rst

        EMOND
        AQUAL
        QUMOND

    """

    #: the gravity type for the class.
    type = "MONDian"
    #: The name of the gravity class.
    name = "mondian"

    _virialization_method = "LMA"
    _a_0 = cgparams["gravity"]["general"]["mond"]["a0"]  # <-- Fetch configuration value

    @classmethod
    def get_virialization_method(cls):
        """Fetches the virialization method for this implementation"""
        return cls._virialization_method

    @classmethod
    def set_a0(cls, value):
        """
        Set the characteristic acceleration scale for the MONDian theory (:math:`a_0`).

        Parameters
        ----------
        value: :py:class:`unyt.quantity`
            The value to use the parameter to.

        """
        cls._a_0 = ensure_ytquantity(value, "m/s**2")

    @classmethod
    def get_a0(cls):
        r"""
        Returns the characteristic acceleration scale for the theory (:math:`a_0`). The typical (literature) value is
        :math:`a_0 = 1.2\times10^{-10} \mathrm{m\;s^{-2}}`; however, users may use the ``.set_a0`` method to alter the value.

        .. note::

            The default value may be set in the ``config.yaml`` file.

        Returns
        -------
        :py:class:`unyt.unyt_quantity`
            The value of the :math:`a_0` parameter.

        """
        return cls._a_0

    @property
    @abstractmethod
    def interpolation_function(self, *args, **kwargs):
        """The MOND interpolation function specific to this gravity theory."""
        pass


class QUMOND(Mondian):
    r"""
    :py:class:`gravity.Gravity`-like representation of the Quasi-Linear MOND (QUMOND) formulation of classical MOND gravity [Milgrom10]_.

    .. math::

        \mathcal{L} = \frac{1}{2}\rho \textbf{v}^2 - \rho \Phi - \frac{1}{8\pi G}\left\{2\nabla \Phi \cdot \nabla \Psi - a_0^2 \mathcal{Q}\left(\frac{\nabla \Psi^2}{a_0^2}\right)\right\}

    References
    ----------
    .. [Milgrom10]  M. Milgrom (2010) Monthly Notices of the Royal Astronomical Society, Volume 403, Issue 2, pp. 886-895.

    """

    #: The name of the gravity class.
    name = "QUMOND"

    _method_requirements = {
        "compute_potential": {
            1: ("gravitational_field", "radius"),
            2: ("total_density", "total_mass", "radius"),
        },
        "compute_dynamical_mass": {1: ("gravitational_field", "radius")},
        "compute_gravitational_field": {
            1: ("total_mass", "radius"),
            2: ("gravitational_potential", "radius"),
        },
    }

    _interpolation_function = cgparams["gravity"]["QUMOND"]["interpolation_function"]

    @classmethod
    def interpolation_function(cls, x):
        r"""
        Evaluate the AQUAL interpolation function associated with this gravity instantiation. By default, this function
        is pulled from the ``config.yaml`` file; however it may be set with the :py:meth:`gravity.AQUAL.set_interpolation_function` method.

        Parameters
        ----------
        x: :py:class:`numpy.ndarray`
            The function input. This should be :math:`\nabla \Phi / a_0`.

            .. warning::

                This should be unitless!

        Returns
        -------
        :py:class:`numpy.ndarray`
            The function output.

        """
        return cls._interpolation_function(x)

    @classmethod
    def set_interpolation_function(cls, function):
        """Sets the interpolation function."""
        cls._interpolation_function = function

    @classmethod
    def compute_potential(cls, fields, method=None):
        r"""
        Computes the QUMOND gravitational potential :math:`\Phi` from the provided ``fields``.

        Parameters
        ----------
        fields: dict of str: :py:class:`unyt.array.unyt_array`
            The dictionary of :py:class:`model.ClusterModel` fields to use for the calculation.
        method: int
            The method number to use. Defaults to ``None``, which will cause the algorithm to seek the first allowable
            method in the list.

        Returns
        -------
        dict of str: :py:class:`unyt.array.unyt_array`
            a new copy of the fields with the added data from the computation.

        Notes
        -----
        - **Method 1**: (``gravitational_field``, ``radius``) Computes the gravitational potential directly from the field:

          .. math::

                \Phi = -\int_r^{r_0} \nabla \Phi dr = \int_{r}^{r_0} \textbf{g} dr.

        - **Method 2**: (``total_mass``,``radius``) Computes the gravitational potential from the total mass component. This
          is effectively a proxy for Method 1 wherein the total mass is used to find the Newtonian field, and the Newtonian field is
          then converted.

        See Also
        --------
        :py:meth:`gravity.Newtonian.compute_gravitational_field`,:py:meth:`gravity.Newtonian.compute_dynamical_mass`

        """
        if method is None:
            try:
                method = cls._determine_best_method(fields, "compute_potential")
            except ValueError:
                raise ValueError(
                    f"No method for gravitational dynamical mass calculation in gravity {cls.name} with fields {fields.keys()} was found."
                )
        else:
            if any(
                i not in fields
                for i in cls._method_requirements["compute_potential"][method]
            ):
                missing = [
                    i
                    for i in cls._method_requirements["compute_potential"][method]
                    if i not in fields
                ]
                raise ValueError(
                    f"Failed to compute dynamical mass in {cls.name} gravity with method {method} because one of {cls._method_requirements['compute_potential'][method]}({missing}) was not found."
                )

        if method == 1:
            field_function = InterpolatedUnivariateSpline(
                fields["radius"].d, fields["gravitational_field"].to("kpc/Myr**2").d
            )
            fields["gravitational_potential"] = unyt_array(
                integrate(field_function, fields["radius"].d), "kpc**2/Myr**2"
            )
        elif method == 2:
            # -- pull the field -- #
            fields["gravitational_field"] = cls.compute_gravitational_field(
                fields, method=1
            )

            field_function = InterpolatedUnivariateSpline(
                fields["radius"].d, fields["gravitational_field"].to("kpc/Myr**2").d
            )
            fields["gravitational_potential"] = unyt_array(
                integrate(field_function, fields["radius"].d), "kpc**2/Myr**2"
            )

        else:
            raise ValueError(
                f"Method {method} for computing potential in {cls.name} doesn't exist."
            )

        return fields["gravitational_potential"]

    @classmethod
    def compute_dynamical_mass(cls, fields, method=None):
        r"""
        Computes the QUMOND dynamical mass :math:`M_{\mathrm{dyn}}(<r)` from the provided ``fields``.

        Parameters
        ----------
        fields: dict of str: :py:class:`unyt.array.unyt_array`
            The dictionary of :py:class:`model.ClusterModel` fields to use for the calculation.
        method: int
            The method number to use. Defaults to ``None``, which will cause the algorithm to seek the first allowable
            method in the list.

        Returns
        -------
        :py:class:`unyt.array.unyt_array`
            The computed array data.

        Notes
        -----
        - **Method 1**: (``gravitational_field``, ``radius``) Computes the dynamical mass from the gravitational field.

          .. math::

                M_{\mathrm{dyn}}(<r) = \frac{r^2 \nabla \Phi}{G} \mu\left(\frac{|\nabla \Phi|}{a_0}\right)


        See Also
        --------
        :py:meth:`gravity.QUMOND.compute_potential`,:py:meth:`gravity.QUMOND.compute_gravitational_field`

        """
        if method is None:
            try:
                method = cls._determine_best_method(fields, "compute_dynamical_mass")
            except ValueError:
                raise ValueError(
                    f"No method for gravitational dynamical mass calculation in gravity {cls.name} with fields {fields.keys()} was found."
                )
        else:
            if any(
                i not in fields
                for i in cls._method_requirements["compute_dynamical_mass"][method]
            ):
                missing = [
                    i
                    for i in cls._method_requirements["compute_dynamical_mass"][method]
                    if i not in fields
                ]
                raise ValueError(
                    f"Failed to compute dynamical mass in {cls.name} gravity with method {method} because one of {cls._method_requirements['compute_dynamical_mass'][method]}({missing}) was not found."
                )
        if method == 1:
            from scipy.optimize import fsolve

            # -- Compute from Gauss' Law and spherical symmetry -- #
            scaled_field = -fields["gravitational_field"] / cls.get_a0()
            sf_signs = np.sign(scaled_field)

            scaled_field = np.abs(scaled_field)

            guess_roots = scaled_field**2 / (1 + scaled_field)  # inversion

            _optimization_function = (
                lambda x: cls._interpolation_function(x) * x - scaled_field
            )
            if np.allclose(_optimization_function(guess_roots), 0, rtol=1e-7):
                # We don't need to waste time on the numerical solver, this is close enough.
                roots = guess_roots
            else:
                roots = fsolve(_optimization_function, guess_roots)

            newtonian = sf_signs * (
                cls.get_a0() * roots
            )  # sign correction and rescaling.

            # Computing the actual mass
            fields["total_mass"] = newtonian * (fields["radius"] ** 2 / G)
        else:
            raise ValueError(
                f"Method {method} for computing dynamical mass in {cls.name} doesn't exist."
            )

        return fields["total_mass"]

    @classmethod
    def compute_gravitational_field(cls, fields, method=None):
        r"""
        Computes the QUMOND gravitational field :math:`-\nabla \Phi` from the provided ``fields``.

        Parameters
        ----------
        fields: dict of str: :py:class:`unyt.array.unyt_array`
            The dictionary of :py:class:`model.ClusterModel` fields to use for the calculation.
        method: int
            The method number to use. Defaults to ``None``, which will cause the algorithm to seek the first allowable
            method in the list.

        Returns
        -------
        :py:class:`unyt.array.unyt_array`
            The computed array data.

        Notes
        -----
        - **Method 1**: (``total_mass``, ``radius``) In Method 1, spherical symmetry is exploited. The Newtonian field is
          calculated (via a direct call to :py:meth:`gravity.Newtonian.compute_gravitational_field`) and the resulting field is then
          used directly to determine the corresponding physical field via

          .. math::

                \nabla \Phi = \nabla \Psi \eta\left(\frac{|\nabla \Psi|}{a_0}\right).

        - **Method 2**: (``gravitational_potential``) Computes the gravitational field as the gradient of the potential.

          .. math::

                \textbf{g} = - \nabla \Phi.

        See Also
        --------
        :py:meth:`gravity.QUMOND.compute_potential`,:py:meth:`gravity.QUMOND.compute_dynamical_mass`

        """
        if method is None:
            try:
                method = cls._determine_best_method(
                    fields, "compute_gravitational_field"
                )
            except ValueError:
                raise ValueError(
                    f"No method for gravitational field calculation in gravity {cls.name} with fields {fields.keys()} was found."
                )
        else:
            if any(
                i not in fields
                for i in cls._method_requirements["compute_gravitational_field"][method]
            ):
                raise ValueError(
                    f"Failed to compute gravitational field in {cls.name} gravity with method {method} because one of {cls._method_requirements['compute_gravitational_field'][method]} was not found."
                )

        if method == 1:
            # -- Compute from Gauss' Law and spherical symmetry -- #
            newtonian_field = Newtonian.compute_gravitational_field(
                fields, method=1
            )  # pull from newtonian calcs.

            fields[
                "gravitational_field"
            ] = newtonian_field * cls._interpolation_function(
                np.abs((fields["gravitational_field"] / cls.get_a0()).d)
            )
            fields["gravitational_field"].convert_to_units("kpc/Myr**2")

        elif method == 2:
            # Compute by taking the gradient
            potential_spline = InterpolatedUnivariateSpline(
                fields["radius"].to("kpc").d,
                fields["gravitational_potential"].to("kpc**2/Myr**2").d,
            )
            fields["gravitational_field"] = -unyt_array(
                potential_spline(fields["radius"].to("kpc").d, 1), "kpc/Myr**2"
            )

        else:
            raise ValueError(
                f"Method {method} for computing gravitational field in {cls.name} doesn't exist."
            )

        return fields["gravitational_field"]


class AQUAL(Mondian):
    r"""
    :py:class:`gravity.Gravity`-like representation of the Aquadtratic Lagrangian MOND (AQUAL) formulation of classical MOND gravity [MiBe84]_.

    .. math::

        \mathcal{L} = \frac{1}{2}\rho \textbf{v}^2 - \rho \Phi - \frac{a_0^2}{8\pi G} \mathcal{F}\left(\frac{|\nabla \Phi|^2}{a_0^2}\right)

    References
    ----------
    .. [MiBe84]  Astrophysical Journal, Part 1 (ISSN 0004-637X), vol. 286, Nov. 1, 1984, p. 7-14. Research supported by the MINERVA Foundation.


    """

    #: The name of the gravity class.
    name = "AQUAL"

    _method_requirements = {
        "compute_potential": {
            1: ("gravitational_field", "radius"),
            2: ("total_density", "total_mass", "radius"),
        },
        "compute_dynamical_mass": {1: ("gravitational_field", "radius")},
        "compute_gravitational_field": {
            1: ("total_mass", "radius"),
            2: ("gravitational_potential", "radius"),
        },
    }

    _interpolation_function = cgparams["gravity"]["AQUAL"]["interpolation_function"]

    @classmethod
    def interpolation_function(cls, x):
        r"""
        Evaluate the AQUAL interpolation function associated with this gravity instantiation. By default, this function
        is pulled from the ``config.yaml`` file; however it may be set with the :py:meth:`gravity.AQUAL.set_interpolation_function` method.

        Parameters
        ----------
        x: :py:class:`numpy.ndarray`
            The function input. This should be :math:`\nabla \Phi / a_0`.

            .. warning::

                This should be unitless!

        Returns
        -------
        :py:class:`numpy.ndarray`
            The function output.

        """
        return cls._interpolation_function(x)

    @classmethod
    def set_interpolation_function(cls, function):
        """Sets the interpolation function."""
        cls._interpolation_function = function

    @classmethod
    def compute_potential(cls, fields, method=None):
        r"""
        Computes the AQUAL gravitational potential :math:`\Phi` from the provided ``fields``.

        Parameters
        ----------
        fields: dict of str: :py:class:`unyt.array.unyt_array`
            The dictionary of :py:class:`model.ClusterModel` fields to use for the calculation.
        method: int
            The method number to use. Defaults to ``None``, which will cause the algorithm to seek the first allowable
            method in the list.

        Returns
        -------
        dict of str: :py:class:`unyt.array.unyt_array`
            a new copy of the fields with the added data from the computation.

        Notes
        -----
        - **Method 1**: (``gravitational_field``, ``radius``) Computes the gravitational potential directly from the field:

          .. math::

                \Phi = -\int_r^{r_0} \nabla \Phi dr = \int_r^{r_0} \textbf{g} dr.

        - **Method 2**: (``total_mass``,``radius``) Computes the gravitational potential from the total mass component. This
          is effectively a proxy for Method 1 wherein the total mass is used to find the field field, and the field is
          then converted.

        See Also
        --------
        :py:meth:`gravity.Newtonian.compute_gravitational_field`,:py:meth:`gravity.Newtonian.compute_dynamical_mass`

        """
        if method is None:
            try:
                method = cls._determine_best_method(fields, "compute_potential")
            except ValueError:
                raise ValueError(
                    f"No method for gravitational dynamical mass calculation in gravity {cls.name} with fields {fields.keys()} was found."
                )
        else:
            if any(
                i not in fields
                for i in cls._method_requirements["compute_potential"][method]
            ):
                missing = [
                    i
                    for i in cls._method_requirements["compute_potential"][method]
                    if i not in fields
                ]
                raise ValueError(
                    f"Failed to compute dynamical mass in {cls.name} gravity with method {method} because one of {cls._method_requirements['compute_potential'][method]}({missing}) was not found."
                )

        if method == 1:
            field_function = InterpolatedUnivariateSpline(
                fields["radius"].d, fields["gravitational_field"].to("kpc/Myr**2").d
            )
            fields["gravitational_potential"] = unyt_array(
                integrate(field_function, fields["radius"].d), "kpc**2/Myr**2"
            )
        elif method == 2:
            # -- pull the field -- #
            fields["gravitational_field"] = cls.compute_gravitational_field(
                fields, method=1
            )

            field_function = InterpolatedUnivariateSpline(
                fields["radius"].d, fields["gravitational_field"].to("kpc/Myr**2").d
            )
            fields["gravitational_potential"] = unyt_array(
                integrate(field_function, fields["radius"].d), "kpc**2/Myr**2"
            )

        else:
            raise ValueError(
                f"Method {method} for computing potential in {cls.name} doesn't exist."
            )

        return fields["gravitational_potential"]

    @classmethod
    def compute_dynamical_mass(cls, fields, method=None):
        r"""
        Computes the AQUAL dynamical mass :math:`M_{\mathrm{dyn}}(<r)` from the provided ``fields``.

        Parameters
        ----------
        fields: dict of str: :py:class:`unyt.array.unyt_array`
            The dictionary of :py:class:`model.ClusterModel` fields to use for the calculation.
        method: int
            The method number to use. Defaults to ``None``, which will cause the algorithm to seek the first allowable
            method in the list.

        Returns
        -------
        :py:class:`unyt.array.unyt_array`
            The computed array data.

        Notes
        -----
        - **Method 1**: (``gravitational_field``, ``radius``) Computes the dynamical mass from the gravitational field.

          .. math::

                M_{\mathrm{dyn}}(<r) = \frac{r^2 \nabla \Phi}{G} \mu\left(\frac{|\nabla \Phi|}{a_0}\right)


        See Also
        --------
        :py:meth:`gravity.AQUAL.compute_potential`,:py:meth:`gravity.AQUAL.compute_gravitational_field`

        """
        if method is None:
            try:
                method = cls._determine_best_method(fields, "compute_dynamical_mass")
            except ValueError:
                raise ValueError(
                    f"No method for gravitational dynamical mass calculation in gravity {cls.name} with fields {fields.keys()} was found."
                )
        else:
            if any(
                i not in fields
                for i in cls._method_requirements["compute_dynamical_mass"][method]
            ):
                missing = [
                    i
                    for i in cls._method_requirements["compute_dynamical_mass"][method]
                    if i not in fields
                ]
                raise ValueError(
                    f"Failed to compute dynamical mass in {cls.name} gravity with method {method} because one of {cls._method_requirements['compute_dynamical_mass'][method]}({missing}) was not found."
                )
        if method == 1:
            fields["total_mass"] = (
                -(fields["radius"] ** 2 * fields["gravitational_field"]) / G
            ) * cls._interpolation_function(
                np.abs((fields["gravitational_field"] / cls.get_a0()).d)
            )
        else:
            raise ValueError(
                f"Method {method} for computing dynamical mass in {cls.name} doesn't exist."
            )

        return fields["total_mass"]

    @classmethod
    def compute_gravitational_field(cls, fields, method=None):
        r"""
        Computes the AQUAL gravitational field :math:`-\nabla \Phi` from the provided ``fields``.

        Parameters
        ----------
        fields: dict of str: :py:class:`unyt.array.unyt_array`
            The dictionary of :py:class:`model.ClusterModel` fields to use for the calculation.
        method: int
            The method number to use. Defaults to ``None``, which will cause the algorithm to seek the first allowable
            method in the list.

        Returns
        -------
        :py:class:`unyt.array.unyt_array`
            The computed array data.

        Notes
        -----
        - **Method 1**: (``total_mass``, ``radius``) Computes the gravitational field from spherical symmetry.
          The Bekenstein-Milgrom equation (modified Poisson equation) provides

          .. math::

                4\pi G \rho = \nabla \cdot \left[\mu\left(\frac{|\nabla \Phi|}{a_0}\right)\nabla \Phi\right],

          which, in turn, implies that (by Gauss' Law)

          .. math::

                \frac{GM_{\mathrm{dyn}}(<r)}{r^2} = \mu\left(\frac{|\nabla \Phi|}{a_0}\right)\nabla \Phi.

          We therefore construct :math:`\nabla \Phi` by numerically solving the non-linear equation above for the field.

        - **Method 2**: (``gravitational_potential``) Computes the gravitational field as the gradient of the potential.

          .. math::

                \textbf{g} = - \nabla \Phi.

        See Also
        --------
        :py:meth:`gravity.AQUAL.compute_potential`,:py:meth:`gravity.AQUAL.compute_dynamical_mass`

        """
        from scipy.optimize import fsolve

        if method is None:
            try:
                method = cls._determine_best_method(
                    fields, "compute_gravitational_field"
                )
            except ValueError:
                raise ValueError(
                    f"No method for gravitational field calculation in gravity {cls.name} with fields {fields.keys()} was found."
                )
        else:
            if any(
                i not in fields
                for i in cls._method_requirements["compute_gravitational_field"][method]
            ):
                raise ValueError(
                    f"Failed to compute gravitational field in {cls.name} gravity with method {method} because one of {cls._method_requirements['compute_gravitational_field'][method]} was not found."
                )

        if method == 1:
            # -- Compute from Gauss' Law and spherical symmetry -- #
            newtonian_scaled_field = (
                -(G * fields["total_mass"]) / (cls.get_a0() * fields["radius"] ** 2)
            ).d
            nsf_signs = np.sign(newtonian_scaled_field)

            newtonian_scaled_field = np.abs(newtonian_scaled_field)

            guess_roots = (newtonian_scaled_field / 2) * (
                1 + np.sqrt(1 + (4 / newtonian_scaled_field))
            )  # Best implementation to avoid catastrophic cancellation.

            _optimization_function = (
                lambda x: cls._interpolation_function(x) * x - newtonian_scaled_field
            )
            if np.allclose(_optimization_function(guess_roots), 0, rtol=1e-7):
                # We don't need to waste time on the numerical solver, this is close enough.
                roots = guess_roots
            else:
                roots = fsolve(_optimization_function, guess_roots)

            fields["gravitational_field"] = nsf_signs * (
                cls.get_a0() * roots
            )  # sign correction and rescaling.
            fields["gravitational_field"].convert_to_units("kpc/Myr**2")

        elif method == 2:
            # Compute by taking the gradient
            potential_spline = InterpolatedUnivariateSpline(
                fields["radius"].to("kpc").d,
                fields["gravitational_potential"].to("kpc**2/Myr**2").d,
            )
            fields["gravitational_field"] = -unyt_array(
                potential_spline(fields["radius"].to("kpc").d, 1), "kpc/Myr**2"
            )

        else:
            raise ValueError(
                f"Method {method} for computing gravitational field in {cls.name} doesn't exist."
            )

        return fields["gravitational_field"]


class EMOND(Mondian):
    r"""
    :py:class:`gravity.Gravity`-like representation of the Extended-MOND theory [ZhFa12]_. Implements additional degrees of
    freedom on the classical MOND theories by using a field Lagrangian of the form

    .. math::

        \mathcal{L} = \frac{1}{2}\rho v^2 - \rho \Phi - \frac{\Lambda}{8\pi G} F(x^2),\;\;\Lambda = A_0(\Phi)^2

    where

    .. math::

        x^2 = y = \frac{|\nabla \Phi|^2}{\Lambda}.

    Case studies include [HoZh17]_

    References
    ----------
    .. [ZhFa12] H. Zhao and B. Famaey (2012) Physical Review D, vol. 86, Issue 6, id. 067301
    .. [HoZh17] A. Hodson and H. Zhao (2017) Astronomy & Astrophysics, Volume 598, id.A127, 18 pp.

    """

    #: The name of the gravity class.
    name = "EMOND"

    _method_requirements = {
        "compute_potential": {1: ("total_density", "total_mass", "radius")},
        "compute_dynamical_mass": {1: ("gravitational_field", "radius")},
        "compute_gravitational_field": {
            1: ("total_mass", "radius"),
            2: ("gravitational_potential", "radius"),
        },
    }


class Newtonian(Classical):
    """
    The Newtonian gravity class.
    """

    #: The name of the gravity class.
    name = "Newtonian"

    _method_requirements = {
        "compute_potential": {
            1: ("gravitational_field", "radius"),
            2: ("total_density", "total_mass", "radius"),
        },
        "compute_dynamical_mass": {1: ("gravitational_field", "radius")},
        "compute_gravitational_field": {
            1: ("total_mass", "radius"),
            2: ("gravitational_potential", "radius"),
        },
    }

    @classmethod
    def compute_potential(cls, fields, method=None):
        r"""
        Computes the Newtonian gravitational potential :math:`\Phi` from the provided ``fields``.

        Parameters
        ----------
        fields: dict of str: :py:class:`unyt.array.unyt_array`
            The dictionary of :py:class:`model.ClusterModel` fields to use for the calculation.
        method: int
            The method number to use. Defaults to ``None``, which will cause the algorithm to seek the first allowable
            method in the list.

        Returns
        -------
        dict of str: :py:class:`unyt.array.unyt_array`
            a new copy of the fields with the added data from the computation.

        Notes
        -----
        - **Method 1**: (``total_mass``, ``radius``, ``total_density``) Computes the gravitational potential from spherical symmetry.

          .. math::

                \Phi = -4\pi G \left[\frac{1}{r}\int_0^r \rho(r')r'^2 dr' + \int_r^\infty \rho(r') r' dr' \right]

        See Also
        --------
        :py:meth:`gravity.Newtonian.compute_gravitational_field`,:py:meth:`gravity.Newtonian.compute_dynamical_mass`

        """
        if method is None:
            try:
                method = cls._determine_best_method(fields, "compute_potential")
            except ValueError:
                raise ValueError(
                    f"No method for gravitational dynamical mass calculation in gravity {cls.name} with fields {fields.keys()} was found."
                )
        else:
            if any(
                i not in fields
                for i in cls._method_requirements["compute_potential"][method]
            ):
                missing = [
                    i
                    for i in cls._method_requirements["compute_potential"][method]
                    if i not in fields
                ]
                raise ValueError(
                    f"Failed to compute dynamical mass in {cls.name} gravity with method {method} because one of {cls._method_requirements['compute_potential'][method]}({missing}) was not found."
                )

        if method == 1:
            field_function = InterpolatedUnivariateSpline(
                fields["radius"].d, fields["gravitational_field"].to("kpc/Myr**2").d
            )
            fields["gravitational_potential"] = unyt_array(
                integrate(field_function, fields["radius"].d), "kpc**2/Myr**2"
            )
        elif method == 2:
            rr = fields["radius"].d
            tdens_func = InterpolatedUnivariateSpline(rr, fields["total_density"].d)
            gpot_profile = lambda r: tdens_func(r) * r
            gpot1 = fields["total_mass"] / fields["radius"]
            gpot2 = unyt_array(4.0 * np.pi * integrate(gpot_profile, rr), "Msun/kpc")
            fields["gravitational_potential"] = -G * (gpot1 + gpot2)
            fields["gravitational_potential"].convert_to_units("kpc**2/Myr**2")
        else:
            raise ValueError(
                f"Method {method} for computing potential in {cls.name} doesn't exist."
            )

        return fields["gravitational_potential"]

    @classmethod
    def compute_dynamical_mass(cls, fields, method=None):
        r"""
        Computes the Newtonian dynamical mass :math:`M_{\mathrm{dyn}}(<r)` from the provided ``fields``.

        Parameters
        ----------
        fields: dict of str: :py:class:`unyt.array.unyt_array`
            The dictionary of :py:class:`model.ClusterModel` fields to use for the calculation.
        method: int
            The method number to use. Defaults to ``None``, which will cause the algorithm to seek the first allowable
            method in the list.

        Returns
        -------
        :py:class:`unyt.array.unyt_array`
            The computed array data.

        Notes
        -----
        - **Method 1**: (``gravitational_field``, ``radius``) Computes the dynamical mass from the gravitational field.

          .. math::

                M_{\mathrm{dyn}}(<r) = \frac{r^2 \nabla \Phi}{G}


        See Also
        --------
        :py:meth:`gravity.Newtonian.compute_potential`,:py:meth:`gravity.Newtonian.compute_gravitational_field`

        """
        if method is None:
            try:
                method = cls._determine_best_method(fields, "compute_dynamical_mass")
            except ValueError:
                raise ValueError(
                    f"No method for gravitational dynamical mass calculation in gravity {cls.name} with fields {fields.keys()} was found."
                )
        else:
            if any(
                i not in fields
                for i in cls._method_requirements["compute_dynamical_mass"][method]
            ):
                missing = [
                    i
                    for i in cls._method_requirements["compute_dynamical_mass"][method]
                    if i not in fields
                ]
                raise ValueError(
                    f"Failed to compute dynamical mass in {cls.name} gravity with method {method} because one of {cls._method_requirements['compute_dynamical_mass'][method]}({missing}) was not found."
                )
        if method == 1:
            fields["total_mass"] = (
                -(fields["radius"] ** 2 * fields["gravitational_field"]) / G
            )
        else:
            raise ValueError(
                f"Method {method} for computing dynamical mass in {cls.name} doesn't exist."
            )

        return fields["total_mass"]

    @classmethod
    def compute_gravitational_field(cls, fields, method=None):
        r"""
        Computes the Newtonian gravitational field :math:`-\nabla \Phi` from the provided ``fields``.

        Parameters
        ----------
        fields: dict of str: :py:class:`unyt.array.unyt_array`
            The dictionary of :py:class:`model.ClusterModel` fields to use for the calculation.
        method: int
            The method number to use. Defaults to ``None``, which will cause the algorithm to seek the first allowable
            method in the list.

        Returns
        -------
        :py:class:`unyt.array.unyt_array`
            The computed array data.

        Notes
        -----
        - **Method 1**: (``total_mass``, ``radius``) Computes the gravitational field from spherical symmetry.

          .. math::

                \textbf{g} = -\nabla \Phi = - \frac{GM(<r)}{r^2}.

        - **Method 2**: (``gravitational_potential``) Computes the gravitational field as the gradient of the potential.

          .. math::

                \textbf{g} = - \nabla \Phi.

        See Also
        --------
        :py:meth:`gravity.Newtonian.compute_potential`,:py:meth:`gravity.Newtonian.compute_dynamical_mass`

        """
        if method is None:
            try:
                method = cls._determine_best_method(
                    fields, "compute_gravitational_field"
                )
            except ValueError:
                raise ValueError(
                    f"No method for gravitational field calculation in gravity {cls.name} with fields {fields.keys()} was found."
                )
        else:
            if any(
                i not in fields
                for i in cls._method_requirements["compute_gravitational_field"][method]
            ):
                raise ValueError(
                    f"Failed to compute gravitational field in {cls.name} gravity with method {method} because one of {cls._method_requirements['compute_gravitational_field'][method]} was not found."
                )

        if method == 1:
            # Compute the field using Gauss' Law in spherical symmetry.
            fields["gravitational_field"] = (-G * fields["total_mass"]) / (
                fields["radius"] ** 2
            )
        elif method == 2:
            # Compute by taking the gradient
            potential_spline = InterpolatedUnivariateSpline(
                fields["radius"].to("kpc").d,
                fields["gravitational_potential"].to("kpc**2/Myr**2").d,
            )
            fields["gravitational_field"] = -unyt_array(
                potential_spline(fields["radius"].to("kpc").d, 1), "kpc/Myr**2"
            )

        else:
            raise ValueError(
                f"Method {method} for computing gravitational field in {cls.name} doesn't exist."
            )

        return fields["gravitational_field"]


def is_valid_gravity(name):
    """
    Checks if the ``name`` is a valid gravity theory.

    Parameters
    ----------
    name: str
        The name of the gravity class to seek out.

    Returns
    -------
    bool

    """
    try:
        _ = Gravity[name]
        return True
    except KeyError:
        return False


def get_gravity_class(name):
    """
    Fetches the gravity class with the name provided.

    Parameters
    ----------
    name: str
        The name of the gravitational class

    Returns
    -------
    :py:class:`gravity.Gravity`
        The corresponding gravity class if it exists.

    Raises
    ------
    ValueError
        If the gravity doesn't exist.

    """
    try:
        return Gravity[name]
    except KeyError:
        raise ValueError(f"Failed to find a gravitation class with name {name}.")


def gravity_operation(name, operation, *args, **kwargs):
    """
    Performs the ``operation`` method of the gravity class corresponding to ``name`` with associated args and kwargs.

    Parameters
    ----------
    name: str
        The name of the :py:class:`gravity.Gravity` class to seek out.
    operation: str
        The method name to execute.
    args:
        arguments to pass to the method.
    kwargs:
        kwargs to pass to the method.

    Returns
    -------
    Any

    """
    gclass = get_gravity_class(name)

    if hasattr(gclass, operation):
        return getattr(gclass, operation)(*args, **kwargs)
    else:
        raise ValueError(f"The operation {operation} is not a method of {name}.")


if __name__ == "__main__":
    print(get_gravity_class("Newtonian"))
