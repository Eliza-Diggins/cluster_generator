"""
Gravitation module for cluster generator.

Navigating This Module
======================

The :py:mod:`gravity` module is one of the more complex regions of the cluster generator base code and, due to this, can be
a little tricky to navigate. The following notes should be kept in mind while seeking for information regarding the module.

- The module is designed based on a class hierarchy. Each gravity class inherits from an abstract class :py:class:`gravity.Gravity`, which
  is effectively a scaffold. Different groups of gravitational theories also have scaffold classes. Properties / methods which should be
  shared between similar gravity theories are found in their progenitor class.
- Each gravity class comprises 3 core methods: :py:meth:`gravity.Gravity.compute_potential`, :py:meth:`gravity.Gravity.compute_gravitational_field`, :py:meth:`gravity.Gravity.compute_dynamical_mass`. Additional
  methods are simply structural needs to build those core methods.
- Different gravity theories may require additional data. This can either be supplied to the class by way of a class method, or through the configuration.
"""
from abc import ABC, abstractmethod
from functools import wraps

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import fsolve
from unyt import unyt_array

from cluster_generator.utils import G, cgparams, ensure_ytquantity, integrate, mylog


def _get_method(class_method):
    """Checks the gravity calculation approach for a valid method and selects it."""
    function_name = class_method.__name__

    @wraps(class_method)
    def method_specific(cls, fields, *args, **kwargs):
        """Wrapper"""
        out_field = function_name.replace("compute_", "")
        if out_field in fields:
            mylog.info(
                f"Skipping call to {function_name} because {out_field} already in provided fields. [gravity={cls.name}]."
            )
        if "method" in kwargs:
            method = kwargs["method"]
            del kwargs["method"]
        else:
            method = None

        if method is None:
            for k, v in cls._method_requirements[function_name].items():
                if all(i in fields for i in v):
                    method = k

            if method is None:  # failed to find a valid method.
                raise ValueError("No valid method was found.")
        else:
            if any(
                i not in fields for i in cls._method_requirements[function_name][method]
            ):
                missing = [
                    i
                    for i in cls._method_requirements[function_name][method]
                    if i not in fields
                ]
                raise ValueError(
                    f"Failed to execute {function_name} in {cls.name} gravity with method {method} because one of {cls._method_requirements['compute_potential'][method]}({missing}) was not found."
                )

        mylog.info(f"Executing {function_name} [method={method}, gravity={cls.name}]")
        return class_method(cls, fields, *args, method=method, **kwargs)

    return method_specific


class Gravity(ABC):
    """
    Abstract class structure / template class for gravitational theory classes.

    .. warning::

        This is an archetypal class and cannot be used by cluster models. If you are trying to use a 'default' gravity,
        see :py:class:`gravity.Newtonian`. This class is only meaningful for developers.

    .. rubric:: Gravity Theories Inherited From This Scaffold

    .. py:currentmodule:: gravity

    .. autosummary::
        :template: class.rst

        Classical
        Mondian
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

    @staticmethod
    def _initialization_decorator(initialization_method):
        """Wraps the model initialization methods to allow for post processing."""
        function_name = initialization_method.__name__

        @wraps(initialization_method)
        def wrapper(cls, *args, **kwargs):
            model = initialization_method(cls, *args, **kwargs)

            if "gravity" in kwargs:
                return get_gravity_class(kwargs["gravity"])._post_process_model(
                    model, function_name
                )
            else:
                return model

        return wrapper

    @classmethod
    def _post_process_model(cls, model, initialization_method):
        """Blank template class"""
        return model

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

    # Forward and backward interpolation functions may be provided (only 1 required) to ease the computation of numerical
    # inverses.
    #

    _forward_interpolation_function = cgparams["gravity"]["AQUAL"][
        "interpolation_function"
    ]  # <-- AQUAL interpolation function.
    _backward_interpolation_function = cgparams["gravity"]["QUMOND"][
        "interpolation_function"
    ]  # <-- QUMOND interpolation function.
    _default_interpolation_direction = None

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

    @classmethod
    def interpolation_function(cls, x):
        r"""
        Computes the relevant interpolation function values for the argument ``x``.

        Parameters
        ----------
        x: :py:class:`numpy.ndarray`
            A 1-D array of numerical values of which to evaluate the interpolation function.

        Returns
        -------
        y: :py:class:`numpy.ndarray`
            A 1-D array of the resulting output values.

        Notes
        -----
        In all of the classical MOND theories, the interpolation function (which we generally denote as :math:`\Sigma` either
        maps the Newtonian gravitational field to the MONDian field (the "backward convention") or it maps the MONDian field to
        the Newtonian (the "forward convention"). In what follows, we adopt the forward convention.

        In spherical symmetry, the MONDian gravitational field obeys an equation of the form

        .. math::

            \nabla \Psi = \nabla \Phi \Sigma\left(\frac{\nabla \Phi}{a_0}\right).

        More generally, :math:`\xi = \tau \Sigma(\tau)`. It is often necessary to invert this equation so that

        .. math::

            \tau = \xi \Sigma^\dagger(\xi),

        where the :math:`\dagger` is referred to here as the Milgromian inverse.

        In most of the classical MOND theories, a Milgromian inverse is computed numerically unless the user provides the
        correct inverse function.

        See Also
        --------
        :py:meth:`gravity.Mondian.inverse_interpolation_function`,:py:meth:`gravity.Mondian.set_inverse_interpolation_function`,:py:meth:`gravity.Mondian.set_interpolation_function`

        """
        if cls._default_interpolation_direction == "forward":
            return cls._forward_interpolation_function(x)
        elif cls._default_interpolation_direction == "backward":
            return cls._backward_interpolation_function(x)
        else:
            raise ValueError(
                f"The hidden attribute `_default_interpolation_direction` must be either forward or backward, not {cls._default_interpolation_direction}"
            )

    @classmethod
    def inverse_interpolation_function(cls, y):
        r"""
        Computes the relevant inverse interpolation function values for the argument ``y``.

        Parameters
        ----------
        y: :py:class:`numpy.ndarray`
            A 1-D array of numerical values of which to evaluate the interpolation function.

        Returns
        -------
        x: :py:class:`numpy.ndarray`
            A 1-D array of the resulting output values.

        Notes
        -----
        In all of the classical MOND theories, the interpolation function (which we generally denote as :math:`\Sigma` either
        maps the Newtonian gravitational field to the MONDian field (the "backward convention") or it maps the MONDian field to
        the Newtonian (the "forward convention"). In what follows, we adopt the forward convention.

        In spherical symmetry, the MONDian gravitational field obeys an equation of the form

        .. math::

            \nabla \Psi = \nabla \Phi \Sigma\left(\frac{\nabla \Phi}{a_0}\right).

        More generally, :math:`\xi = \tau \Sigma(\tau)`. It is often necessary to invert this equation so that

        .. math::

            \tau = \xi \Sigma^\dagger(\xi),

        where the :math:`\dagger` is referred to here as the Milgromian inverse.

        In most of the classical MOND theories, a Milgromian inverse is computed numerically unless the user provides the
        correct inverse function.

        See Also
        --------
        :py:meth:`gravity.Mondian.interpolation_function`,:py:meth:`gravity.Mondian.set_inverse_interpolation_function`,:py:meth:`gravity.Mondian.set_interpolation_function`

        """
        if cls._default_interpolation_direction == "forward":
            inv_func = cls._backward_interpolation_function
            forw_func = cls._forward_interpolation_function
            archetypal_equation = lambda x: (0.5 * x) * (
                1 + np.sqrt(1 + (4 / x))
            )  # Archetypal guess function
        elif cls._default_interpolation_direction == "backward":
            inv_func = cls._forward_interpolation_function
            forw_func = cls._backward_interpolation_function
            archetypal_equation = lambda x: (x**2) / (1 + x)
        else:
            raise ValueError(
                f"The hidden attribute `_default_interpolation_direction` must be either forward or backward, not {cls._default_interpolation_direction}"
            )

        if inv_func is not None:
            # The inverse function actually exists, we can simply evaluate
            return inv_func(y)

        else:
            # We have to solve the problem numerically.
            numerical_function = lambda x: x * forw_func(x) - y

            return fsolve(numerical_function, archetypal_equation(y)) / y

    @classmethod
    def set_interpolation_function(cls, function):
        """
        Set the interpolation function for the gravity class.

        Parameters
        ----------
        function: callable
            The function to set the interpolation function to.

        """
        if cls._default_interpolation_direction == "forward":
            cls._forward_interpolation_function = function
        elif cls._default_interpolation_direction == "backward":
            cls._backward_interpolation_function = function
        else:
            raise ValueError(
                f"The hidden attribute `_default_interpolation_direction` must be either forward or backward, not {cls._default_interpolation_direction}"
            )

    @classmethod
    def set_inverse_interpolation_function(cls, function):
        """
        Set the interpolation function's Milgromian inverse for the gravity class.

        Parameters
        ----------
        function: callable
            The function to set the interpolation function to.

        """
        if cls._default_interpolation_direction == "forward":
            cls._backward_interpolation_function = function
        elif cls._default_interpolation_direction == "backward":
            cls._forward_interpolation_function = function
        else:
            raise ValueError(
                f"The hidden attribute `_default_interpolation_direction` must be either forward or backward, not {cls._default_interpolation_direction}"
            )


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

    _backward_interpolation_function = cgparams["gravity"]["QUMOND"][
        "interpolation_function"
    ]
    _forward_interpolation_function = cgparams["gravity"]["QUMOND"]["milgrom_inv"]
    _default_interpolation_direction = "backward"

    @classmethod
    @_get_method
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
    @_get_method
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
        if method == 1:
            # -- Compute from Gauss' Law and spherical symmetry -- #
            scaled_field = -fields["gravitational_field"] / cls.get_a0()
            sf_signs = np.sign(scaled_field)

            scaled_field = np.abs(scaled_field)

            roots = scaled_field * cls.inverse_interpolation_function(scaled_field)

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
    @_get_method
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
        if method == 1:
            # -- Compute from Gauss' Law and spherical symmetry -- #
            newtonian_field = Newtonian.compute_gravitational_field(
                fields, method=1
            )  # pull from newtonian calcs.

            fields[
                "gravitational_field"
            ] = newtonian_field * cls._backward_interpolation_function(
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

    _forward_interpolation_function = cgparams["gravity"]["AQUAL"][
        "interpolation_function"
    ]
    _backward_interpolation_function = cgparams["gravity"]["AQUAL"]["milgrom_inv"]
    _default_interpolation_direction = "forward"

    @classmethod
    @_get_method
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
    @_get_method
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
        if method == 1:
            fields["total_mass"] = (
                -(fields["radius"] ** 2 * fields["gravitational_field"]) / G
            ) * cls._forward_interpolation_function(
                np.abs((fields["gravitational_field"] / cls.get_a0()).d)
            )
        else:
            raise ValueError(
                f"Method {method} for computing dynamical mass in {cls.name} doesn't exist."
            )

        return fields["total_mass"]

    @classmethod
    @_get_method
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
        if method == 1:
            # -- Compute from Gauss' Law and spherical symmetry -- #
            newtonian_scaled_field = (
                -(G * fields["total_mass"]) / (cls.get_a0() * fields["radius"] ** 2)
            ).d
            nsf_signs = np.sign(newtonian_scaled_field)

            newtonian_scaled_field = np.abs(newtonian_scaled_field)

            roots = newtonian_scaled_field * cls.inverse_interpolation_function(
                newtonian_scaled_field
            )

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
        "compute_potential": {
            1: ("radius", "gravitational_field"),
            2: ("radius", "total_mass"),
        },
        "compute_dynamical_mass": {1: ()},
        "compute_gravitational_field": {
            1: ("radius", "total_mass"),
            2: ("radius", "gravitational_potential"),
        },
        "compute_A0": {
            1: ("gravitational_field", "radius", "gas_mass"),
            2: ("gravitational_field", "radius", "density"),
        },
    }

    _interpolation_function = cgparams["gravity"]["EMOND"]["interpolation_function"]
    _inverse_interpolation_function = cgparams["gravity"]["EMOND"]["milgrom_inv"]
    _default_interpolation_direction = "forward"
    _A0 = cgparams["gravity"]["EMOND"]["a0_function"]
    _gauge = cgparams["gravity"]["EMOND"]["default_gauge"]

    @classmethod
    def gauge_point(cls):
        r"""
        The gauge point of the EMOND paradigm. This property dictates the boundary conditions for potential computation.

        Returns
        -------
        tuple
            The gauge point ``(a,b)``, where ``a`` is the factor and ``b`` is the overdensity size.
            As an example, if the boundary should be at :math:`5r_{500}`, then the gauge point will be ``(5,500)``.

        Notes
        -----
        The gauge point is generically represented as :math:`Ar_{B}`, where :math:`r_B` is a particular overdensity radius (:math:`r_{200},r_{500},\cdots`).
        By setting a standard gauge in terms of the overdensity size, the potential can be standardized across systems in a consistent way. This is critical
        in EMOND because the paradigm is gauge dependent.

        .. admonition:: Definition

            The overdensity radius :math:`r_{\chi}` is defined in the context of EMOND as the radius for which the **Newtonian** acceleration
            of the system is equivalent to that of a sphere of uniform density :math:`\chi \rho_{\mathrm{crit}}` at that radius.

        Given that the overdensity mass is

        .. math::

            M_\chi = \frac{4\pi}{3} \chi \rho_{\mathrm{crit}} r_\chi^3,

        The resulting gravitational field (in a Newtonian paradigm) is

        .. math::

            \nabla \Phi_{\chi} = \frac{GM_\chi}{r_\chi^2} = \frac{4\pi G \rho_{\mathrm{crit}} \chi}{3} r_\chi = \nabla \Phi_{\chi}

        Thus,

        .. math::

            \Sigma(r_\chi) = \frac{\nabla \Phi(r\chi)}{G\rho_{\mathrm{crit}} r_\chi} = \frac{4\pi}{3}\chi.

        .. note::

            In the event that :math:`r_\chi` cannot be found within the domain, we assume that

            .. math::

                \nabla \Phi(r) = \frac{r_{\mathrm{end}}^2}{r^2}\nabla \Phi(r_{\mathrm{end}})

            and compute the relevant radius from that prescription.

        """
        return cls._gauge

    @classmethod
    def set_gauge_point(cls, value):
        """
        Sets the gauge point property.

        Parameters
        ----------
        value: tuple
            The gauge point ``(a,b)``, where ``a`` is the factor and ``b`` is the overdensity size.

        Returns
        -------
        None

        """
        cls._gauge = value

    @classmethod
    def find_gauge_radius(
        cls, fields, z=0, cosmology=cgparams["physics"]["default_cosmology"]
    ):
        r"""
        Computes the relevant radius of the gauge point for the given set of fields.

        Parameters
        ----------
        fields: dict of str: :py:class:`unyt.array.unyt_array`
            The dictionary of :py:class:`model.ClusterModel` fields to use for the calculation.
        z: float, optional
            The redshift of the system from which to calculate. Default is 0.
        cosmology: :py:class:`yt.Cosmology`, optional
            The cosmology to use in the calculation.

        Returns
        -------
        :py:class:`unyt.array.unyt_array`
            The radius of the over-density position.

        Notes
        -----
        The overdensity is computed either from the total mass profile, total density profile, or the potential profile in
        that order of precedence. In order, the resulting formulations are

        - **Total Mass**: Given that the average density is just

          .. math::

                \bar{\rho} = \frac{M(<r)}{(4/3)\pi r^3} \implies \chi(r) = \frac{3M(<r)}{4\pi r^3 \rho_{\mathrm{crit}}}

          The relevant overdensity specified by the :py:meth:`gravity.EMOND.gauge_point` is then located from :math:`\chi(r)`. If :math:`\chi(r)` does not
          extend sufficiently low over the provided radius, then it is assumed that the density is truncated at the largest radius and thus

          .. math::

            \chi(r > r_0) = \chi(r_0) \left(\frac{r_0}{r}\right)^3 \implies r = r_0 \left(\frac{\chi_0}{\chi}\right)^{1/3}

        - **Density**: If the total density is provided, a similar approach is carried out using a cumulative average to determine average density.
        - **Gravitational Field**: If the gravitational field is provided, then the corresponding Newtonian mass is

          .. math::

                M = \frac{r^2}{G}\nabla \Phi,

          and the procedure may progress as it did in the first method.

          .. warning::

                It should be noted here that, when it concerns the gauge of EMOND systems, we **always** set the gauge
                relative to the over-densities implied by a Newtonian interpretation. This choice is valid because it is not the
                underlying cosmology that is important, only the sufficient homogeneity of the gauge choices between systems.

        """
        from yt.utilities.cosmology import Cosmology

        if cosmology is None:
            cosmology = Cosmology()
        elif isinstance(cosmology, str):
            try:
                from colossus.cosmology import cosmo

                cosmology = cosmo.setCosmology(cosmology)

            except ImportError:
                raise ValueError(
                    "A custom cosmology was detected, but Colossus was not installed."
                )

        rho_crit = cosmology.critical_density(z).to_value("Msun/kpc**3")

        if all(
            i not in fields
            for i in ["gravitational_field", "total_mass", "total_density"]
        ):
            raise ValueError(
                "One of ('gravitational_field','total_mass','total_density') is required to compute a gauge radius, but no such fields were found."
            )

        if "total_mass" in fields:
            average_density = (3 * fields["total_mass"]) / (
                4 * np.pi * fields["radius"] ** 3
            )
        elif "total_density" in fields:
            average_density = np.cumsum(fields["total_density"]) / np.arange(
                1, len(fields["total_density"])
            )
        else:
            average_density = -(3 * fields["gravitational_field"]) / (
                4 * G * np.pi * fields["radius"]
            )

        _gauge_scale, _gauge_factor = cls.gauge_point()

        chi = average_density.to_value("Msun/kpc**3") / rho_crit  # overdensity factor.
        if np.amin(chi) < _gauge_factor < np.amax(chi):
            # The gauge factor falls within the computed range.
            r_base = fields["radius"][
                np.where(
                    np.abs(chi - _gauge_factor) == np.amin(np.abs(chi - _gauge_factor))
                )
            ]

            return _gauge_scale * r_base

        else:
            # The gauge factor is not clearly attainable.
            return (
                _gauge_scale
                * fields["radius"][-1]
                * (chi[-1] / _gauge_factor) ** (1 / 3)
            )

    @classmethod
    def A0(cls, potential):
        r"""
        Evaluates the :math:`A_0(\Phi)` function associated with the gravity implementation.

        Parameters
        ----------
        potential: :py:class:`unyt.array.unyt_array`
            The potential array on which to evaluate :math:`A_0(\Phi)`.

        Returns
        -------
        :py:class:`unyt.array.unyt_array`
            The calculated acceleration threshold.

        """
        if cls._A0 is None:
            raise ValueError(
                "There is no A0 yet associated with this gravity implementation."
            )
        else:
            return cls._A0(potential)

    @classmethod
    def set_A0(cls, function):
        r"""
        Sets the :math:`A_0(\Phi)` function for the gravity implementation.

        Parameters
        ----------
        function: callable
            The function to assign to the :math:`A_0(\Phi)` role.

        Returns
        -------
        None

        """
        cls._A0 = function

    @classmethod
    @_get_method
    def compute_A0(cls, fields, method=None):
        r"""
        Computes the :math:`A_0(\Phi)` function for which the cluster requires no dark matter content in the EMOND paradigm.

        Parameters
        ----------
        fields: dict of str: :py:class:`unyt.array.unyt_array`
            The dictionary of :py:class:`model.ClusterModel` fields to use for the calculation.
        method: int
            The method number to use. Defaults to ``None``, which will cause the algorithm to seek the first allowable
            method in the list.

        Returns
        -------
        callable
            The function :math:`A_0(\phi)`.

        Notes
        -----
        This computation can only be accomplished if :math:`\rho_{\mathrm{gas}}` (or some suitable proxy thereof) is known as well
        as either :math:`\Phi` or :math:`\nabla \Phi`. The algorithm will attempt to identify opportunities to use the provided fields
        to obtain these quantities; however, if insufficient information is provided, an error will be raised.

        Raises
        ------
        ValueError
            The algorithm failed to identify a reasonable path by which to compute the function.

        """
        if method == 1:  # using the field, radius, and gas mass.
            # forcing the field in this case to be monotone decreasing
            if np.any(fields["gravitational_field"].d > 0):
                mylog.warning(
                    "The gravitational field has non-negative elements. Implies non-monotone potential. Calculating best monotone estimate of A0."
                )
                temp_field = np.minimum.accumulate(
                    fields["gravitational_field"].copy()[::-1]
                )[::-1]
                temp_pot = cls.compute_potential(
                    {"radius": fields["radius"], "gravitational_field": temp_field}
                )
            else:
                temp_field = fields["gravitational_field"].copy()
                if "gravitational_potential" not in fields:
                    fields["gravitational_potential"] = cls.compute_potential(
                        fields, method=1
                    )
                temp_pot = fields["gravitational_potential"].d

            newtonian_field = -fields["gas_mass"] * G / fields["radius"] ** 2
            field_ratio = np.abs(
                newtonian_field.to_value("kpc/Myr**2")
                / temp_field.to_value("kpc/Myr**2")
            )
            field_ratio[np.where(field_ratio > 0.75)] = 0.75
            numerical_function = (
                lambda x: cls._interpolation_function(np.abs(temp_field.d) / x)
                - field_ratio
            )
            guess = np.abs(temp_field.d) * (1 - field_ratio) / field_ratio

            roots = fsolve(numerical_function, guess)
            _A0_uncorrected = unyt_array(roots, temp_field.units)
            func = InterpolatedUnivariateSpline(
                temp_pot.to_value("kpc**2/Myr**2"),
                _A0_uncorrected.to_value("kpc/Myr**2"),
            )

            return func
        elif method == 2:  # using the field, radius, and density
            from cluster_generator.utils import integrate_mass

            rr = fields["radius"].to_value("kpc")
            dfunc = InterpolatedUnivariateSpline(
                rr, fields["density"].to_value("Msun/kpc**3")
            )
            fields["gas_mass"] = unyt_array(integrate_mass(dfunc, rr), "Msun")
            return cls.compute_A0(fields, method=1)
        else:
            raise ValueError(
                f"Method {method} for computing A0 in {cls.name} doesn't exist."
            )

    @classmethod
    @_get_method
    def compute_potential(cls, fields, method=None, **kwargs):
        r"""
        Computes the EMOND gravitational potential :math:`\Phi` from the provided ``fields``.

        Parameters
        ----------
        fields: dict of str: :py:class:`unyt.array.unyt_array`
            The dictionary of :py:class:`model.ClusterModel` fields to use for the calculation.
        method: int
            The method number to use. Defaults to ``None``, which will cause the algorithm to seek the first allowable
            method in the list.
        **kwargs:
            Additional keyword arguments to be passed through the :py:meth:`gravity.EMOND.find_gauge_radius`.

        Returns
        -------
        dict of str: :py:class:`unyt.array.unyt_array`
            a new copy of the fields with the added data from the computation.

        Notes
        -----
        .. warning::

            The EMOND paradigm differs from other paradigms in that it is highly gauge dependent. Before generating a cluster,
            you should always check the gauge settings (:py:attr:`gravity.EMOND.gauge`).

        - **Method 1**: (``gravitational_field``, ``radius``) Computes the gravitational potential directly from the field. Unlike in
          more classical paradigms, the gauge is explicitly set in this approach. If the gauge point (:math:`r_\star`) falls within the
          ``radius`` array, then a simple correction is made to adjust the gauge as necessary. Otherwise, the field is extrapolated to the
          necessary radius under the classical Milgromian presumption that :math:`\nabla \Phi \sim r^{-1}` for sufficiently low accelerations.

          Once extrapolation and gauge correction occur, the potential is simply given by

          .. math::

                \Phi = -\int_r^{r_\star} \nabla \Phi dr = \int_{r}^{r_\star} \textbf{g} dr.

        .. warning::

            Pathological gauge points, particularly when placed well within the Newtonian domain of the system will lead to
            highly erroneous results. It is suggested that a cosmological gauge be chosen on the order of :math:`r_{200}` or
            further out.

            It may be worthwhile to compute the cluster in AQUAL or QUMOND first to gauge the state of play for selecting the boundary
            condition.

        - **Method 2**: (``total_mass``, ``radius``) Method 2 uses the :math:`A_0(\Phi)` function to construct and solve a differential
          equation for the potential. Formally,

          .. math::

                \mu\left(\frac{|\nabla \Phi|}{A_0(\Phi(r))}\right) \nabla \Psi = \nabla \Psi = \frac{G M_{\mathrm{dyn}}(<r)}{r^2}

          Letting :math:`\omega = |\nabla \Phi|/A_0(\Phi(r))` and :math:`\kappa = |\nabla \Psi|/A_0(\Phi(r))`,

          .. math::

                \mu(\omega)\omega = \kappa \implies \omega = \kappa \mu^\dagger\left(\kappa\right)= \kappa \eta\left(\kappa\right),

          Where :math:`\mu^\dagger` is the **Milgromian Inverse**. This yields the following non-linear, 1st order ODE

          .. math::

                \frac{\partial \Phi}{\partial r} = \nabla \Psi \eta\left(\frac{|\nabla \Psi|}{A_0(\Phi(r))}\right).

          This equation is then solved numerically using an RK67 algorithm.

        See Also
        --------
        :py:meth:`gravity.Newtonian.compute_gravitational_field`,:py:meth:`gravity.Newtonian.compute_dynamical_mass`

        """
        gauge_radius = cls.find_gauge_radius(fields, **kwargs).to_value("kpc")
        rr_base = fields["radius"].to_value("kpc")
        if method == 1:  # Compute from the gravitational field and radius.
            if gauge_radius > rr_base[-1]:
                # the radii need to be extended. Determine the frequency of points then extrapolate.
                point_density = (np.log10(rr_base[-1] / rr_base[0])) / len(rr_base)
                rr_extension = np.geomspace(
                    rr_base[-1],
                    gauge_radius,
                    int(np.ceil(point_density * np.log10(gauge_radius / rr_base[-1]))),
                )[1:]

                rr = unyt_array(
                    np.concatenate([rr_base, rr_extension]), fields["radius"].units
                ).to_value("kpc")
                gf = unyt_array(
                    np.concatenate(
                        [
                            fields["gravitational_field"].to_value("kpc/Myr**2"),
                            fields["gravitational_field"].to_value("kpc/Myr**2")[-1]
                            * (fields["radius"][-1] / rr_extension),
                        ]
                    )
                )
            else:
                rr = rr_base
                gf = fields["gravitational_field"].to_value("kpc/Myr**2")

            # Potential can now be constructed on extended domain.
            field_function = InterpolatedUnivariateSpline(rr, gf)
            fields["gravitational_potential"] = unyt_array(
                integrate(field_function, fields["radius"].d), "kpc**2/Myr**2"
            )
            r_beyond_gauge = rr[np.where(rr > gauge_radius)]

            if len(r_beyond_gauge):
                # correcting for gauge within the domain
                fields["gravitational_potential"] -= fields["gravitational_potential"][
                    np.where(rr > gauge_radius)
                ][0]

        elif method == 2:
            from scipy.integrate import solve_ivp

            # Compute from total mass and radius.
            assert (
                cls._inverse_interpolation_function is not None
            ), f"Method 2 for computing potential from {cls.name} requires an explicit milgromian inverse."
            assert (
                cls._A0 is not None
            ), f"Failed to find the acceleration function for {cls.name}. A0 must be manually specified to initialize by this method."

            newtonian_field = (-G / fields["radius"] ** 2) * fields["total_mass"]
            newtonian_field = newtonian_field.to_value("kpc/Myr**2")
            newtonian_field_spline = InterpolatedUnivariateSpline(
                fields["radius"].d, newtonian_field
            )
            d_function = lambda r, phi: newtonian_field_spline(
                r
            ) * cls._inverse_interpolation_function(
                np.abs(newtonian_field_spline(r)) / cls._A0(phi)
            )
            r_span = (fields["radius"].d[0], fields["radius"].d[-1])

            phi_solved = solve_ivp(d_function, r_span, [0], t_eval=fields["radius"].d)
            assert (
                phi_solved.success
            ), f"Failed to solve the differential equation for potential, message={phi_solved.message}"
            fields["gravitational_potential"] = -unyt_array(
                phi_solved.y[0] - phi_solved.y[0, -1], "kpc**2/Myr**2"
            )
            # Compute the gauge transformation necessary.
            if gauge_radius < r_span[1]:
                fields["gravitational_potential"] -= fields["gravitational_potential"][
                    np.where(fields["radius"].to_value("kpc") > gauge_radius)
                ][0]
            else:
                fields["gravitational_potential"] -= (
                    G * fields["total_mass"][-1] / fields["radius"][-1]
                ) * np.log(gauge_radius / fields["radius"][-1])

        else:
            raise ValueError(
                f"Method {method} for computing potential in {cls.name} doesn't exist."
            )

        return fields["gravitational_potential"]

    @classmethod
    @_get_method
    def compute_dynamical_mass(cls, fields, method=None):
        r"""
        Computes the EMOND dynamical mass :math:`M_{\mathrm{dyn}}(<r)` from the provided ``fields``.

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
        from cluster_generator.utils import integrate_mass

        if method == 1:
            if "gravitational_potential" not in fields:
                fields["gravitational_potential"] = cls.compute_potential(fields)

            if cls._A0 is None:
                rho_g_func = InterpolatedUnivariateSpline(
                    fields["radius"].d, fields["density"].to_value("Msun/kpc**3")
                )
                fields["total_mass"] = unyt_array(
                    integrate_mass(rho_g_func, fields["radius"].d), "Msun"
                )
                return fields["total_mass"]

            fields["total_mass"] = (
                -(fields["radius"] ** 2 * fields["gravitational_field"]) / G
            ) * cls._forward_interpolation_function(
                np.abs(
                    (
                        fields["gravitational_field"].d
                        / cls._A0(
                            fields["gravitational_potential"].to_value("kpc**2/Myr**2")
                        )
                    )
                )
            )
        else:
            raise ValueError(
                f"Method {method} for computing dynamical mass in {cls.name} doesn't exist."
            )
        return fields["total_mass"]

    @classmethod
    @_get_method
    def compute_gravitational_field(cls, fields, method=None):
        r"""
        Computes the EMOND gravitational field :math:`-\nabla \Phi` from the provided ``fields``.

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
        if method == 1:
            if "gravitational_potential" not in fields:
                fields["gravitational_potential"] = cls.compute_potential(fields)
            return cls.compute_gravitational_field(fields, method=2)
        if method == 2:
            # -- Compute from Gauss' Law and spherical symmetry -- #
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

    @classmethod
    def _post_process_model(cls, model, initialization_method):
        if initialization_method == "from_dens_and_temp":
            # clean up the machine epsilon error in the dm density.
            model["dark_matter_density"] = unyt_array(
                np.zeros(model["radius"].size), "Msun/kpc**3"
            )
        return model


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
    @_get_method
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
        if method == 1:
            field_function = InterpolatedUnivariateSpline(
                fields["radius"].d, fields["gravitational_field"].to("kpc/Myr**2").d
            )
            fields["gravitational_potential"] = unyt_array(
                integrate(field_function, fields["radius"].d), "kpc**2/Myr**2"
            )
            # add the gauge correction
            fields["gravitational_potential"] += (
                fields["gravitational_field"][-1] * fields["radius"][-1]
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
    @_get_method
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
    @_get_method
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
