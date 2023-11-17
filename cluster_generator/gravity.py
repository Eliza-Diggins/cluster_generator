"""
Module containing the core gravitational classes for Cluster Generator.
"""
from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from unyt import unyt_array

from cluster_generator.utils import G, integrate


class Gravity(ABC):
    """
    The archetypal gravitation class.

    .. warning::

        This is an archetypal class and cannot be used by cluster models. If you are trying to use a 'default' gravity,
        see :py:class:`gravity.Newtonian`.

    """

    _method_requirements = {}

    @property
    @abstractmethod
    def name(self):
        """Name of the gravitational theory. This is the name that is recognized as the string name."""
        pass

    @property
    @abstractmethod
    def citation(self):
        """The most relevant citation for the gravitational theory."""
        pass

    @property
    @abstractmethod
    def type(self):
        """The type of gravitational theory represented by this class."""
        pass

    @property
    @abstractmethod
    def virialization_method(self):
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
    Abstract archetypal class for classical gravitational theories.

    .. warning::

        This is an archetypal class and cannot be used by cluster models. If you are trying to use a 'default' gravity,
        see :py:class:`gravity.Newtonian`.
    """

    type = "classical"


class Mondian(Gravity, ABC):
    """
    Abstract archetypal class for Mondian gravitational theories.

    .. warning::

        This is an archetypal class and cannot be used by cluster models. If you are trying to use a 'default' gravity,
        see :py:class:`gravity.Newtonian`.
    """

    type = "mondian"


class Newtonian(Classical):
    """
    The Newtonian gravity class.
    """

    #: The name of the gravity class.
    name = "Newtonian"
    #: The citation for the gravity class if one exists.
    citation = "NA"

    _method_requirements = {
        "compute_potential": {1: ("total_density", "total_mass", "radius")},
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
        - **Method 1**: (``total_mass``, ``radius``) Computes the dynamical mass from the gravitational field.

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
