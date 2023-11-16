"""
Module containing the core gravitational classes for Cluster Generator.
"""
from abc import ABC, abstractmethod


class Gravity(ABC):
    """
    The archetypal gravitation class.

    .. warning::

        This is an archetypal class and cannot be used by cluster models. If you are trying to use a 'default' gravity,
        see :py:class:`gravity.Newtonian`.

    """

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

    name = "Newtonian"
    citation = "NA"

    @classmethod
    def compute_potential(cls, fields):
        """TODO"""
        pass

    @classmethod
    def compute_dynamical_mass(cls, fields):
        """TODO"""
        pass


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
    print(is_valid_gravity("newtonian"))
