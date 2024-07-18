"""Special types and type hinting utilities."""
from numbers import Number
from typing import Any, Callable, Collection, Iterable, Mapping

import numpy as np
from more_itertools import always_iterable
from numpy._typing import ArrayLike
from numpy.typing import NDArray
from unyt import Unit, unyt_array, unyt_quantity

try:
    from typing import Self  # noqa
except ImportError:
    from typing_extensions import Self as Self  # noqa


NumericInput = Number | NDArray[Number]
MaybeUnitScalar = unyt_quantity | Number
MaybeUnitVector = unyt_array | Collection[Number]
PRNG = int | np.random.RandomState


class AttrDict(dict):
    """Attribute accessible dictionary."""

    def __init__(self, mapping: Mapping):
        super(AttrDict, self).__init__(mapping)
        self.__dict__ = self

        for key in self.keys():
            self[key] = self.__class__.from_nested_dict(self[key])

    @classmethod
    def from_nested_dict(cls, data: Any) -> Self:
        """Construct nested AttrDicts from nested dictionaries."""
        if not isinstance(data, dict):
            return data
        else:
            return AttrDict({key: cls.from_nested_dict(data[key]) for key in data})

    @classmethod
    def clean_types(cls, _mapping):
        for k, v in _mapping.items():
            if isinstance(v, AttrDict):
                _mapping[k] = cls.clean_types(_mapping[k])
            else:
                pass
        return dict(_mapping)

    def clean(self):
        return self.clean_types(self)


class Registry:
    """Registry utility class."""

    def __init__(self):
        self._mapping = AttrDict(
            {}
        )  # This is an empty attribute dict that contains the registry objects.

    def __getattr__(self, name: str) -> Any:
        try:
            super().__getattribute__(name)
        except AttributeError:
            return getattr(self._mapping, name).obj

    def __str__(self):
        return f"Registry[{len(self._mapping)} items]"

    def __repr__(self):
        return self.__str__()

    @property
    def meta(self):
        return self._mapping

    def register(self, name: str, obj: Any, overwrite: bool = False, **kwargs):
        """Register an entity in the registry.

        Parameters
        ----------
        name: str
            The name of the entity to register.
        obj: Any
            The object to register.
        overwrite: bool
            Allow the registration to overwrite existing entries.
        kwargs:
            Additional metadata to associate with the registered object.
        """
        from types import SimpleNamespace

        # Check that the overwriting is valid.
        if name in self._mapping:
            assert (
                overwrite
            ), f"Cannot set {name} in {self} because overwrite = False and it is already registered."

        self._mapping[name] = SimpleNamespace(obj=obj, **kwargs)

    def unregister(self, name: str):
        """Unregister an entity from the registry.

        Parameters
        ----------
        name: str
            The object to remove from the registry.
        """
        del self._mapping[name]

    def autoregister(self, **meta) -> Callable[[Callable], Callable]:
        """Decorator for registration of objects at interpretation time.

        Parameters
        ----------
        **kwargs:
            Additional meta-data to attach to the registry for the specified item.
        """

        def _decorator(function: Callable) -> Callable:
            self.register(function.__name__, function, **meta)

            return function

        return _decorator

    def keys(self) -> Iterable[Any]:
        return self._mapping.keys()

    def values(self) -> Iterable[Any]:
        return self._mapping.values()

    def items(self) -> Iterable[tuple[Any, Any]]:
        return self._mapping.items()


def parse_value(value: MaybeUnitVector, default_units: str | Unit) -> unyt_array:
    """Parses an array of values into the correct units.

    Parameters
    ----------
    value: array-like or tuple
        The array from which to convert values to correct units. If ``value`` is a ``unyt_array``, the unit is simply converted,
        if ``value`` is a tuple in the form ``(v_array,v_unit)``, the conversion will be made and will return an ``unyt_array``.
        Finally, if ``value`` is an array, it is assumed that the ``default_units`` are correct.
    default_units: str
        The default unit for the quantity.
    Returns
    -------
    unyt_array:
        The converted array.
    """
    return ensure_ytarray(value, default_units)


def ensure_ytquantity(
    x: Number | unyt_quantity, default_units: Unit | str
) -> unyt_quantity:
    """Ensure that an input ``x`` is a unit-ed quantity with the expected units.

    Parameters
    ----------
    x: Any
        The value to enforce units on.
    default_units: Unit
        The unit expected / to be applied if missing.

    Returns
    -------
    unyt_quantity
        The corresponding quantity with correct units.
    """
    if isinstance(x, unyt_quantity):
        return unyt_quantity(x.v, x.units).in_units(default_units)
    elif isinstance(x, tuple):
        return unyt_quantity(x[0], x[1]).in_units(default_units)
    else:
        return unyt_quantity(x, default_units)


def ensure_ytarray(x: ArrayLike, default_units: Unit | str) -> unyt_array:
    """Ensure that an input ``x`` is a unit-ed array with the expected units.

    Parameters
    ----------
    x: Any
        The values to enforce units on.
    default_units: Unit
        The unit expected / to be applied if missing.

    Returns
    -------
    unyt_array
        The corresponding array with correct units.
    """
    if isinstance(x, unyt_array):
        return x.to(default_units)
    elif isinstance(x, tuple) and len(x) == 2:
        return unyt_array(*x).to(default_units)
    else:
        return unyt_array(x, default_units)


def ensure_list(x: Collection) -> list:
    """Convert generic iterable to list."""
    return list(always_iterable(x))
