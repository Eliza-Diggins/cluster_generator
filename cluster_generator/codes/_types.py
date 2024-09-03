from typing import Any, Generic, TypeVar

import numpy as np
import unyt
from pydantic import Field
from pydantic.json_schema import GetJsonSchemaHandler, JsonSchemaValue
from pydantic_core import core_schema

# Define a type variable that can be any numpy dtype
DType = TypeVar("DType", bound=np.generic)
UnitType = TypeVar("UnitType", bound=str)


class _Missing:
    # generic sentinel for missing values.
    pass


MissingValue = _Missing

# Define type variables
DType = TypeVar("DType", bound=np.generic)
UnitType = TypeVar("UnitType", bound=str)


class UField:
    def __new__(cls, *args: Any, **kwargs: Any) -> Field:
        # Define the fixed metadata to be added to each field
        fixed_metadata = {"field_class": "user"}

        # Merge fixed metadata with any existing extra metadata in kwargs
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        kwargs["extra"].update(fixed_metadata)

        # Return the standard Pydantic Field with updated metadata
        return Field(*args, **kwargs)


class CField:
    def __new__(cls, *args: Any, **kwargs: Any) -> Field:
        # Define the fixed metadata to be added to each field
        fixed_metadata = {"field_class": "compile"}

        # Merge fixed metadata with any existing extra metadata in kwargs
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        kwargs["extra"].update(fixed_metadata)

        # Return the standard Pydantic Field with updated metadata
        return Field(*args, **kwargs)


class UnytUnit(Generic[UnitType]):
    """
    A generic type for unyt units that validates and parses unit types using Pydantic.

    Attributes
    ----------
    unit : unyt.Unit
        The unyt unit instance.

    Methods
    -------
    __get_pydantic_core_schema__() -> Any:
        Returns a Pydantic core schema that validates UnitType objects.
    __get_pydantic_json_schema__() -> JsonSchemaValue:
        Returns a JSON schema for UnitType.
    """

    def __init__(self, unit: UnitType):
        """
        Initializes a UnitTypeModel object.

        Parameters
        ----------
        unit : UnitType
            The unyt unit instance.
        """
        self.unit = unit

    def __repr__(self):
        """Returns a string representation of the UnitTypeModel object."""
        return f"UnytUnit(unit={self.unit})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Any,
    ) -> Any:
        """
        Returns a Pydantic core schema that validates UnitType objects.

        * Strings will be parsed as `"cm"`
        * UnytUnit instances are directly validated
        """

        def validate_from_string(value: str) -> UnytUnit:
            """Convert a string representation to a unyt.Unit."""
            try:
                return unyt.Unit(value)
            except Exception as e:
                raise ValueError(f"Error parsing string to UnitType: {e}")

        from_string_schema = core_schema.chain_schema(
            [
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(validate_from_string),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_string_schema,
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(unyt.Unit),
                    from_string_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: str(instance)
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, _core_schema: Any, handler: Any) -> Any:
        """Returns a JSON schema for UnitType."""
        return handler(core_schema.str_schema())


class UnytArray(Generic[DType, UnitType]):
    """
    A generic type for unyt arrays with a specific numpy dtype and physical units.

    Attributes
    ----------
    array : unyt.unyt_array
        The unyt array instance with a specified dtype and units.
    dtype : DType
        The numpy dtype of the unyt array.
    units : UnitType
        The physical units of the unyt array (e.g., "length", "velocity").

    Methods
    -------
    value() -> np.ndarray:
        Returns the numerical values of the unyt array.
    unit() -> str:
        Returns the unit of the unyt array.
    __get_pydantic_core_schema__() -> Any:
        Returns a Pydantic core schema that validates UnytArray objects.
    __get_pydantic_json_schema__() -> JsonSchemaValue:
        Returns a JSON schema for UnytArray.
    """

    def __init__(self, array: unyt.unyt_array, dtype: DType, units: UnitType):
        """
        Initializes a UnytArray object.

        Parameters
        ----------
        array : unyt.unyt_array
            The unyt array instance.
        dtype : DType
            The numpy dtype of the unyt array.
        units : UnitType
            The physical units of the unyt array.
        """
        self.array = array
        self.dtype = dtype
        self.units = units

    def value(self) -> np.ndarray:
        """Returns the numerical values of the unyt array."""
        return self.array.value

    def unit(self) -> str:
        """Returns the unit of the unyt array."""
        return str(self.array.units)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Any,
    ) -> Any:
        """
        Returns a Pydantic core schema that validates UnytArray objects.

        * Strings will be parsed as `[1, 2, 3] cm`
        * Lists or tuples will be parsed as `([1, 2, 3], "cm")`
        * Dictionaries will be parsed as `{"array": [1, 2, 3], "units": "cm"}`
        """

        def validate_from_string(value: str) -> unyt.unyt_array:
            """Convert a string representation to a unyt.unyt_array."""
            import re

            try:
                match = re.match(r"\[(.*)\]\s*([a-zA-Z/ ]+)$", value.strip())
                if not match:
                    raise ValueError(f"Invalid format for UnytArray string: {value}")

                array_part, unit_part = match.groups()
                array = np.array(
                    eval(f"[{array_part}]")
                )  # Using eval to convert string to list
                units = unyt.Unit(unit_part.strip())
                return unyt.unyt_array(array, units)
            except Exception as e:
                raise ValueError(f"Error parsing string to UnytArray: {e}")

        def validate_from_tuple_or_list(value: list | tuple) -> unyt.unyt_array:
            """Convert a tuple or list to a unyt.unyt_array."""
            try:
                array, units = value
                return unyt.unyt_array(array, units)
            except Exception as e:
                raise ValueError(f"Error parsing tuple or list to UnytArray: {e}")

        def validate_from_dict(value: dict) -> unyt.unyt_array:
            """Convert a dictionary to a unyt.unyt_array."""
            try:
                array = value["array"]
                units = value["units"]
                return unyt.unyt_array(array, units)
            except KeyError as e:
                raise ValueError(f"Missing key {e} in dictionary for UnytArray.")
            except Exception as e:
                raise ValueError(f"Error parsing dictionary to UnytArray: {e}")

        from_string_schema = core_schema.chain_schema(
            [
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(validate_from_string),
            ]
        )

        from_tuple_or_list_schema = core_schema.chain_schema(
            [
                core_schema.union_schema(
                    [
                        core_schema.list_schema(),
                        core_schema.tuple_schema(
                            items_schema=[
                                core_schema.any_schema(),
                                core_schema.str_schema(),
                            ]
                        ),
                    ]
                ),
                core_schema.no_info_plain_validator_function(
                    validate_from_tuple_or_list
                ),
            ]
        )

        from_dict_schema = core_schema.chain_schema(
            [
                core_schema.dict_schema(),
                core_schema.no_info_plain_validator_function(validate_from_dict),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=core_schema.union_schema(
                [from_string_schema, from_tuple_or_list_schema, from_dict_schema]
            ),
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(unyt.unyt_array),
                    from_string_schema,
                    from_tuple_or_list_schema,
                    from_dict_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: (instance.value, str(instance.units))
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Returns a JSON schema for UnytArray."""
        return handler(core_schema.list_schema())


class UnytQuantity(Generic[DType, UnitType]):
    """
    A generic type for unyt quantities with a specific numpy dtype and physical units.

    Attributes
    ----------
    quantity : unyt.unyt_quantity
        The unyt quantity instance with a specified dtype and units.
    dtype : DType
        The numpy dtype of the unyt quantity.
    units: UnitType
        The physical units of the unyt quantity (e.g., "length", "mass").

    Methods
    -------
    value() -> float | np.ndarray:
        Returns the numerical value(s) of the unyt quantity.
    unit() -> str:
        Returns the unit of the unyt quantity.
    __repr__() -> str:
        Returns a string representation of the UnytQuantity object.
    __get_pydantic_core_schema__() -> Any:
        Returns a Pydantic core schema that validates UnytQuantity objects.
    __get_pydantic_json_schema__() -> Any:
        Returns a JSON schema for UnytQuantity.
    """

    def __init__(self, quantity: unyt.unyt_quantity, dtype: DType, units: UnitType):
        """
        Initializes a UnytQuantity object.

        Parameters
        ----------
        quantity : unyt.unyt_quantity
            The unyt quantity instance.
        dtype : DType
            The numpy dtype of the unyt quantity.
        units : UnitType
            The physical units of the unyt quantity.
        """
        self.quantity = quantity
        self.dtype = dtype
        self.units = units

    def value(self) -> float | np.ndarray:
        """Returns the numerical value(s) of the unyt quantity."""
        return self.quantity.value

    def unit(self) -> str:
        """Returns the unit of the unyt quantity."""
        return str(self.quantity.units)

    def __repr__(self):
        """Returns a string representation of the UnytQuantity object."""
        return f"UnytQuantity(quantity={self.quantity}, dtype={self.dtype}, units='{self.units}')"

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Any,
    ) -> Any:
        """
        Returns a Pydantic core schema that validates UnytQuantity objects.

        * Strings will be parsed as `"10 m/s"`
        * Lists or tuples will be parsed as `(10, "m/s")`
        * Dictionaries will be parsed as `{"value": 10, "unit": "m/s"}`
        """

        def validate_from_string(value: str) -> unyt.unyt_quantity:
            """Convert a string representation to a unyt.unyt_quantity."""
            import re

            try:
                match = re.match(
                    r"(\d+(\.\d+)?)\s*([a-zA-Z \d\/\*\(\)]+)$", value.strip()
                )
                if not match:
                    raise ValueError(f"Invalid format for UnytQuantity string: {value}")

                value_part, _, unit_part = match.groups()
                value = float(value_part)
                units = unyt.Unit(unit_part.strip())

                return unyt.unyt_quantity(value, units)
            except Exception as e:
                raise ValueError(f"Error parsing string to UnytQuantity: {e}")

        def validate_from_tuple_or_list(value: list | tuple) -> unyt.unyt_quantity:
            """Convert a tuple or list to a unyt.unyt_quantity."""
            try:
                val, units = value
                return unyt.unyt_quantity(val, units)
            except Exception as e:
                raise ValueError(f"Error parsing tuple or list to UnytQuantity: {e}")

        def validate_from_dict(value: dict) -> unyt.unyt_quantity:
            """Convert a dictionary to a unyt.unyt_quantity."""
            try:
                val = value["value"]
                units = value["unit"]
                return unyt.unyt_quantity(val, units)
            except KeyError as e:
                raise ValueError(f"Missing key {e} in dictionary for UnytQuantity.")
            except Exception as e:
                raise ValueError(f"Error parsing dictionary to UnytQuantity: {e}")

        from_string_schema = core_schema.chain_schema(
            [
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(validate_from_string),
            ]
        )

        from_tuple_or_list_schema = core_schema.chain_schema(
            [
                core_schema.union_schema(
                    [
                        core_schema.list_schema(),
                        core_schema.tuple_schema(
                            items_schema=[
                                core_schema.any_schema(),
                                core_schema.str_schema(),
                            ]
                        ),
                    ]
                ),
                core_schema.no_info_plain_validator_function(
                    validate_from_tuple_or_list
                ),
            ]
        )

        from_dict_schema = core_schema.chain_schema(
            [
                core_schema.dict_schema(),
                core_schema.no_info_plain_validator_function(validate_from_dict),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=core_schema.union_schema(
                [from_string_schema, from_tuple_or_list_schema, from_dict_schema]
            ),
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(unyt.unyt_quantity),
                    from_string_schema,
                    from_tuple_or_list_schema,
                    from_dict_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: (instance.value, str(instance.units))
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Returns a JSON schema for UnytQuantity."""
        return handler(
            core_schema.union_schema(
                [
                    core_schema.str_schema(),
                    core_schema.list_schema(),
                    core_schema.dict_schema(),
                ]
            )
        )
