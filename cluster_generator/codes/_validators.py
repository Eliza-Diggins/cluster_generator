from typing import Type, get_args

import unyt
from pydantic import BaseModel
from unyt.exceptions import UnitConversionError

from cluster_generator.codes._types import UnytArray, UnytQuantity


def validate_units(model_class: Type[BaseModel], model: BaseModel):
    """
    Validates and coerces units for fields of type UnytArray or UnytQuantity after model creation.

    This function is designed to be used as a model validator in Pydantic models that use
    generic types like UnytArray or UnytQuantity to represent quantities with specific units.
    It iterates over all fields of the model, checks their type annotations to determine
    the expected units, and attempts to coerce the provided units to the expected units.

    If coercion is not possible, the function raises a ValueError. This ensures that all
    unit-bearing fields in the model conform to the specified unit types, providing robust
    type compliance for physical quantities.

    Parameters
    ----------
    model_class : Type[BaseModel]
        The Pydantic model class being validated. This provides access to the model's type annotations.

    model : BaseModel
        The instance of the Pydantic model being validated. This contains the actual field values
        that are checked and potentially coerced.

    Returns
    -------
    BaseModel
        The validated model instance with fields coerced to the expected units where possible.

    Raises
    ------
    ValueError
        If units cannot be coerced to the expected type or if an incompatible unit conversion is attempted.

    Usage
    -----
    This function can be added as a model validator to a generic base model in Pydantic to
    ensure type compliance for all fields representing quantities with specific units. For example:

    class PhysicalModel(BaseModel):
        quantity: UnytArray[float, 'keV']
        mass: UnytQuantity[float, 'kg']

        _validate_units = model_validator(mode="after", allow_reuse=True)(validate_units)

    In this example, the `validate_units` function is used as a validator to check and coerce
    the units of `quantity` and `mass` fields to 'keV' and 'kg', respectively.
    """
    # To include inherited fields
    all_fields = {}
    if hasattr(model_class, "__mro__"):
        for cls in model_class.__mro__:
            if hasattr(cls, "__annotations__"):
                all_fields.update(cls.__annotations__)
    else:
        all_fields = model_class.__annotations__

    for field_name, field_type in all_fields.items():
        args = get_args(field_type)
        if len(args) == 2 and (
            field_type.__name__ in [UnytArray.__name__, UnytQuantity.__name__]
        ):
            dtype, unit_ref = args
            if isinstance(unit_ref, str):
                expected_units = unyt.Unit(unit_ref)
            elif hasattr(unit_ref, "__forward_arg__"):
                # Extract the string from ForwardRef
                expected_units = unyt.Unit(unit_ref.__forward_arg__)
            else:
                expected_units = unit_ref
            value = getattr(model, field_name)

            if value is None:
                continue
            try:
                if isinstance(value, unyt.unyt_array) and value.units != expected_units:
                    coerced_value = value.to(expected_units)
                    setattr(model, field_name, coerced_value)
                elif (
                    isinstance(value, unyt.unyt_quantity)
                    and value.units != expected_units
                ):
                    coerced_value = value.to(expected_units)
                    setattr(model, field_name, coerced_value)
            except (UnitConversionError, ValueError):
                raise ValueError(
                    f"Field {field_name} requires dimensions {expected_units.dimensions} but value has {value.units.dimensions}."
                )
    return model
