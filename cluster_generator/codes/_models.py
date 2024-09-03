from typing import Type

from pydantic import BaseModel, Field
from pydantic import create_model as pydantic_create_model
from pydantic import model_validator

from cluster_generator.codes._types import UnytArray, UnytQuantity
from cluster_generator.codes._validators import validate_units


class AutoValidatedBaseModel(BaseModel):
    def __setattr__(self, name, value):
        """
        Override the default `__setattr__` method to automatically validate fields on assignment.

        Parameters
        ----------
        name : str
            The name of the attribute being set.
        value : Any
            The value to be assigned to the attribute.

        Raises
        ------
        ValidationError
            If the value being assigned does not conform to the model's schema.
        """
        if name in self.model_fields:
            # Use Pydantic's validate_python method for the specific field
            value = getattr(
                self.__class__.__pydantic_validator__.validate_assignment(
                    self.__class__.model_construct(), name, value
                ),
                name,
            )

        super().__setattr__(name, value)


class ClusterGeneratorModel(AutoValidatedBaseModel):
    """
    A custom Pydantic BaseModel that automatically validates attributes on assignment.

    The `ClusterGeneratorModel` class extends the Pydantic `BaseModel` to provide automatic validation
    whenever an attribute is modified. This ensures that any changes made to an instance of the model
    remain consistent with the model's schema and constraints.

    Attributes are validated both upon initial creation and whenever an attribute's value is updated.
    This is particularly useful for models that are frequently modified and need to maintain strict
    data integrity.

    Attributes
    ----------
    None specific to this class beyond what is defined in Pydantic's BaseModel.

    Methods
    -------
    __setattr__(self, name, value)
        Automatically validate attributes on assignment to ensure they conform to the model schema.

    units_validate(cls, val)
        Validate that all fields with units conform to the specified unit requirements.

    __repr__(self) -> str
        Return a concise and informative string representation of the model instance for debugging.

    __str__(self) -> str
        Return a readable string representation of the model instance, suitable for displaying
        model content.

    See Also
    --------
    pydantic.BaseModel : The base class that `ClusterGeneratorModel` extends.
    validate_units : A utility function used to validate unit-bearing fields.

    Examples
    --------
    ```python
    class ExampleModel(ClusterGeneratorModel):
        field1: int
        field2: float

    example = ExampleModel(field1=1, field2=2.0)
    print(example)  # Output will show a readable representation of the model
    example.field1 = "invalid"  # Raises a ValidationError due to invalid type
    """

    @model_validator(mode="after")
    def units_validate(cls, val):
        """
        Validate that all fields with units conform to the specified unit requirements.

        This method is automatically called after the model is instantiated or modified to ensure
        that unit-bearing fields have valid units.

        Parameters
        ----------
        val : BaseModel
            The model instance being validated.

        Returns
        -------
        BaseModel
            The validated model instance.
        """
        return validate_units(cls, val)

    def __repr__(self) -> str:
        """
        Return a concise and informative string representation of the model instance for debugging.

        If the model has more than 5 fields, the output is truncated with '...' to reduce length.

        Returns
        -------
        str
            A string representation of the `ClusterGeneratorModel` instance showing the class name
            and key fields, truncated if there are more than 5 fields.
        """
        max_fields = 5
        fields = [(k, v) for k, v in self.__dict__.items() if not k.startswith("_")]

        if len(fields) > max_fields:
            visible_fields = (
                fields[: max_fields // 2]
                + [("...", "...")]
                + fields[-max_fields // 2 :]
            )
        else:
            visible_fields = fields

        field_reprs = ", ".join(f"{k}={v!r}" for k, v in visible_fields)
        return f"{self.__class__.__name__}({field_reprs})"

    def __str__(self) -> str:
        """
        Return a readable string representation of the model instance, suitable for displaying model content.

        If the model has more than 5 fields, the output is truncated with '...' to reduce length.

        Returns
        -------
        str
            A string representation of the `ClusterGeneratorModel` instance, displaying the class name
            and all fields in a readable format, truncated if there are more than 5 fields.
        """
        max_fields = 5
        fields = [(k, v) for k, v in self.__dict__.items() if not k.startswith("_")]

        if len(fields) > max_fields:
            visible_fields = (
                fields[: max_fields // 2]
                + [("...", "...")]
                + fields[-max_fields // 2 :]
            )
        else:
            visible_fields = fields

        field_strs = ", ".join(f"{k}={v}" for k, v in visible_fields)
        return f"{self.__class__.__name__}({field_strs})"


def create_model(model_name, **fields) -> Type[BaseModel]:
    """
    Custom create_model function that uses AutoValidatingBaseModel as the base class.
    """
    return pydantic_create_model(model_name, __base__=ClusterGeneratorModel, **fields)


class OutputTimeSpecifier(ClusterGeneratorModel):
    START: UnytQuantity[float, unyt.Unit("Myr")] = Field()
    END: UnytQuantity[float, unyt.Unit("Myr")] = Field()
    OUTPUT_TIMES: UnytArray[float, unyt.Unit("Myr")] = Field(default=None)
    INTERVAL: UnytQuantity[float, unyt.Unit("Myr")] = Field(default=None)

    @model_validator(mode="after")
    def units_validate(cls, val):
        return validate_units(cls, val)

    @model_validator(mode="after")
    def check_times(cls, values):
        start = values.START
        end = values.END
        interval = values.INTERVAL
        output_times = values.OUTPUT_TIMES

        # Ensure either INTERVAL or OUTPUT_TIMES is provided
        if interval is None and output_times is None:
            raise ValueError("Either INTERVAL or OUTPUT_TIMES must be provided.")

        # Validate OUTPUT_TIMES
        if output_times is not None:
            # Ensure OUTPUT_TIMES are within START and END
            if any(time < start or time > end for time in output_times):
                raise ValueError(
                    "All OUTPUT_TIMES must be within the range of START and END."
                )

            # Ensure the first and last elements of OUTPUT_TIMES are START and END
            if output_times[0] != start:
                output_times[0] = start  # Correct the first element to be START
            if output_times[-1] != end:
                output_times[-1] = end  # Correct the last element to be END

            # Update the validated OUTPUT_TIMES in values
            values.OUTPUT_TIMES = output_times

        return values
