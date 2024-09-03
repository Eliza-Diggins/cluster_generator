"""Abstract base classes for implementing hydrodynamics code support."""
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, ClassVar, Type, TypeVar

import unyt
from pydantic import BaseModel, Field, model_validator
from ruamel.yaml import YAML

from cluster_generator.codes._models import (
    ClusterGeneratorModel,
    OutputTimeSpecifier,
    create_model,
)
from cluster_generator.codes._types import MissingValue, UField, UnytArray, UnytQuantity
from cluster_generator.codes._validators import validate_units
from cluster_generator.ics import ClusterICs
from cluster_generator.utils import LogDescriptor, _config_directory, _MissingParameter

Instance = TypeVar("Instance", bound="CodeFrontend")
Value = TypeVar("Value")
Attribute = TypeVar("Attribute")

runtime_parameters_directory = Path(
    os.path.join(_config_directory.parents[0], "runtime_parameters")
)

yaml = YAML(typ="safe")


class RuntimeParameters(ABC):
    """
        Abstract base class for implementing hydrodynamics code support through runtime parameter management.

        The `RuntimeParameters` class serves as an abstract base class that provides methods for dynamically
        generating and managing runtime parameters (RTPs) for hydrodynamics simulations. It uses Pydantic for
        validation and YAML for schema definition, allowing easy customization and extension by developers.

        Attributes
        ----------
        TYPE_MAP : dict
            A mapping of supported data types to their corresponding Python types. This mapping allows the RTP file
            to have numpy-like type specification but for them to be read as pydantic-like python types. The true
            types (as stored in the ``.yaml`` file) are retained and used when writing the RTP file.

        WRITE_VALIDATORS : list of Callable
            During the RTP file generation process, the field values storing the user's RTPs need to be converted
            to a version which can be written to a correctly formatted file. ``WRITE_VALIDATORS`` is a list of functions, each
            with signature ``def f(instance,field,field_info,value) -> errors,skip,value`` which are to be used
            during validation. See the section below on writing-time validation for more information.

        See Also
        --------
        pydantic.BaseModel : The Pydantic base class used for data validation.
        ruamel.yaml.YAML : YAML parsing used for loading RTP schemas.
        ClusterICs : A class representing initial conditions for cluster simulations.


        Notes
        -----
        .. rubric:: Format of the Underlying YAML

        The YAML schema file defines the runtime parameters (RTPs) used in hydrodynamics simulations. It provides a
        structured way to specify each parameter's data type (`dtype`), shape (`shape`), default value (`default_value`),
         and units (`default_units`) along with any other optional metadata the developer needs to incorporate the frontend.
        Each parameter is represented as a key in the YAML file, with its metadata as nested fields.

        For example, a simple YAML schema might look like this:

        .. code-block:: yaml

            density:
              dtype: float
              shape: 1
              default_value: 1.0
              default_units: g/cm**3

            temperature:
              dtype: float
              shape: 1
              default_value: 300
              default_units: K

            particles:
              dtype: int
              shape: 1
              default_value: 1000

        This schema is parsed by the `RuntimeParameters` class to dynamically create a ``Pydantic`` model class which the user
        then fills with RTP values and (eventually) uses to create the RTP template.
        The model then serves as a blueprint for validating and managing the RTPs in the simulation code.

    .. rubric:: Model Generation and Validators

        The `RuntimeParameters` class dynamically generates a Pydantic model based on the YAML schema.
        This process involves several steps:

        1. **Reading the YAML Schema**: The `_read_yaml_schema` method reads the YAML file specified during initialization
           and loads its contents into a Python dictionary.
           This should almost never need to be altered as it's a very simple ``yaml`` loading procedure.

        2. **Creating Schema Fields**: The `_create_schema_fields` method iterates over the YAML dictionary to create
           Pydantic-compatible fields. It determines the field type, default value, and whether to use `default` or `default_factory`
           based on the provided metadata. This may frequently need to be implemented in custom subclasses to manage non-standard
           fields of various sorts.

        3. **Adding Field Validators**: Custom field validators can be added using the `_get_field_validators` method.
           These validators enforce specific rules for each field. For example, a validator might ensure that a numeric
           field stays within a certain range or that a string field matches a particular format.

        4. **Adding Model Validators**: The `_add_model_validators` method adds model-level validators,
           such as a unit validator to ensure that all unit-bearing fields conform to the required unit system.
           Validators like `units_validator` are essential for maintaining consistency across the simulation's runtime parameters.

        When called, the :py:class:`RuntimeParameters` descriptor will dynamically create the model. If it is called as a class attribute,
        it will return :py:meth:`RuntimeParameters.initialize_base_model`, otherwise it will return :py:meth:`RuntimeParameters.initialize_instance_model`.
        By default, these are the same model; however, in some cases, it is necessary to implement a model that is dependent on a particular
        :py:class:`CodeFrontend` instance. In this case, the :py:meth:`RuntimeParameters.initialize_instance_model` method should be altered.

        The resulting Pydantic model provides a robust framework for managing RTPs, ensuring they meet the requirements defined in the YAML schema.

        .. rubric:: Writing RTP Files and Writing Validation

        The `RuntimeParameters` class includes functionality for writing RTP files, which are configuration files for simulations.
        Before writing these files, it is crucial to ensure that all parameters are valid and conform to the expected formats.
        This validation is performed using several static methods defined in the class:

        All of the validators for this process are defined as static methods ``_write_validator_<>``. Then have the following
        signature: ``def foo(instance,field,field_info,value) -> errors,skip,value``. Here, the instance is the :py:class:`CodeFrontend` instance
        being validated, ``field`` is the field name, ``field_info`` is the ``pydantic`` field data, and ``value`` is the current value of the
        parameter. The validator should return a list of errors (``errors``) it encounters, ``skip``, to indicate that the parameter
        should be left out of the final file, and ``value``: the validated value of the parameter.

        All fields must pass all validators, so they should properly handle running over all of the available fields.

        .. rubric:: Setter Methods for Custom Field Handling

        The `RuntimeParameters` class supports custom setter methods to dynamically set field values based on specific rules or computations.
        This can be particularly useful when certain runtime parameters are derived from other fields or require complex logic to determine
        their values.

        Setter methods are defined by prefixing the method name with `set_`, followed by the field name. For example, to define a setter
        for a field named `temperature`, you would define a method `set_temperature`. These methods can then be used to compute and
        assign values dynamically during the runtime parameter setup.

        The :py:meth:`find_field_setters` method automatically identifies these setter methods and maps them to their corresponding fields.
        When filling in the RTP values (using the :py:meth:`fill_values` method), the class will call the appropriate setter method for each field
        that has one defined. If no setter method is found for a given field, the default behavior is to use the current value of the field
        as is.

        Developers can override or extend setter methods in subclasses to customize how specific fields are set, providing greater control
        and flexibility in defining runtime parameters for different simulation scenarios.
    """

    TYPE_MAP = {
        "float": float,
        "float64": float,
        "f8": float,
        "float32": float,
        "f4": float,
        "uint8": int,
        "uint16": int,
        "uint32": int,
        "uint64": int,
        "u4": int,
        "u8": int,
        "u16": int,
        "int": int,
        "int8": int,
        "int16": int,
        "int32": int,
        "int64": int,
        "i4": int,
        "i8": int,
        "i16": int,
        "str": str,
        "char": str,
        "bool": bool,
    }

    def __init__(self, path: str | Path):
        """
        Initialize the RTP class from a `.yaml` file.

        Parameters
        ----------
        path : str or Path
            The path to the RTP YAML schema file.

        Notes
        -----
        The initialization process involves loading the RTP schema from the specified YAML file
        and setting up the base Pydantic model for runtime parameters.

        See Also
        --------
        _read_yaml_schema : Method for reading the YAML schema file.
        """
        self.path: Path = Path(path)
        """ str: The path to the ``.yaml`` file where the **default** RTPs are stored."""
        self.base_model: Type[BaseModel] = self.initialize_base_model()
        """ Type[BaseModel]: The pydantic-validated base model for the RTPs."""
        self._owner: Instance = None

    def __set_name__(self, owner, name):
        # This allows us to fetch the owner and ensure that the model can be initialized with
        # knowledge of the owner.
        self._owner = owner

    def __get__(self, instance, owner):
        """Get the model instance associated with the owning class instance."""
        # If the instance is none, we return the base model so that it can
        # be generically accessed as a class attribute. This provides
        # access to the fields if needed.
        if instance is None:
            return self.base_model

        # Otherwise, check for initialization, then return.
        if instance._rtp is None:
            # initialize the base model with defaults.
            instance._rtp = self.initialize_instance_model(instance)()

        return instance._rtp

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the object for debugging.

        Returns
        -------
        str
            A string representation of the `RuntimeParameters` instance showing the path and model details.
        """
        return (
            f"<RuntimeParameters(path={self.path!r}, "
            f"base_model={self.base_model.__name__})>"
        )

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the object.

        Returns
        -------
        str
            A string summarizing the `RuntimeParameters` instance.
        """
        return f"[RuntimeParameters: {self.path}]"

    def __eq__(self, other: Any) -> bool:
        """
        Check if two `RuntimeParameters` instances are equal based on their paths and models.

        Parameters
        ----------
        other : Any
            The object to compare against.

        Returns
        -------
        bool
            True if the instances are equal, False otherwise.
        """
        if not isinstance(other, RuntimeParameters):
            return NotImplemented
        return self.path == other.path and self.base_model == other.base_model

    def __iter__(self):
        """
        Allow iteration over the fields of the base model.

        Yields
        ------
        tuple
            Pairs of field names and their corresponding field definitions.
        """
        for name, field in self.base_model.__fields__.items():
            yield name, field

    def __contains__(self, item: str) -> bool:
        """
        Check if a given field is present in the runtime parameters.

        Parameters
        ----------
        item : str
            The name of the field to check.

        Returns
        -------
        bool
            True if the field exists, False otherwise.
        """
        return item in self.base_model.__fields__

    def __len__(self) -> int:
        """
        Return the number of fields defined in the runtime parameters.

        Returns
        -------
        int
            The number of fields in the base model.
        """
        return len(self.base_model.__fields__)

    def initialize_base_model(self) -> Any:
        """
        Initialize the base Pydantic model from the YAML schema.

        Returns
        -------
        Type[BaseModel]
            The dynamically created Pydantic model based on the YAML schema.

        Notes
        -----
        This method reads the YAML schema file and uses it to define fields and validators for the
        dynamic Pydantic model. It also adds any required model-level validators.

        This is the base model, meaning it may be missing any instance-specific fields which were introduced
        in the instance model.

        See Also
        --------
        initialize_instance_model : Initializes the base model with defaults.
        """
        rtp_schema = self._read_yaml_schema()
        fields = self._create_schema_fields(rtp_schema)
        field_validators = self._get_field_validators(fields)

        model = create_model(
            "DynamicRuntimeParameters", **fields, __validators__=field_validators
        )

        self._add_model_validators(model)

        return model

    def initialize_instance_model(self, _: Instance) -> Any:
        """
        Initialize the instance-specific model, often used for dynamically generated subclasses.

        Parameters
        ----------
        _ : Instance
            An instance of the owning class.

        Returns
        -------
        Type[BaseModel]
            A dynamically created Pydantic model instance specific to the owning class.

        See Also
        --------
        initialize_base_model : Initializes the base model with defaults.
        """
        # Load the rtp schema from file.
        return self.initialize_base_model()

    def _get_field_validators(
        self, _: dict[str, tuple[Type, Field]]
    ) -> dict[str, Callable]:
        """
        Generate field-specific validators for the dynamic model.

        Parameters
        ----------
        _ : dict
            Dictionary mapping field names to types and Pydantic Field objects.

        Returns
        -------
        dict
            A dictionary of field names and their corresponding validator functions.

        Notes
        -----
        This method allows for customization of field-level validation logic.
        """
        return {}

    def _add_model_validators(self, model: Type[BaseModel]) -> Type[BaseModel]:
        """
        Add standardized validators to the model, such as unit validation.

        Parameters
        ----------
        model : Type[BaseModel]
            The Pydantic model to which validators will be added.

        Returns
        -------
        Type[BaseModel]
            The modified Pydantic model with added validators.

        Notes
        -----
        Adds a unit validator as a standardized model-level validator.

        Generally, this meands defining a new class ``_DynamicModel`` which inherits from the
        dynamically generated model and implements new validators for the necessary things.
        """

        class _DynamicModel(model):
            # add the unit validator to the model. This is the only standardized
            # validator for the entire model.
            @model_validator(mode="after")
            def units_validator(cls, values):
                return validate_units(cls, values)

        return _DynamicModel

    def _read_yaml_schema(self) -> dict:
        """
        Read and parse the YAML schema file to obtain the RTP configuration.

        Returns
        -------
        dict
            A dictionary representing the RTP schema loaded from the YAML file.

        Notes
        -----
        This method reads the YAML file specified during initialization and parses its contents into a dictionary.
        """
        with open(self.path, "r") as file:
            rtp_schema = yaml.load(file)

        return rtp_schema

    def _create_schema_fields(self, schema: dict) -> dict:
        """
        Create Pydantic model fields from the YAML schema definitions.

        Parameters
        ----------
        schema : dict
            The YAML schema dictionary defining field properties.

        Returns
        -------
        dict
            A dictionary where keys are field names and values are tuples of field types and Pydantic Field objects.

        Notes
        -----
        This method dynamically constructs field types, default values, and other metadata required for
        creating a Pydantic model.
        """
        fields = {}
        for flag, metadata in schema.items():
            _raw_type = metadata.get("dtype", "str")
            _basetype = self.__class__.TYPE_MAP[_raw_type]
            _shape = metadata.get("shape", 1)
            _default_value = metadata.get("default_value", None)
            _default_units = metadata.get("default_units", None)

            # Make sure that these all end up in the field metadata to make writing easy
            # at the end.
            for mval, mkey in zip(
                [_raw_type, _shape, _default_value, _default_units],
                ["dtype", "shape", "default_value", "default_units"],
            ):
                metadata[mkey] = mval

            # Determine if the value is scalar or not.
            _is_scalar = _shape == 1

            # construct the field kwargs. We need to determine default / default_factory etc.
            _field_kwargs = {**metadata}

            if (_default_units is not None) and (_default_value is not None):
                _default_value = (
                    unyt.unyt_array(_default_value, _default_units)
                    if not _is_scalar
                    else unyt.unyt_quantity(_default_value, _default_units)
                )

            if (_default_value is None) or _is_scalar:
                # the default is None, we set it and enforce scalar.
                _field_kwargs["default"] = _default_value
                _ = _field_kwargs.pop("default_factory", None)
            else:
                _field_kwargs["default_factory"] = lambda _df=_default_value: _df
                _ = _field_kwargs.pop("default", None)

            # Constructing the relevant field type.
            if (not _default_units) and (not _is_scalar):
                dtype = list[_basetype]
            elif (_default_units is not None) and (not _is_scalar):
                dtype = UnytArray[_basetype, unyt.Unit(_default_units)]
            elif (_default_units is None) and (_is_scalar):
                dtype = _basetype
            else:
                dtype = UnytQuantity[_basetype, unyt.Unit(_default_units)]

            # We now create the field and add it to the dictionary.
            fields[flag] = (dtype, Field(**_field_kwargs))

        return fields

    def find_all_setters(self) -> list[Callable]:
        """
        Find and return a dictionary of available field setters for the RTPs.

        Returns
        -------
        dict
            A dictionary containing field names as keys and their corresponding setter functions as values.

        Notes
        -----
        The method searches for both complex and simple setters within the owning class.
        """
        setters = []
        # Load complex setters
        for k, v in self.__class__.__dict__.items():
            # iterate through all of the class's definitions and seek out set_...
            if "_setter_" in k and isinstance(v, Callable):
                setters.append(v)

        # Finding field setters (simple setters)
        for field_name, _ in self._owner.__fields__.items():
            flag = self._owner.get_field_meta(field_name, "flag", None)
            setter = self._owner.get_field_meta(field_name, "setter", None)

            if flag is None:
                continue
            if setter is None:
                setter_func = lambda _inst, _, __, _fn=field_name, _fln=flag: {
                    _fln: getattr(_inst, _fn)
                }
            else:
                setter_func = lambda _inst, _own, _ic, _fln=flag, _func=setter: {
                    _fln: _func(_inst, _own, _ic)
                }
            setters.append(setter_func)

        return setters

    def fill_values(self, instance: Instance, ics: ClusterICs):
        """
        Fill RTP values for a given simulation code instance based on compile-time fields and user fields.

        Parameters
        ----------
        instance : Instance
            The instance of the simulation code for which RTP values are being generated.
        ics : ClusterICs
            The initial conditions for the simulation.

        Returns
        -------
        bool
            `True` if the process failed, `False` otherwise.

        Notes
        -----
        This method uses the owning class's attributes and methods to set values in the RTP model.
        """
        instance.logger.info("Constructing RTP values for %s...", instance)
        setters = self.find_all_setters()
        instance.logger.debug(
            "Using %s setters to set RTPs in %s.", len(setters), instance
        )

        for setter in setters:
            _attribute_dictionary = setter(instance, type(instance), ics)
            for k, v in _attribute_dictionary.items():
                setattr(instance.rtp, k, v)
                instance.logger.debug("\t%s -> %s.", k, getattr(instance.rtp, k))

    @staticmethod
    def _write_validator_required(_, field, field_info, value):
        """
        Validate that required fields have values and determine if they should be skipped in the output.
        """
        _errors = []
        _skip = False
        try:
            _default_value = field_info.json_schema_extra.get("default_value", None)
            _required = field_info.json_schema_extra.get("required", False)
            # validate.
            if _required and (value is None):
                _errors.append(
                    ValueError(f"Field {field} is required but value is None.")
                )
                _skip = True
            elif (not _required) and (value is None):
                _skip = True
            elif (not _required) and (value is not None):
                if value == _default_value:
                    _skip = True
            else:
                pass

        except Exception as e:
            _errors.append(
                ValueError(f"Failed to validate required status for {field}: {e}")
            )
            _skip = True

        return _errors, _skip, value

    @staticmethod
    def _write_validator_units(instance, field, field_info, value):
        """
        Ensure that unit-bearing fields are correctly converted to the instance's unit system.

        Parameters
        ----------
        instance : Instance
            The instance of the simulation code that owns the RTPs.
        field : str
            The name of the field being validated.
        field_info : pydantic.fields.ModelField
            The field information object containing metadata about the field.
        value : Any
            The value of the field to be validated.

        Returns
        -------
        tuple
            A tuple containing a list of errors, a boolean indicating whether the field should be skipped,
            and the converted field value.

        Notes
        -----
        This validator ensures that unit-bearing fields are correctly converted to the unit system of the
        simulation instance, preventing unit mismatches and ensuring consistency.
        """
        _errors = []
        _skip = False
        try:
            _default_units = field_info.json_schema_extra.get("default_units", None)

            if (_default_units is not None) and (value is not None):
                value = value.in_base(instance.unit_system).value

        except Exception as e:
            _errors.append(ValueError(f"Field {field} failed unit validation: {e}."))
            _skip = True

        return _errors, _skip, value

    _WRITE_VALIDATORS: ClassVar[list] = [
        _write_validator_required,
        _write_validator_units,
    ]

    def _setup_write(self, instance):
        """
        Prepare values for writing to the RTP file, enforcing norms about data types and shapes.

        Parameters
        ----------
        instance : Instance
            The instance of the simulation code for which RTP values are being prepared.

        Returns
        -------
        dict
            A dictionary of field values ready to be written to the RTP file.

        Raises
        ------
        ErrorGroup
            If any validation errors are encountered during the preparation process.

        Notes
        -----
        This method iterates through all fields in the RTP model, applies field-specific validators, and
        collects validated values for writing. If any fields fail validation, an `ErrorGroup` is raised
        containing all encountered errors. It should not need to be altered. New validators should instead
        be implemented to meet the needs of the new frontend.
        """
        rtp_model = instance.rtp

        _write_values = {}
        errors = []
        for field, field_info in rtp_model.__fields__.items():
            skip = False
            if field in ["_WRITE_VALIDATORS"]:
                continue

            value = getattr(instance.rtp, field)

            if field_info.json_schema_extra is None:
                _write_values[field] = value
                continue

            for validator in self.__class__._WRITE_VALIDATORS:
                _validation_errors, skip, value = validator(
                    instance, field, field_info, value
                )
                errors += _validation_errors

                if len(_validation_errors) or skip:
                    break

            if not skip:
                _write_values[field] = value

        if len(errors):
            from cluster_generator.utils import ErrorGroup

            raise ErrorGroup(f"Found {len(errors)} validation errors!", errors)
        return _write_values

    @abstractmethod
    def write_rtp_template(
        self,
        instance: Instance,
        path: str | Path,
        overwrite: bool = False,
    ) -> Path:
        """
        Generate a ready-to-run RTP template for a given :py:class:`SimulationCode` instance.

        Parameters
        ----------
        instance : Instance
            The simulation code instance to generate the RTP template for.
        path : str or Path
            The path to the RTP file/directory at which to generate the template.
        overwrite : bool, optional
            Allow method to overwrite an existing file at this path. Default is `False`.

        Returns
        -------
        Path
            The path to the generated RTP template file.

        Raises
        ------
        ValueError
            If the RTPs found in `instance` are invalid or have any issues.

        Notes
        -----
        .. warning::

            The template generated for the simulation code may not be the optimal (or even functional) RTPs
            for your science case. Templates are written to correctly specify all IC-related fields, but
            additional fields are generally filled by defaults or simply not present. The user should consult
            with the user guide for their code to fill in/change any missing values.

        This method must be implemented by subclasses to provide specific logic for writing RTP templates
        tailored to different hydrodynamics codes.
        """
        pass


class _MissingRTP(_MissingParameter):
    # Sentinel for missing RTPs
    pass


class CodeFrontend(ClusterGeneratorModel, ABC):
    """
    Abstract base class for all simulation code frontends.

    The :py:class:`CodeFrontend` class provides the necessary interface and common functionality required by all simulation code frontends.
    It extends :py:class:`~cluster_generator.codes.types.ClusterGeneratorModel` to include automatic validation and integrates with
    :py:class:`RuntimeParameters` for managing runtime parameters specific to each simulation code.

    This class also defines the common methods and properties that must be implemented by any subclass to ensure compatibility
    with the framework. This includes generating initial conditions (ICs) and managing unit systems.

    Attributes
    ----------
    rtp : ClassVar[RuntimeParameters]
        The runtime parameters descriptor for the simulation code. This does not need to be explicitly specified for each code.
    logger : ClassVar[logging.Logger]
        The logger for this code frontend.
    OUTPUTS : OutputTimeSpecifier
        User-defined outputs for the simulation, specified as a `UField` (user field).


    See Also
    --------
    ClusterGeneratorModel : The base class that provides automatic validation on attribute assignment.
    RuntimeParameters : Manages runtime parameters for simulation codes.
    OutputTimeSpecifier : Specifies the user-defined outputs for the simulation.
    ClusterICs : Represents the initial conditions for cluster simulations.

    Examples
    --------

    .. code-block:: python

        class MyCodeFrontend(CodeFrontend):
            rtp = RuntimeParameters(path="path/to/rtp.yaml")

            def generate_ics(self, initial_conditions: ClusterICs, overwrite: bool = False, **kwargs) -> Path:
                # Custom implementation for generating initial conditions
                ...

            @property
            def unit_system(self) -> unyt.UnitSystem:
                # Return the unit system defined by the code
                return unyt.UnitSystem("cgs")

        # Example usage
        my_frontend = MyCodeFrontend()
        ics = ClusterICs(...)  # Initialize with appropriate parameters
        my_frontend.generate_ics(ics)
        my_frontend.generate_rtp_template(path="path/to/output")

    """

    rtp: ClassVar[RuntimeParameters] = None
    logger: ClassVar[logging.Logger] = LogDescriptor()

    # USER PARAMETERS
    OUTPUTS: OutputTimeSpecifier = UField()

    # COMPILE-TIME PARAMETERS

    def model_post_init(self, __context: dict):
        # Initialize the instance._rtp attribute to store instance-level RTPs.
        self._rtp = None
        self._unit_system = None

    def __str__(self):
        return f"<SimulationCode name={self.__class__.__name__}>"

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def generate_ics(
        self, initial_conditions: ClusterICs, overwrite: bool = False, **kwargs
    ) -> Path:
        """Convert a :py:class:`ClusterICs` object into a format that is ready to
        run in this simulation software.

        Parameters
        ----------
        initial_conditions : :py:class:`ClusterICs`
            The initial conditions object to convert to the correct IC filetype for use in the simulation software.
        overwrite : bool, optional
            Allow the IC to overwrite existing data. ``False`` by default.
        kwargs : dict, optional
            Additional keyword arguments defined by particular codes.

        Returns
        -------
        :py:class:`Path`
            The path to the initial conditions file.
        """
        pass

    @property
    @abstractmethod
    def unit_system(self) -> unyt.UnitSystem:
        """The unit system defined by the code.

        Returns
        -------
        :py:class:`unyt.UnitSystem`
            The resulting unit system.
        """
        pass

    @classmethod
    def get_field_meta(cls, field: str, meta_key: str, default: Any = MissingValue):
        """
        Utility method to get the extra metadata from a Pydantic model's field.

        Parameters
        ----------
        field : str
            The name of the field to get the metadata for.
        meta_key : str
            The metadata key to seek.
        default : Any, optional
            The default value to assign if the value is not present.

        Returns
        -------
        Any
            The extra metadata for the specified field.

        Raises
        ------
        ValueError
            If the specified field is not found or the metadata key is not specified.
        """
        try:
            field = cls.__fields__[field]
        except KeyError:
            raise ValueError(
                f"Cannot fetch metadata from field {field} of {cls.__name__}. Failed to find field in model."
            )

        value = field.json_schema_extra.get(meta_key, default)

        if value == MissingValue:
            raise ValueError(
                f"Meta key {meta_key} for field {field} in {cls.__name__} was not specified."
            )
        else:
            return value

    @classmethod
    def get_rtp_class(cls) -> RuntimeParameters:
        """Fetch the :py:class:`RuntimeParameters` class associated with this simulation
        code.

        Returns
        -------
        :py:class:`RuntimeParameters`
            The class object for the given code's RTPs.

        Notes
        -----
        When accessed as a class variable (``SimulationCode.rtp``), or as the attribute of a class instance (``self.rtp``), an
        **instance** of the RTP descriptor class is returned. This method provides access to the actual **class** object.
        """
        return cls.__dict__["rtp"]

    def generate_rtp_template(self, path: str | Path, overwrite: bool = False) -> Path:
        """Create the necessary files / execute necessary procedures to setup RTPs for
        the simulation.

        Parameters
        ----------
        path: str
            The path at which to generate the RTPs. In some simulation codes, this may be a directory, in others a single
            path.
        overwrite: bool, optional
            Allow method to overwrite an existing file at this path. Default is ``False``.

        Notes
        -----
        This method is designed to take the RTPs specified in the code-class and write them in the necessary format for a run.
        This only extends to dumping a "as-complete-as-possible" set of RTPs; if users are executing the simulation software
        with additional flags or special physics, these may not be present in the RTPs generated by ``cluster_generator``. The
        intention of this method is to fill-in the RTPs that are directly relevant to the ``cluster_generator`` ICs.

        .. note::

            For developers, this method never needs to be changed. Under-the-hood, it's the :py:class:`RuntimeParameters` class,
            particularly the :py:meth:`RuntimeParameters.write_rtp_template` method which actually does the leg-work here. It is
            that method which needs to be written fresh by the developer for each new code.
        """
        return self.get_rtp_class().write_rtp_template(self, path, overwrite=overwrite)
