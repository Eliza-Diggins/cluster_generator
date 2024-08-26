"""Abstract base classes for implementing hydrodynamics code support."""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Type

import unyt
from ruamel.yaml import YAML

from cluster_generator.ics import ClusterICs
from cluster_generator.io.yaml import unyt_array_constructor, unyt_quantity_constructor
from cluster_generator.utilities.logging import LogDescriptor
from cluster_generator.utilities.types import Instance, Self, Value

if TYPE_CHECKING:
    from ruamel.yaml import Loader, Node


class _YAMLDescriptor:
    """Descriptor class for loading and managing YAML configurations.

    This descriptor allows for single-time loading of a custom YAML parser
    for a given class. It is intended to be used within the `RuntimeParameters`
    class to manage configuration files dynamically.

    Parameters
    ----------
    base : YAML, optional
        A base YAML parser object. If not provided, a safe YAML parser is instantiated.

    Attributes
    ----------
    base : YAML
        The base YAML parser object.
    yaml : YAML or None
        The dynamically loaded YAML parser with custom constructors.

    Examples
    --------
    Define a custom YAML descriptor in a class:

    .. code-block:: python

        class MyClass:
            yaml = _YAMLDescriptor()

        instance = MyClass()
        yaml_parser = instance.yaml

    Notes
    -----
    Developers can use this descriptor to add custom YAML constructors for specific types. This is useful for
    adding unit recognition to the RTP files or other tasks of that sort.
    """

    def __init__(self, base: "YAML" = None):
        # load the base parser.
        if base is None:
            self.base = YAML(typ="safe")
        else:
            self.base = base

        self.yaml = None  # -> changes dynamically when called and loaded.

    def __get__(self, instance: Instance, owner: Type[Instance]) -> "YAML":
        if self.yaml is None:
            # for each constructor, we add the constructor.
            self.yaml = self.base
            for loader_tag, loader in owner._yaml_loaders.items():
                self.yaml.constructor.add_constructor(loader_tag, loader)

        return self.yaml


class RuntimeParameters(ABC):
    """Abstract base class for managing runtime parameters (RTPs) of simulation codes.

    This class provides the core functionality for handling runtime parameters required by various
    simulation codes. It is designed to be subclassed by specific code implementations (e.g., AREPO,
    Gadget-2), allowing them to define and manage their own RTPs.


    Notes
    -----

    The :py:class:`RuntimeParameters` class serves four primary functions:

    1. **Reading Defaults**: Loads the default RTPs from a YAML file located at
       ``bin/code_name/runtime_parameters.yaml``.
    2. **Instance Management**: Maintains a set of RTPs for each instance of a
       :py:class:`SimulationCode`, allowing for instance-specific adjustments.
    3. **Dynamic Filling**: Provides mechanisms for dynamically filling in missing RTPs
       based on user-specified parameters (USPs) and compile-time parameters (CTPs).
    4. **Exporting**: Supports exporting RTPs to a format that can be directly used by
       the simulation software.

    .. rubric:: Setters and Instance Management

    Setters are methods that dynamically determine the value of a specific runtime parameter
    based on the current state of the simulation instance, including its initial conditions (ICs)
    and compile-time/user-specified parameters.

    Each setter method should be named in the format ``set_<RTP_name>`` and accept three arguments:

    - ``instance``: The instance of the :py:class:`SimulationCode` for which the RTP is being set.
    - ``owner``: The :py:class:`SimulationCode` class itself.
    - ``ic``: The initial conditions (:py:class:`ClusterICs`) relevant to the simulation.

    .. rubric:: Example of a Setter Method

    A setter method for an RTP named ``softening_length`` might look like this:

    .. code-block:: python

        def set_softening_length(instance, owner, ic):
            # Example logic to set the softening length based on instance properties
            return instance.compute_softening_length(ic)

    These setters are identified dynamically by the :py:meth:`deduce_instance_values` method, which populates
    the RTPs for a specific simulation instance using both the class-defined setters and simple
    direct mappings from the instance attributes.

    .. rubric:: Role of RuntimeParameters as a Class Variable

    The :py:class:`RuntimeParameters` class should be declared as a class variable within each subclass of
    :py:class:`SimulationCode`. This design allows each :py:class:`SimulationCode` subclass to have a unique set of
    RTPs that are specific to the requirements and configurations of that simulation code.

    For example:

    .. code-block:: python

        @dataclass
        class MySimulationCode(SimulationCode):
            rtp: ClassVar[RuntimeParameters] = MyRuntimeParameters

    When accessed as a class attribute (e.g., ``MySimulationCode.rtp``), the :py:class:`RuntimeParameters` provides
    access to the default parameters defined in the configuration file. When accessed as an instance
    attribute (e.g., ``my_instance.rtp``), it allows for manipulation of the specific parameters for
    that simulation instance.

    This dual access pattern ensures that each simulation code can manage its unique parameter set while
    allowing for flexible modifications at the instance level, promoting reusability and customization.

    See Also
    --------
    :py:class:`cluster_generator.codes.abc.SimulationCode`

    Examples
    --------
    Load runtime parameters from a YAML file:

    .. code-block:: python

        rtp = RuntimeParameters('/path/to/runtime_parameters.yaml')
        defaults = rtp.get_defaults()

    Setting RTPs dynamically:

    .. code-block:: python

        def set_example_rtp(instance, owner, ic):
            return 1.0

        rtp.set_example_rtp = set_example_rtp

    .. note::

        Subclasses should implement specific methods to handle RTPs unique to their simulation code.
        Developers should ensure that all required RTPs are accounted for and correctly set up using
        either the default values, direct instance attributes, or dynamic setters.
    """

    _type_converters: ClassVar[dict[Type, Callable]] = {}
    _yaml_loaders: ClassVar[dict[str, Callable[["Loader", "Node"], Any]]] = {
        "!unyt_arr": unyt_array_constructor,
        "!unyt_qty": unyt_quantity_constructor,
    }

    yaml: ClassVar["YAML"] = _YAMLDescriptor()
    """ YAML: The class yaml loader.

    This may be customized to recognize special tags necessary to represent the front end.
    """

    def __init__(self, path: str | Path):
        """Initialize the RTP class from a .yaml file.

        Parameters
        ----------
        path: str
            The path to the RTP file.
        """
        self.path: Path = Path(path)
        """ str: The path to the ``.yaml`` file where the **default** RTPs are stored."""

        # private backend variables
        self._defaults = (
            None  # Stores the defaults from the /bin/codes/<code_name> directory.
        )

    def __get__(self, instance: Instance, owner: Type[Instance]) -> Value:
        """Retrieve the RTPs stored in this object.

        Parameters
        ----------
        instance: :py:class:`SimulationCode`
            The simulation code instance to which the RTPs belong.
        owner: SimulationCode
            The class (non-instantiated) type of which ``instance`` belongs.

        Returns
        -------
        dict
            The RTPs. If an instance of a :py:class:`SimulationCode` accesses its RTPs, then the actual values of the RTPs are returned.
            If a class object accesses its RTPs (as a class variable), then the defaults are returned.
        """
        if instance is None:
            # This is being accessed as a class variable. We want to provide the defaults, including loading them if needed.
            if self._defaults is not None:
                pass
            else:
                with open(self.path, "r") as f:
                    self._defaults = self.__class__.yaml.load(f)

            return (
                self._defaults
            )  # --> This will be a dict with ``flag``, ``default_value`` and other meta-data.

        else:
            # There is an instance provided, we expect to be accessing the instance._rtp attribute.
            if hasattr(instance, "_rtp") and instance._rtp is not None:
                # the instance _rtp already exists and should just be returned.
                pass
            else:
                # We need to set instance._rtp
                _defaults_raw = self.__get__(None, owner)  # create the defaults.

                instance._rtp = {
                    k: v["default_value"] for k, v in _defaults_raw.items()
                }

            return instance._rtp

    @classmethod
    def _converter(cls, typ: Type):
        # Converter decorator.
        def decorator(
            func: Callable[[typ, Instance], Any]
        ) -> Callable[[typ, Instance], Any]:
            # Take the function, add it to the registry, return the same function.
            cls._type_converters[typ] = func
            return func

        return decorator

    @classmethod
    def get_setters(
        cls, _: Instance, owner: Type[Instance]
    ) -> dict[str | Callable[[Instance, Type[Instance], ClusterICs], Any]]:
        """Get a dictionary of the RTP setters defined for this Runtime Parameter
        class."""
        setters = {}

        # Load complex setters
        for k, v in cls.__dict__.items():
            # iterate through all of the class's definitions and seek out set_...
            if "set_" in k and isinstance(v, Callable):
                # k is a setter,
                set_var: str = str(k).replace("set_", "")  # The RTP this setter is for.
                setters[set_var] = v

        # Load simple setters.
        for user_field in [
            _field for _field in fields(owner) if _field.metadata.get("type") == "U"
        ]:
            if user_field.metadata.get("flag", None) is not None:
                # this field has a flag to set.
                set_var: str = str(user_field.metadata["flag"])

                if user_field.metadata.get("setter", None) is not None:
                    setters[set_var] = user_field.metadata["setter"]
                else:
                    setters[set_var] = lambda _inst, _, __, _n=user_field.name: getattr(
                        _inst, _n
                    )

        return setters

    def deduce_instance_values(
        self, instance: Instance, owner: Type[Instance], ics: ClusterICs
    ) -> tuple[bool, Any]:
        """Given an instance of :py:class:`SimulationCode`, generate the corresponding
        RTPs from its compile-time fields and user fields.

        Parameters
        ----------
        instance: SimulationCode
            The instance to fill.
        owner: SimulationCode
            The class owning the instance.
        ics: ClusterICs
            The initial conditions.

        Returns
        -------
        bool
            ``True`` if the process failed. ``False`` otherwise.
        """
        from cluster_generator.utilities.logging import ErrorGroup

        instance.logger.info(f"Constructing RTP values for {instance}...")

        # Determine the available setters
        _setters = self.__class__.get_setters(instance, owner)
        instance.logger.debug(f"Setting {len(_setters)} RTPs in {instance}.")

        errors = []
        for k, v in _setters.items():
            try:
                instance.rtp[k] = v(instance, owner, ics)
                instance.logger.debug(f"\t{k} -> {instance.rtp[k]}.")
            except Exception as e:
                errors.append(e.__class__(f"({k}): {e}"))

        if len(errors):
            return False, ErrorGroup(
                f"Failed to set RTPs for {instance}!", error_list=errors
            )
        else:
            return True, None

    @classmethod
    def _convert_rtp_to_output_types(cls, instance: Instance) -> dict[str, Any]:
        converters = cls._type_converters

        _rtp_output = {}

        for k, v in instance.rtp.items():
            converter = converters.get(v.__class__, None)

            if converter is not None:
                _rtp_output[k] = converter(v, instance)
            else:
                _rtp_output[k] = v

        return _rtp_output

    @abstractmethod
    def write_rtp_template(
        self,
        instance: Instance,
        owner: Type[Instance],
        path: str | Path,
        overwrite: bool = False,
    ) -> Path:
        """Generate a ready-to-run RTP template for a given :py:class:`SimulationCode`
        instance.

        Parameters
        ----------
        instance: SimulationCode
            The simulation code instance to generate the rtp template for.
        owner: type
            The owner type for the simulation Code.
        path: str
            The path to the RTP file/directory at which to generate the template.
        overwrite: bool, optional
            Allow method to overwrite an existing file at this path. Default is ``False``.

        Raises
        ------
        ValueError
            If the RTPs found in ``instance`` are invalid or have any issue.

        Notes
        -----

        .. warning::

            The template generated for the simulation code may not be the optimal (or even functional) RTPs for your science case.
            Templates are written to correctly specify all IC related fields, but additional fields are generally filled by defaults
            or simply not present. The user should consult with the user-guide for their code to fill in / change any missing values.
        """
        pass

    def check_valid(
        self, instance: Instance, owner: Type[Instance], strict: bool = False
    ) -> tuple[bool, list[Exception]]:
        """Check that all of the RTP values are values permitted by the code / cluster
        generator.

        Parameters
        ----------
        instance: :py:class:`SimulationCode`
            The instance of :py:class:`SimulationCode` on which to check the RTPs. If ``None``, then the
            default RTPs are checked.
        owner: :py:class:`SimulationCode`
            The :py:class:`SimulationCode` type / subtype to use for checking.
        strict: bool
            If ``True``, then errors are raised. Otherwise, a boolean is returned.

        Returns
        -------
        tuple
        """
        # fetch the RTPs, get allowed values, etc.
        allowed_rtp_values = {
            k: v.get("allowed_values", None)
            for k, v in self.__get__(None, owner).items()
        }
        current_rtp_values = {
            k: (v if instance is not None else v["default_value"])
            for k, v in self.__get__(instance, owner).items()
        }

        errors = []
        # Check all of the RTPs in order.
        for checked_key, allowed_values in allowed_rtp_values.items():
            if allowed_values is None:
                continue

            if current_rtp_values[checked_key] not in allowed_values:
                errors.append(
                    ValueError(
                        f"{owner}-RTP {checked_key} has value {current_rtp_values[checked_key]} but only {allowed_values} are permitted."
                    )
                )

        result = len(errors) == 0

        if strict and not result:
            from cluster_generator.utilities.logging import ErrorGroup

            raise ErrorGroup(f"{owner} RTP check failed!", errors)

        return result, errors

    @staticmethod
    def _convert_unyt_quantity(value: unyt.unyt_quantity, instance: Instance) -> Any:
        return value.in_base(instance.unit_system).value

    @staticmethod
    def _convert_unyt_array(value: unyt.unyt_array, instance: Instance) -> Any:
        return value.in_base(instance.unit_system).d


RuntimeParameters._convert_unyt_array = RuntimeParameters._converter(unyt.unyt_array)(
    RuntimeParameters._convert_unyt_array
)
RuntimeParameters._convert_unyt_quantity = RuntimeParameters._converter(
    unyt.unyt_quantity
)(RuntimeParameters._convert_unyt_quantity)


@dataclass
class SimulationCode(ABC):
    """Abstract base class representing a specific (magneto)hydrodynamics code frontend.

    This class serves as a template for defining the interface and essential methods required
    for any simulation code that integrates with the cluster generator framework. It manages
    both compile-time parameters (CTPs) and user-specified parameters (USPs), and provides
    methods for handling runtime parameters (RTPs) necessary for simulation setup and execution.

    Examples
    --------
    To use a specific simulation code frontend, subclass `SimulationCode` and implement the abstract methods:

    .. code-block:: python

        from cluster_generator.codes.abc import SimulationCode, RuntimeParameters

        class MyHydroCode(SimulationCode):
            rtp = MyHydroRuntimeParameters()

            @property
            def unit_system(self) -> unyt.UnitSystem:
                # Return the unit system defined by this code
                pass

            def generate_ics(self, initial_conditions: ClusterICs, overwrite: bool = False, **kwargs) -> Path:
                # Implementation to generate ICs
                pass

            def from_install_directory(cls, installation_directory: str | Path, **parameters) -> Self:
                # Implementation to instantiate the class from an installation directory
                pass

    Notes
    -----
    The `SimulationCode` class is designed for developers to add support for new hydrodynamics codes
    by subclassing and implementing required methods and attributes. It leverages Python's abstract base
    classes to ensure all necessary functionality is provided for integration with the cluster generator
    framework.

    The runtime parameters (RTPs) are particularly critical, as they determine the configuration
    and behavior of the simulation. Developers should ensure that their `RuntimeParameters` class
    properly handles all necessary conversions and checks for the specific code being implemented.

    See Also
    --------
    :py:class:`RuntimeParameters` : Class responsible for managing runtime parameters.
    :py:class:`ClusterICs` : Class representing initial conditions for simulations.
    """

    rtp: ClassVar[RuntimeParameters] = None
    """ RuntimeParameters: The runtime parameters for the simulation code.

    See the :py:class:`codes.abc.RuntimeParameters` documentation for a complete overview. This attribute can be accessed on
    the class itself (yielding default RTPs stored in the ``/bin`` directory), or on an instance of the class (yielding the
    current simulation's RTPs).
    """
    # Setting up the logging system
    logger: ClassVar[logging.Logger] = LogDescriptor()

    # USER PARAMETERS
    #
    # These are the parameters the user need to provide at the time of instantiation.

    # COMPILE-TIME PARAMETERS
    #
    # All CTPs should be attributes of the class and fill-able during initialization. They should have defaults
    # provided.

    def __post_init__(self):
        # Initialize the instance._rtp attribute to store instance-level RTPs.
        # This never needs to be changed, but it can be added to if need arises.
        self._rtp = None
        self._unit_system = None

    def __str__(self):
        return f"<SimulationCode name={self.__class__.__name__}>"

    def __repr__(self):
        return self.__str__()

    def check_rtp(self, strict: bool = False) -> tuple[bool, list[Exception]]:
        """Check that all of the RTP values are values permitted by the code / cluster
        generator.

        Parameters
        ----------
        strict: bool
            If ``True``, then errors are raised. Otherwise, a boolean is returned.

        Returns
        -------
        tuple
        """
        self.logger.info(f"Checking RTPs of {self}...")
        res, err = self.__class__.get_rtp_class().check_valid(
            self, self.__class__, strict=strict
        )
        return res, err

    def check_fields(self, strict: bool = False) -> tuple[bool, list[Exception]]:
        """Check that all of the class fields (CTPs and user settings) are valid.

        Parameters
        ----------
        strict: bool
            If ``True``, then errors are raised. Otherwise, a boolean is returned.

        Returns
        -------
        tuple
        """
        self.logger.info(f"Checking fields of {self}...")
        # fetch the class's fields and determine allowed values.
        allowed_values = {
            field.name: field.metadata.get("allowed_values", None)
            for field in fields(self.__class__)
        }
        values = {
            field.name: getattr(self, field.name) for field in fields(self.__class__)
        }

        errors = []
        # Check all of the RTPs in order.
        for checked_key, allowed in allowed_values.items():
            if allowed is None:
                continue

            if values[checked_key] not in allowed:
                errors.append(
                    ValueError(
                        f"{self}-FIELD {checked_key} has value {values[checked_key]} but only {allowed} are permitted."
                    )
                )

        result = len(errors) == 0

        if strict and not result:
            from cluster_generator.utilities.logging import ErrorGroup

            raise ErrorGroup(f"{self} field check failed!", errors)

        return result, errors

    def check_params(self, strict: bool = False) -> tuple[bool, list[Exception]]:
        """
        Check all of the parameters for this simulation code: CTPs, RTPs, and user settings.

        Parameters
        ----------
        strict: bool
            If ``True``, then errors are raised. Otherwise, a boolean is returned.

        Returns
        -------
        tuple

        """
        self.logger.info(f"Checking parameters of {self}...")
        r_rtp, l_rtp = self.check_rtp(strict=False)
        r_f, l_f = self.check_fields(strict=False)

        res = r_rtp and r_f
        errors = l_rtp + l_f

        if strict and not res:
            from cluster_generator.utilities.logging import ErrorGroup

            raise ErrorGroup(f"{self} check failed!", errors)

        return res, errors

    @classmethod
    def _field_hash(cls):
        return {f.name: f for f in fields(cls)}

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

        When accessed as a class variable (``SimulationCode.rpt``), or as the attributes of a class instance (``self.rtp``), an
        **instance** of the RPT descriptor class is returned. This method provides access to the actual **class** object.
        """
        return cls.__dict__["rtp"]

    def determine_runtime_params(self, ics: ClusterICs) -> None:
        """Use the specified compile-time settings and user settings to fill in missing
        RTPs.

        Notes
        -----

        When the user instantiates a :py:class:`SimulationCode` instance, it's :py:attr:`SimulationCode.rtp` attribute is
        set to take the default values found in the configuration. The user parameters (fed to the :py:class:`SimulationCode` during
        instantiation) are then used to "fill-in-the-blanks" in the RTPs so that they reflect the simulation cluster generator is
        assisting in setting up.

        This method is the part of the :py:class:`SimulationCode` which does this.

        .. note::

            For developers, this method never needs to be changed. Under-the-hood, it's the :py:class:`RuntimeParameters` class,
            particularly the :py:meth:`RuntimeParameters.deduce_instance_values` method which actually does the leg-work here. It is
            that method which needs to be written fresh by the developer for each new code.
        """
        rtp_class: RuntimeParameters = self.get_rtp_class()
        status, error = rtp_class.deduce_instance_values(self, self.__class__, ics)

        if not status:
            raise error

    @classmethod
    def from_install_directory(
        cls, installation_directory: str | Path, **parameters
    ) -> Self:
        """Instantiate this :py:class:`SimulationCode` class with help from the
        installation directory for the corresponding simulation software.

        Parameters
        ----------
        installation_directory: str
            The directory in which the software is installed.
        parameters
            User-specified parameters expected by the :py:class:`SimulationCode` class.

            .. note::

                The :py:class:`SimulationCode` class has two types of attributes: compile-time parameters (CTPs) and
                user-specified parameters (USPs). The USPs for the class are provided in the ``parameters`` passed to this
                method. The CTPs are determined by inspecting the ``installation_directory`` and determining their value from
                the installed software.

        Returns
        -------
        SimulationCode
            The corresponding :py:class:`SimulationCode` instance.

        Notes
        -----

        .. warning::

            This method may not be implemented in all simulation software classes.
        """
        raise NotImplementedError(f"This method is not implemented for {cls.__name__}.")

    @abstractmethod
    def generate_ics(
        self, initial_conditions: ClusterICs, overwrite: bool = False, **kwargs
    ) -> Path:
        """Convert a :py:class:`ics.ClusterICs` object into a format that is ready to
        run in this simulation software.

        Parameters
        ----------
        initial_conditions: ClusterICs
            The initial conditions object to convert to the correct IC filetype for use in the simulation
            software.
        overwrite: bool, optional
            Allow the IC to overwrite existing data. ``False`` by default.
        kwargs:
            Additional kwargs defined by particular codes.

        Returns
        -------
        Path
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
        return self.get_rtp_class().write_rtp_template(
            self, type(self), path, overwrite=overwrite
        )
