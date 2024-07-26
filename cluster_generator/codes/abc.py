"""Abstract base classes for implementing hydrodynamics code support."""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Type

import unyt
from ruamel.yaml import YAML

from cluster_generator.ics import ClusterICs
from cluster_generator.utilities.io import (
    unyt_array_constructor,
    unyt_quantity_constructor,
)
from cluster_generator.utilities.logging import LogDescriptor
from cluster_generator.utilities.types import Instance, Self, Value

if TYPE_CHECKING:
    from ruamel.yaml import Loader, Node


class _YAMLDescriptor:
    # Allow for single try loading of custom yaml for an RTP class.
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
    """Super-class runtime-parameters container.

    Notes
    -----

    The :py:class:`RuntimeParameters` class for a given simulation code serves 4 functions:

    1. Read the default RTPs from disk (``/bin/code_name/runtime_parameters.yaml``) and allow the user to access them.
    2. Store a set of RTPs for each instance of the :py:class:`SimulationCode` which the user may manipulate to suite their use-case.
    3. Provide a way for :py:class:`SimulationCode` instances to "fill-in" missing RTPs given the user's USPs.
    4. Provide a method for writing the RTPs for a simulation to disk in a ready-to-run format.

    To achieve this, the :py:class:`RuntimeParameters` class is a `descriptor class <https://realpython.com/python-descriptors/>`_.
    Every :py:class:`SimulationCode` has a :py:attr:`~SimulationCode.rtp` attribute which can be accessed to see a set of
    the default RTPs for the simulation code.

    Once the :py:class:`SimulationCode` has been instantiated, (i.e. ``code = SimulationCode()``), accessing the :py:attr:`~SimulationCode.rtp`
    attribute (``code.rtp``) will provide access to the **actual RTPs** for the simulation run represented by that instance. These can be
    changed by the user as needed.

    .. hint::

        RTPs (when accessed by the class or the instance) are permitted to be **any** legitimate class. For example, a softening length
        parameter may have a default which is an :py:class:`unyt.unyt_quantity`. It will be "converted" to the correct code-format
        only when the user creates an RTP template to feed into the code (:py:meth:`SimulationCode.generate_rtp_template`).

    Once the user is ready, the :py:meth:`SimulationCode.determine_runtime_params` uses the CTP and USP of the :py:class:`SimulationCode` instance
    to deduce the correct RTP values (fill-in-the-blank) for the user's simulation. Finally, the :py:meth:`SimulationCode.generate_rtp_template` method
    will export those RTPs in a format recognized by the simulation software.

    .. rubric:: Setting RTPs

    For each RTP recognized by the code, the developer can write a method in the RTP class named ``set_<RTP_name>`` with 3 arguments:
    ``instance:SimulationCode``, ``owner: Type[SimulationCode]``, and ``ic: ClusterICs``. This should return the correct value of
    the given RTP as deduced from the ICs and the code object.

    The :py:meth:`RuntimeParameter.deduce_instance_values` then identifies all of these "setters" and uses them to set all
    of the needed RTPs at runtime. Additionally, if a field in the :py:class:`SimulationCode` class corresponds **directly** to
    an RTP, then it can be specified with a "setter" parameter in its metadata using the syntax like

    .. code-block:: python

        class SimulationCode:

            FIELD: Any = ufield(
                        default=None,            # Default field value
                        flag="RTP_name",         # RTP name
                        setter= setter_function  # The setter function.
                        )

    .. rubric:: Converting RTPs

    Once the RTPs for an instance are set, they are still (generically) in any number of different classes. If the particular
    simulation code expects a class ``C`` to be written in a particular format that is not the default (``__repr__``) of the
    class, then a custom "converter" must be provided. To do so, we simply implement a method with the signature
    ``generic_converter(value) -> str`` which converts a particular type to its correct string representation. To indicate
    the type that should be converted, below the class definition, add the following:

    .. code-block:: python

        RuntimeParameters.<converter_method> = RuntimeParameters._converter(type)(RuntimeParameters.<converter_method>)
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
    """Generic abstract class representing a specific (magneto)hydrodynamics code
    frontend.

    Notes
    -----

    :py:class:`SimulationCode` classes are ``dataclasses`` and have two types of parameters.

    - **Compile-Time Parameters** (CTPs) are (generally optional) parameters that the user specifies when installing the
      software.

    - **User Specified Parameters** (USPs) are the parameters that ``cluster_generator`` needs to figure out how to format your
      initial conditions, where to put them, and all the other necessary information.

    All of these parameters can be specified simply as ``keyword arguments`` when initializing the class.


    In order to run, the simulation code needs a bunch of **runtime parameters**. These are set to their default values when
    you initialize the class (:py:attr:`SimulationCode.rtp`), but can be changed at any time. When you're ready to create ICs for the
    code, you should call the :py:meth:`SimulationCode.determine_runtime_params` method, which will use the USPs and CTPs to fill in any
    missing or incorrect RTPs.

    Once you've filled in the RTPs, you can export them in a format ready for use in the simulation code using the :py:meth:`SimulationCode.generate_rtp_template`
    method.

    With RTPs out of the way, we can now convert any :py:class:`ics.ClusterICs` object into a format that the simulation software recognizes by calling the
    :py:meth:`SimulationCode.generate_ics` method. This will do all the work necessary to get the simulation ready to run.
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
