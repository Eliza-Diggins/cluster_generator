import importlib
from typing import Type, TypeVar, Union

from numpy.typing import NDArray

# Generic type variables for flexibility
Profile = TypeVar("Profile")
ProfileResult = TypeVar("ProfileResult", float, NDArray[float], int, NDArray[int])
ProfileInput = ProfileResult


class MissingOptionalAttribute:
    """
    Sentinel descriptor that raises a NotImplementedError when accessed.
    Used to enforce abstract class attributes.

    This descriptor is used to enforce that subclasses must define certain
    optional attributes. If the attribute is accessed without being defined,
    a `NotImplementedError` is raised.

    Attributes
    ----------
    _name : str
        Name of the attribute.

    Examples
    --------
    .. code-block:: python

        class Example:
            missing_attribute = MissingOptionalAttribute()

            def __init__(self):
                print(self.missing_attribute)

        # Will raise NotImplementedError:
        e = Example()
    """

    def __set_name__(self, owner: Type["Profile"], name: str):
        self._name: str = name

    def __get__(self, instance: "Profile", owner: Type["Profile"]):
        raise NotImplementedError(f"{owner.__name__} does not define {self._name}.")


class ProfileParameter:
    """
    Descriptor class for handling profile parameters with optional default values.

    This class allows defining parameters for profiles, providing a default value if none is set,
    and raising an error when a parameter is missing without a default.

    Parameters
    ----------
    default : Union[float, int], optional
        The default value for the profile parameter, by default None.
    desc : str, optional
        A description of the parameter for documentation purposes, by default None.

    Attributes
    ----------
    name : str
        Name of the attribute this descriptor represents, set during class definition.

    Examples
    --------
    .. code-block:: python

        class ExampleProfile:
            amplitude = ProfileParameter(default=1.0, desc="Amplitude of the profile")

        profile = ExampleProfile()
        print(profile.amplitude)  # Outputs the default value of 1.0
    """

    def __init__(self, default: Union[float, int] = None, desc: str = None):
        self._default: Union[float, int, None] = default
        self._desc: Union[str, None] = desc

    def __set_name__(self, owner: Type["Profile"], name: str):
        self.name: str = name

    def __get__(self, instance: "Profile", owner: Type["Profile"]) -> Union[float, int]:
        """
        Get the value of the profile parameter from the instance's parameters.
        If the parameter is missing, the default value will be used if provided.

        Parameters
        ----------
        instance : Profile
            The instance from which the parameter value is accessed.
        owner : Type[Profile]
            The class that owns this descriptor.

        Returns
        -------
        Union[float, int]
            The value of the profile parameter.

        Raises
        ------
        ValueError
            If the parameter is not set and no default value exists.
        """
        if self.name in instance._parameters:
            return instance._parameters[self.name]
        elif self._default is not None:
            instance._parameters[self.name] = self._default
            return self._default
        else:
            raise ValueError(
                f"Parameter '{self.name}' has no value in {instance} and no default exists."
            )


class ProfileLinkDescriptor:
    """
    Descriptor that links a Profile class to another class or instance.

    - When accessed on the class, returns the linked Profile class.
    - When accessed on an instance, returns the Profile class initialized
      with the same parameters as the instance.

    Parameters
    ----------
    profile_class : class or str, optional
        The Profile class or its name (in the form of a string) to link to the class or instance.
        If a string is provided, it must be in the format 'module_name.ClassName'.
        If not provided, a ValueError will be raised when accessed.

    module : str, optional
        If provided, this will be the module where the profile class is defined. This is used for lazy importing.

    Attributes
    ----------
    profile_class : class or str
        The Profile class this descriptor links to or the name of the class to be imported.

    Examples
    --------
    .. code-block:: python

        class ParentProfile:
            def __init__(self, **kwargs):
                self._parameters = kwargs

        class LinkedProfile:
            def __init__(self, **kwargs):
                self._parameters = kwargs

            def __repr__(self):
                return f"LinkedProfile({self._parameters})"

        class ExampleClass(ParentProfile):
            linked_profile = ProfileLinkDescriptor(LinkedProfile)

        # Accessing via class
        print(ExampleClass.linked_profile)  # Outputs: <class '__main__.LinkedProfile'>

        # Accessing via an instance
        instance = ExampleClass(param1=5, param2=10)
        print(instance.linked_profile)  # Outputs: LinkedProfile({'param1': 5, 'param2': 10})
    """

    def __init__(
        self, profile_class: Union[str, Type["Profile"]] = None, module: str = None
    ):
        """
        Initialize the descriptor with a Profile class or module name.

        Parameters
        ----------
        profile_class : class or str, optional
            The Profile class or its name (in the form of a string) to link to the class or instance.
            If a string is provided, it must be in the format 'ClassName' or 'module_name.ClassName'.
        module : str, optional
            The module where the profile class is defined for lazy importing.
        """
        self.profile_class = profile_class
        self.module_name = module

    def _lazy_import(self):
        """
        Lazily imports the profile class if it's not already loaded.

        Returns
        -------
        type
            The Profile class.

        Raises
        ------
        ValueError
            If the class or module could not be imported.
        """
        if isinstance(self.profile_class, str):
            if not self.module_name:
                raise ValueError(
                    f"Module name must be provided for lazy importing of {self.profile_class}."
                )
            try:
                module = importlib.import_module(self.module_name)
                self.profile_class = getattr(module, self.profile_class)
            except (ImportError, AttributeError) as e:
                raise ValueError(
                    f"Could not import {self.profile_class} from {self.module_name}: {e}"
                )
        return self.profile_class

    def __set_name__(self, owner, name):
        """
        Set the name of the descriptor when it is added to a class.

        Parameters
        ----------
        owner : class
            The class to which the descriptor belongs.
        name : str
            The name of the attribute in the class.
        """
        self.name = name

    def __get__(self, instance, owner):
        """
        Descriptor behavior for accessing the attribute.

        If accessed from the class, returns the Profile class itself.
        If accessed from an instance, returns the Profile class initialized
        with the same parameters as the instance.

        Parameters
        ----------
        instance : object
            The instance of the class.
        owner : class
            The owner class.

        Returns
        -------
        Profile or Profile instance
            The Profile class or an initialized Profile instance.

        Raises
        ------
        ValueError
            If no Profile class was provided at initialization.
        """
        profile_class = (
            self._lazy_import()
            if isinstance(self.profile_class, str)
            else self.profile_class
        )

        if profile_class is None:
            raise ValueError(
                f"There is no {self.name} profile link for {owner.__name__}."
            )

        if instance is None:
            # If accessed via the class, return the Profile class itself.
            return profile_class
        else:
            # If accessed via an instance, initialize the Profile class with the instance's parameters.
            return profile_class(**instance._parameters)
