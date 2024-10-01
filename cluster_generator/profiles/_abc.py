"""
Profile Module for Cluster Generator
====================================

This module defines the abstract base classes for profiles, specifically scalar field
profiles dependent on spatial coordinates. These profiles can be customized to fit
various geometries such as radial profiles (e.g., spherical) and other coordinate systems.

Classes
-------
- Profile: Abstract base class for scalar field profiles.
- RadialProfile: A specialized subclass for profiles with radial symmetry.

Each profile class provides:
- Methods for evaluating profiles at specific coordinates.
- Arithmetic operations between profiles and scalars.
- Plotting capabilities for both 1D and 2D projections using Matplotlib.

Dependencies
------------
- numpy: For numerical operations and handling arrays.
- matplotlib: For plotting functionalities.

Module Contents
---------------
This module includes the `Profile` class, which represents scalar fields dependent
on spatial coordinates, and the `RadialProfile` class, which reduces the profile to
a function of a single radial coordinate.

See Also
--------
:py:class:`RadialProfile` : A subclass for radial profiles with spherical symmetry.
:py:class:`ProfileParameter` : Descriptor class for defining parameters of a profile.
:py:func:`matplotlib.pyplot.plot` : Function for creating 1D plots.
:py:func:`matplotlib.pyplot.imshow` : Function for displaying 2D image data.

Examples
--------
Creating a custom radial profile:

.. code-block:: python

    from cluster_generator.profiles.base import RadialProfile
    import numpy as np

    class CustomRadialProfile(RadialProfile):
        PARAMETERS = {
            'amplitude': {'dtype': float, 'units': None},
            'scale': {'dtype': float, 'units': None}
        }
        PRINCIPLE_FUNCTIONS = {
            'Spherical': lambda r, amplitude, scale: amplitude * np.exp(-r / scale)
        }

    profile = CustomRadialProfile(amplitude=1.0, scale=5.0)
    r = np.linspace(0.1, 10, 100)
    values = profile(r)
    print(values)

Notes
-----
The module assumes subclasses of `Profile` will define profile parameters
using the `ProfileParameter` descriptors. These parameters must be validated
and resolved to create callable profiles compatible with different geometries.
"""

import pickle
from abc import ABC, ABCMeta
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type, Union

import h5py
from numpy.typing import NDArray
from ruamel.yaml import YAML

from cluster_generator.geometry.radial import (
    RadialGeometryHandler,
    SphericalGeometryHandler,
)
from cluster_generator.profiles._types import (
    ProfileInput,
    ProfileParameter,
    ProfileResult,
)

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes, Figure

    from cluster_generator.geometry._abc import GeometryHandler

yaml = YAML()


class ProfileRegistry:
    """
    A class to maintain a registry of profile classes.

    This class provides methods to register, retrieve, and list all registered
    profile classes.

    Attributes
    ----------
    registry : dict
        A dictionary where the keys are profile class names and the values
        are profile class objects.
    """

    def __init__(self):
        """
        Initialize an empty profile registry.
        """
        self.registry = {}

    def register(self, profile_class: "ProfileMeta"):
        """
        Register a profile class in the registry.

        Parameters
        ----------
        profile_class : Type[Profile]
            The profile class to register.

        Raises
        ------
        ValueError
            If a profile class with the same name already exists in the registry.
        """
        profile_name = profile_class.__name__
        if profile_name in self.registry:
            raise ValueError(f"Profile class '{profile_name}' is already registered.")
        self.registry[profile_name] = profile_class

    def get(self, profile_name: str) -> Type["Profile"]:
        """
        Retrieve a profile class by its name.

        Parameters
        ----------
        profile_name : str
            The name of the profile class to retrieve.

        Returns
        -------
        Type[Profile]
            The profile class corresponding to the provided name.

        Raises
        ------
        KeyError
            If the profile class is not found in the registry.
        """
        if profile_name not in self.registry:
            raise KeyError(f"Profile class '{profile_name}' is not registered.")
        return self.registry[profile_name]

    def list_profiles(self) -> list:
        """
        List all registered profile class names.

        Returns
        -------
        list
            A list of all registered profile class names.
        """
        return list(self.registry.keys())


DEFAULT_PROFILE_REGISTRY = ProfileRegistry()


class ProfileMeta(ABCMeta):
    """
    Metaclass for automatically registering profile classes when they are created.
    """

    def __init__(cls, name, bases, clsdict):
        """
        Called when the class is created. Registers the class with the default profile registry.

        Parameters
        ----------
        name : str
            Name of the class being created.
        bases : tuple
            Base classes for the new class.
        clsdict : dict
            Dictionary of attributes and methods for the class.
        """
        super().__init__(name, bases, clsdict)

        # Automatically register the class when it is created
        if cls.__name__ not in [
            "Profile",
            "RadialProfile",
            "RadialDensityProfile",
            "RadialTemperatureProfile",
            "RadialMassProfile",
        ] and (
            cls.__name__ not in DEFAULT_PROFILE_REGISTRY.registry
        ):  # Avoid registering the abstract base classes
            DEFAULT_PROFILE_REGISTRY.register(cls)


class Profile(metaclass=ProfileMeta):
    """
    Abstract base class for scalar field profiles over spatial coordinates.

    This class provides the framework for defining and working with profiles,
    which represent scalar fields dependent on spatial coordinates, with
    dimensionality and functional behavior tied to the provided geometry.

    Subclasses should define the profile's function(s) using `CLASS_FUNCTIONS` and
    parameters via `ProfileParameter` descriptors. Profiles can be evaluated at
    specific coordinates using the call syntax and support basic arithmetic operations.

    Attributes
    ----------
    NDIMS : int
        Expected number of dimensions for the geometry.
    CLASS_FUNCTIONS : dict
        Mapping of geometry names to functions defining the profile.

    See Also
    --------
    :py:class:`RadialProfile` : A subclass representing radial profiles with spherical symmetry.
    :py:class:`ProfileParameter` : Descriptor class for defining parameters of a profile.

    Notes
    -----
    This class assumes that subclasses will define profile parameters using descriptors.
    The parameters and associated profile functions must be validated and resolved to
    create callable profiles, compatible with different geometries.
    """

    # FUNCTION PARAMETERS:
    # Each of these should be named the same as they appear in the
    # CLASS_FUNCTIONS, and should be equal to a ProfileParameter() instance.

    # CLASS ATTRIBUTES
    NDIMS: int = None
    CLASS_FUNCTIONS: Dict[str, Callable[..., ProfileParameter]] = None

    def __init__(self, geometry: "GeometryHandler" = None, **kwargs):
        """
        Initialize the `Profile` instance.

        Parameters
        ----------
        geometry : GeometryHandler, optional
            The geometry in which the profile is defined. Default is None.
        registry : ProfileRegistry, optional
            A registry where the profile class will be registered.
        **kwargs:
            Additional keyword arguments corresponding to profile parameters,
            which should be defined in the `PARAMETERS` dictionary of the subclass.

        Raises
        ------
        ValueError
            If the geometry's dimensions do not match the profile's `NDIMS`.
        ValueError
            If extra or missing parameters are provided.
        """
        # Set the geometry handler and validate it.
        self.geometry_handler: "GeometryHandler" = (
            geometry or SphericalGeometryHandler()
        )
        self._validate_geometry_handler()

        # Set the parameters up and validate them.
        self._parameters: Dict[str, Any] = kwargs
        self._validate_parameters()

        # Resolve the underlying function.
        self._function = self._validate_function()

    def __call__(self, *args: ProfileInput) -> ProfileResult:
        """
        Evaluate the profile at the given coordinates.

        Parameters
        ----------
        *args : tuple
            The coordinates at which to evaluate the profile.

        Returns
        -------
        ProfileResult
            The result of evaluating the profile at the given coordinates.
        """
        print(args)
        return self._function(*args)

    def __repr__(self) -> str:
        """
        Return a string representation of the Profile instance.

        Returns
        -------
        str
            A string representation showing the class name and parameters.
        """
        param_str = ", ".join(f"{k}={v}" for k, v in self._parameters.items())
        return f"<{self.__class__.__name__}({param_str}, geometry='{self.geometry_handler.NAME}')>"

    def _do_operation(
        self, other: Union["Profile", float, int], op: Callable
    ) -> Type["Profile"]:
        """
        Perform arithmetic operations between profiles or between a profile and a scalar.

        Parameters
        ----------
        other : Union[Profile, float, int]
            The other operand, which can be a Profile instance or a scalar.
        op : Callable
            The operation to perform (e.g., addition, multiplication).

        Returns
        -------
        Profile
            A new Profile instance resulting from the operation.
        """
        current_func = self._function

        if isinstance(other, Profile):
            if self.geometry_handler != other.geometry_handler:
                raise ValueError(
                    "Cannot perform operation on Profiles with different geometry systems."
                )
            other_func = other._function
            new_func = lambda *args: op(current_func(*args), other_func(*args))
        else:
            new_func = lambda *args: op(current_func(*args), other)

        return Profile.create_profile(new_func)

    def __add__(self, other: Union["Profile", float]) -> "Profile":
        return self._do_operation(other, lambda x, y: x + y)

    def __radd__(self, other: float) -> "Profile":
        return self.__add__(other)

    def __sub__(self, other: Union["Profile", float]) -> "Profile":
        return self._do_operation(other, lambda x, y: x - y)

    def __rsub__(self, other: float) -> "Profile":
        return self._do_operation(other, lambda x, y: y - x)

    def __mul__(self, other: Union["Profile", float]) -> "Profile":
        return self._do_operation(other, lambda x, y: x * y)

    def __rmul__(self, other: float) -> "Profile":
        return self.__mul__(other)

    def __truediv__(self, other: Union["Profile", float]) -> "Profile":
        return self._do_operation(other, lambda x, y: x / y)

    def __rtruediv__(self, other: float) -> "Profile":
        return self._do_operation(other, lambda x, y: y / x)

    def _validate_geometry_handler(self):
        """
        Validate the geometry handler to ensure dimensional compatibility.
        Raises an error if the dimensions do not match the expected `NDIMS`.
        """
        geometry_dimensions = len(self.geometry_handler.FREE_AXES)

        if geometry_dimensions != self.__class__.NDIMS:
            raise AttributeError(
                f"The geometry handler {self.geometry_handler} has {geometry_dimensions} dimensions, "
                f"but {self.__class__.__name__} expected {self.__class__.NDIMS}."
            )

    def _validate_parameters(self):
        """
        Validate the parameters provided in the constructor against the defined
        parameter descriptors in the class. Raises errors for missing or extra parameters.
        """
        # Get parameter descriptors for the class
        parameter_descriptors = self.__class__._get_parameter_descriptors()

        # Ensure that all required parameters are provided
        for parameter_name, _ in parameter_descriptors.items():
            # Automatically fetch default values or raise errors if needed
            _ = getattr(self, parameter_name)

        # Check for unexpected parameters
        unexpected_params = set(self._parameters.keys()) - set(
            parameter_descriptors.keys()
        )
        if unexpected_params:
            raise ValueError(f"Unexpected parameters provided: {unexpected_params}.")

    def _validate_function(self):
        """
        Validate and return the correct function from `CLASS_FUNCTIONS` based on the geometry.
        Raises an error if no compatible function is found.

        Returns
        -------
        Callable
            The resolved function for the profile.
        """
        # Fetch the default function or the one corresponding to the geometry
        default_function = self.__class__.CLASS_FUNCTIONS.get("default", None)
        function = self.__class__.CLASS_FUNCTIONS.get(
            self.geometry_handler.NAME, default_function
        )

        if function is None:
            raise AttributeError(
                f"Failed to find a profile formula in {self.__class__.__name__} compatible with {self.geometry_handler}."
            )

        # Bind the parameters to the function
        return partial(function, **self._parameters, **self.geometry_handler.parameters)

    @classmethod
    def _get_parameter_descriptors(cls) -> Dict[str, ProfileParameter]:
        """
        Retrieve all descriptors of type `ProfileParameter` from the class and its ancestors.

        Returns
        -------
        dict
            A dictionary mapping attribute names to descriptor instances.
        """
        descriptors = {}

        # Traverse the class hierarchy
        for klass in cls.__mro__:
            for attr_name, attr_value in klass.__dict__.items():
                if isinstance(attr_value, ProfileParameter):
                    descriptors[attr_name] = attr_value

        return descriptors

    @classmethod
    def create_profile_class(
        cls,
        func: Union[Callable, Dict[str, Callable]],
        ndim: int = 1,
        name: str = "CustomProfile",
        **parameters,
    ):
        """
        Dynamically create a new subclass of `Profile` using a provided function or
        dictionary of functions for different geometries.

        This method allows you to create a custom profile class dynamically by passing a callable
        function or a dictionary of functions that define the profile for different geometries.
        It also accepts additional keyword arguments as parameters for the profile, which will
        automatically be added as class attributes.

        Parameters
        ----------
        func : Union[Callable, Dict[str, Callable]]
            A callable function representing the profile, or a dictionary where keys are geometry types
            (e.g., 'Spherical', 'Cartesian') and values are callable functions for each geometry type.
        ndim : int, optional
            The number of dimensions for the profile (e.g., 1 for 1D, 2 for 2D, etc.). Default is 1.
        name : str, optional
            The name of the dynamically generated profile class. Default is "CustomProfile".
        **parameters:
            Additional keyword arguments defining parameters for the profile. These are
            automatically added as class attributes of type `ProfileParameter`.

        Returns
        -------
        Type[Profile]
            A new subclass of `Profile` with the specified function(s) and parameters.

        Examples
        --------
        Create a profile class for an exponential decay:

        .. code-block:: python

            def exponential_profile(r, amplitude, scale):
                return amplitude * np.exp(-r / scale)

            # Dynamically create a profile class
            ExponentialProfile = Profile.create_profile_class(
                exponential_profile,
                name="ExponentialProfile",
                amplitude=1.0,
                scale=5.0
            )

            # Instantiate the profile and evaluate at r = 1.0
            profile = ExponentialProfile()
            result = profile(1.0)
            print(result)

        Create a profile class with different functions for different geometries:

        .. code-block:: python

            # Define functions for different geometries
            func_dict = {
                'Spherical': lambda r, amplitude: amplitude / (r**2),
                'Cartesian': lambda x, y, amplitude: amplitude / (x**2 + y**2)
            }

            # Dynamically create a multi-geometry profile class
            MultiGeometryProfile = Profile.create_profile_class(
                func_dict,
                name="MultiGeometryProfile",
                amplitude=10.0
            )

            # Instantiate with a specific geometry handler
            profile = MultiGeometryProfile(geometry=SphericalGeometryHandler)
            result = profile(1.0)  # Evaluate in spherical coordinates
            print(result)

        Notes
        -----
        The `func` argument can either be a single callable (for simple profiles) or a dictionary of
        functions keyed by geometry type (e.g., 'Spherical', 'Cartesian'). The additional `**parameters`
        keyword arguments are automatically converted into class attributes of type `ProfileParameter`.
        """
        # If a single function is provided, create a default geometry mapping
        if isinstance(func, Callable):
            class_functions = {"default": func}
        elif isinstance(func, dict):
            class_functions = func
        else:
            raise ValueError(
                "`func` must be a callable or a dictionary of callables for different geometries."
            )

        # Define the new class attributes (parameters and class functions)
        class_attributes = {
            "NDIMS": ndim,
            "CLASS_FUNCTIONS": class_functions,
            **{
                param_name: ProfileParameter(default=param_value)
                for param_name, param_value in parameters.items()
            },
        }

        # Dynamically create the new class
        return type(name, (cls,), class_attributes)

    def to_array(self, *args: NDArray) -> NDArray:
        """
        Evaluate the profile at the given array of points and return the results as a NumPy array.

        Parameters
        ----------
        *args:
            The meshgrid on which to evaluate the profile. Thus, if ``X`` and ``Y`` are the profile meshgrid,
            then ``to_array(X,Y)`` will produce ``Z`` of the same shape.

        Returns
        -------
        NDArray
            A NumPy array containing the evaluated profile at each point.

        Examples
        --------
        .. code-block:: python

            # Create a radial profile and evaluate it over a grid of radii
            profile = CustomRadialProfile(amplitude=1.0, scale=5.0)
            radii = np.linspace(0.1, 10, 100)
            profile_values = profile.to_array(radii)
            print(profile_values)
        """
        return self.__call__(*args)

    def to_yaml(self, file_path: str) -> None:
        """
        Serialize the profile instance to a YAML file. Only the class name and parameters are stored.

        Parameters
        ----------
        file_path : str
            The path to the file where the YAML representation of the profile will be written.

        Returns
        -------
        None
        """
        data = {"class_name": self.__class__.__name__, "parameters": self._parameters}
        with open(file_path, "w") as file:
            yaml.dump(data, file)

    def to_dict(self) -> str:
        """
        Serialize the profile instance to a YAML string. Only the class name and parameters are stored.

        Returns
        -------
        str
            The YAML representation of the profile.
        """
        data = {"class_name": self.__class__.__name__, "parameters": self._parameters}
        return yaml.dump(data)

    @classmethod
    def from_dict(
        cls, data: dict, registry: ProfileRegistry = DEFAULT_PROFILE_REGISTRY
    ) -> "Profile":
        """
        Deserialize a profile from a dictionary.

        Parameters
        ----------
        data : dict
            The dictionary containing the serialized profile data (class name and parameters).
        registry : ProfileRegistry, optional
            The registry to use for looking up the profile class. Defaults to DEFAULT_PROFILE_REGISTRY.

        Returns
        -------
        Profile
            The deserialized profile instance.

        Raises
        ------
        KeyError
            If the profile class is not found in the registry.
        """
        profile_class = registry.get(data["class_name"])
        return profile_class(**data["parameters"])

    @classmethod
    def from_yaml(
        cls, file_path: str, registry: ProfileRegistry = DEFAULT_PROFILE_REGISTRY
    ) -> "Profile":
        """
        Deserialize a profile from a YAML file.

        Parameters
        ----------
        file_path : str
            The path to the YAML file to deserialize.
        registry : ProfileRegistry, optional
            The registry to use for looking up the profile class. Defaults to DEFAULT_PROFILE_REGISTRY.

        Returns
        -------
        Profile
            The deserialized profile instance.

        Raises
        ------
        KeyError
            If the profile class is not found in the registry.
        """
        with open(file_path, "r") as file:
            data = yaml.load(file)
        profile_class = registry.get(data["class_name"])
        return profile_class(**data["parameters"])

    def to_pickle(self) -> bytes:
        """
        Serialize the profile instance to a binary pickle format.

        Returns
        -------
        bytes
            The pickled binary representation of the profile.
        """
        data = {"class_name": self.__class__.__name__, "parameters": self._parameters}
        return pickle.dumps(data)

    @classmethod
    def from_pickle(
        cls, pickle_data: bytes, registry: ProfileRegistry = DEFAULT_PROFILE_REGISTRY
    ) -> "Profile":
        """
        Deserialize a profile from a pickle binary string.

        Parameters
        ----------
        pickle_data : bytes
            The pickled data to deserialize.
        registry : ProfileRegistry, optional
            The registry to use for looking up the profile class. Defaults to DEFAULT_PROFILE_REGISTRY.

        Returns
        -------
        Profile
            The deserialized profile instance.

        Raises
        ------
        KeyError
            If the profile class is not found in the registry.
        """
        data = pickle.loads(pickle_data)
        profile_class = registry.get(data["class_name"])
        return profile_class(**data["parameters"])

    def to_hdf5(self, hdf5_file: Union[str, h5py.File], group_path: str) -> None:
        """
        Save the profile instance to the specified HDF5 file and group path.

        Parameters
        ----------
        hdf5_file : str or h5py.File
            The path to the HDF5 file or an open h5py.File object where the profile will be saved.
        group_path : str
            The path within the HDF5 file where the profile will be stored.

        Notes
        -----
        This method saves the class name and parameters to the specified group path within
        the HDF5 file. The geometry itself should be handled separately and is not stored here.
        """
        # Determine if we need to open the file ourselves
        close_file = isinstance(hdf5_file, str)
        if close_file:
            hdf5_file = h5py.File(hdf5_file, "a")

        try:
            # Create or access the specified group within the HDF5 file
            group = hdf5_file.require_group(group_path)

            # Store the profile class name as an attribute
            group.attrs["class_name"] = self.__class__.__name__

            # Store the profile parameters as attributes
            for key, value in self._parameters.items():
                group.attrs[key] = value

        finally:
            if close_file:
                hdf5_file.close()

    @classmethod
    def from_hdf5(
        cls,
        hdf5_file: Union[str, h5py.File],
        group_path: str,
        geometry: Optional["GeometryHandler"] = None,
    ) -> "Profile":
        """
        Load a profile instance from the specified HDF5 file and group path.

        Parameters
        ----------
        hdf5_file : str or h5py.File
            The path to the HDF5 file or an open h5py.File object from which the profile will be loaded.
        group_path : str
            The path within the HDF5 file where the profile is stored.
        geometry : GeometryHandler, optional
            An optional geometry handler to be associated with the profile. If not provided,
            a default geometry handler is used.

        Returns
        -------
        Profile
            An instance of the profile object initialized from the HDF5 file.

        Raises
        ------
        ValueError
            If the class specified in the HDF5 file cannot be found.
        """
        # Determine if we need to open the file ourselves
        close_file = isinstance(hdf5_file, str)
        if close_file:
            hdf5_file = h5py.File(hdf5_file, "r")

        try:
            # Access the specified group within the HDF5 file
            group = hdf5_file[group_path]

            # Retrieve the profile class name from attributes
            class_name = group.attrs["class_name"]

            # Get the profile class by name
            profile_class = cls._get_class_by_name(class_name)
            if profile_class is None:
                raise ValueError(
                    f"Unknown profile class '{class_name}' in file '{hdf5_file}'."
                )

            # Load the parameters for the profile from the group attributes
            parameters = {
                key: value for key, value in group.attrs.items() if key != "class_name"
            }

            # Instantiate the profile object with loaded parameters and geometry handler
            return profile_class(geometry=geometry, **parameters)

        finally:
            if close_file:
                hdf5_file.close()

    @staticmethod
    def _get_class_by_name(class_name: str) -> Type["Profile"]:
        """
        Retrieve a profile class by its name.

        Parameters
        ----------
        class_name : str
            The name of the profile class.

        Returns
        -------
        Type[Profile]
            The class type corresponding to the provided class name, or None if not found.
        """

        def find_in_subclasses(base_class):
            """
            Recursively search subclasses for the class with the given name.

            Parameters
            ----------
            base_class : Type[Profile]
                The class to start the search from.

            Returns
            -------
            Type[Profile] or None
                The class type if found, otherwise None.
            """
            for subclass in base_class.__subclasses__():
                if subclass.__name__ == class_name:
                    return subclass
                # Recursively search in subclasses of the current subclass
                result = find_in_subclasses(subclass)
                if result is not None:
                    return result
            return None

        # Start the search from the base Profile class
        return find_in_subclasses(Profile)


class RadialProfile(Profile, ABC):
    """
    Abstract class for profiles with radial symmetry, reducing the profile to
    a function of a single radial coordinate.

    Attributes
    ----------
    NDIMS : int
        The number of free dimensions in the profile, set to 1 for radial profiles.
    """

    NDIMS = 1

    def __init__(self, geometry: RadialGeometryHandler = None, **kwargs):
        """
        Initialize the `Profile` instance.

        Parameters
        ----------
        geometry : RadialGeometryHandler, optional
            The geometry in which the profile is defined. Default is None.
        **kwargs:
            Additional keyword arguments corresponding to profile parameters,
            which should be defined in the `PARAMETERS` dictionary of the subclass.

        Raises
        ------
        ValueError
            If the geometry's dimensions do not match the profile's `NDIMS`.
        ValueError
            If extra or missing parameters are provided.
        """
        # Set the geometry handler and validate it.
        super(RadialProfile, self).__init__(geometry, **kwargs)

        # Set the geometry handler to RadialGeometryHandler type
        if not isinstance(self.geometry_handler, RadialGeometryHandler):
            raise TypeError(
                f"Expected a RadialGeometryHandler, but got {type(self.geometry_handler)}"
            )

        self.geometry_handler: RadialGeometryHandler = self.geometry_handler

    def __call__(self, r: ProfileInput) -> ProfileResult:
        """
        Evaluate the radial profile at the given radius.

        Parameters
        ----------
        r : ProfileInput
            The radius at which to evaluate the profile.

        Returns
        -------
        ProfileResult
            The profile values evaluated at the input radius.
        """
        return self._function(r)

    def plot_1d(
        self,
        rmin: float,
        rmax: float,
        num_points: int = 1000,
        xscale: str = "linear",
        yscale: str = "linear",
        fig=None,
        ax=None,
        **kwargs,
    ) -> tuple["Figure", "Axes"]:
        """
        Plot the radial profile as a 1D function of radius.

        This method plots the profile evaluated over a range of radius values
        using Matplotlib. The radius values are sampled on a geometric scale,
        and the plot scales for both axes can be specified.

        Parameters
        ----------
        rmin : float
            The minimum radius for the plot.
        rmax : float
            The maximum radius for the plot.
        num_points : int, optional
            The number of points to sample between `rmin` and `rmax`. Default is 1000.
        xscale : {'linear', 'log'}, optional
            Scaling for the x-axis. Default is 'linear'.
        yscale : {'linear', 'log'}, optional
            Scaling for the y-axis. Default is 'linear'.
        fig : matplotlib.figure.Figure, optional
            An optional Matplotlib Figure object. A new figure is created if None is provided.
        ax : matplotlib.axes.Axes, optional
            An optional Matplotlib Axes object. A new axis is created if None is provided.
        **kwargs:
            Additional keyword arguments passed to `matplotlib.pyplot.plot`.

        Returns
        -------
        tuple
            A tuple containing the Matplotlib figure and axis objects.

        Examples
        --------
        .. code-block:: python

            import numpy as np
            from cluster_generator.profiles.base import RadialProfile

            class ExponentialRadialProfile(RadialProfile):
                PARAMETERS = {'amplitude': {'dtype': float}, 'scale': {'dtype': float}}
                PRINCIPLE_FUNCTIONS = {
                    'Spherical': lambda r, amplitude, scale: amplitude * np.exp(-r / scale)
                }

            profile = ExponentialRadialProfile(amplitude=1.0, scale=5.0)
            fig, ax = profile.plot_1d(0.1, 10)
            plt.show()

        See Also
        --------
        :py:func:`matplotlib.pyplot.plot` : The Matplotlib function used to plot the profile.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if fig is None:
            fig = plt.figure(figsize=(8, 6))
        if ax is None:
            ax = fig.add_subplot(111)

        # Generate radius values between rmin and rmax
        r = np.geomspace(rmin, rmax, num_points)
        profile_values = self(r)

        # Plot the radial profile
        ax.plot(r, profile_values, **kwargs)

        # Determine the radius label based on the coordinate system
        _r_symbol = r"\xi" if self.geometry_handler.NAME != "Spherical" else "r"
        _r_label = (
            "Effective Radius"
            if self.geometry_handler.NAME != "Spherical"
            else "Radius"
        )

        # Set axis labels and scales
        ax.set_xlabel(f"{_r_label}, ${_r_symbol}$ / $[\\text{{kpc}}]$")
        ax.set_ylabel(f"$f({_r_symbol})$")
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        return fig, ax

    def plot_2d(
        self,
        extent: float,
        num_points: int = 100,
        fig=None,
        ax=None,
        viewing_axes: str = "xy",
        **kwargs,
    ) -> tuple[NDArray, "Figure", "Axes"]:
        """
        Plot the 2D projection of the radial profile in the specified coordinate system.

        This method visualizes the 2D projection of the profile by sampling points over
        the specified extent and projecting them onto the selected axes. The profile values
        are evaluated at these points and displayed as a color map using Matplotlib.

        Parameters
        ----------
        extent : float
            The extent of the plot in both x and y directions.
        num_points : int, optional
            The number of points in each dimension to sample. Default is 100.
        fig : matplotlib.figure.Figure, optional
            A Matplotlib Figure object to add the plot to. If None, a new figure is created.
        ax : matplotlib.axes.Axes, optional
            A Matplotlib Axes object to add the plot to. If None, a new axis is created.
        viewing_axes : str, optional
            A string indicating which two axes to project onto. Default is 'xy'.
        **kwargs:
            Additional keyword arguments to pass to `matplotlib.pyplot.imshow`.

        Returns
        -------
        tuple
            A tuple containing the image array, Matplotlib figure, and axis objects.

        Raises
        ------
        ValueError
            If `viewing_axes` does not contain exactly two distinct axes.

        Examples
        --------
        .. code-block:: python

            from cluster_generator.profiles.base import RadialProfile

            profile = RadialProfile()
            fig, ax = profile.plot_2d(extent=10)
            plt.show()

        Notes
        -----
        This method is intended for profiles with radial symmetry, where the
        projection can be effectively visualized in 2D.

        See Also
        --------
        :py:func:`matplotlib.pyplot.imshow` : The Matplotlib function used for displaying 2D image data.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Validate the viewing_axes argument
        if len(viewing_axes) != 2 or not set(viewing_axes).issubset({"x", "y", "z"}):
            raise ValueError(
                f"Invalid viewing axes: {viewing_axes}. Must be a combination of 'x', 'y', 'z'."
            )

        if fig is None:
            fig = plt.figure(figsize=(8, 6))
        if ax is None:
            ax = fig.add_subplot(111)

        # Generate mesh grid for the specified viewing axes
        x1 = np.linspace(-extent, extent, num_points)
        x2 = np.linspace(-extent, extent, num_points)
        X1, X2 = np.meshgrid(x1, x2)

        # Map viewing axes to Cartesian coordinates
        axis_map = {"x": 0, "y": 1, "z": 2}
        axis1 = axis_map[viewing_axes[0]]
        axis2 = axis_map[viewing_axes[1]]

        coords = np.zeros((3, X1.size))
        coords[axis1] = X1.ravel()
        coords[axis2] = X2.ravel()

        # Convert the Cartesian coordinates to radial coordinates
        coordinates: NDArray[float] = self.geometry_handler.from_cartesian(*coords)[0]
        # Compute the profile values at the radial coordinates
        profile_values: NDArray[float] = self(coordinates)
        image = profile_values.reshape(X1.shape)

        # Optionally apply log scaling to the image
        if kwargs.pop("log_image", True):
            image = np.log10(image)

        # Display the image
        im = ax.imshow(
            image, extent=(-extent, extent, -extent, extent), origin="lower", **kwargs
        )
        fig.colorbar(im, ax=ax, label="Profile Value")

        # Set axis labels
        ax.set_xlabel(rf"${viewing_axes[0]}_{{\rm proj}} \ \left[{{\rm kpc}}\right]$")
        ax.set_ylabel(rf"${viewing_axes[1]}_{{\rm proj}} \ \left[{{\rm kpc}}\right]$")

        return image, fig, ax
