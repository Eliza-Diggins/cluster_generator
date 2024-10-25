from typing import TYPE_CHECKING, Union, Type, Optional
from cluster_generator.grids.indexers import FieldIndex

if TYPE_CHECKING:
    from cluster_generator.profiles._abc import Profile
    from cluster_generator.models.abc import ClusterModel
    from cluster_generator.profiles.density import RadialDensityProfile
    from cluster_generator.profiles.mass import RadialMassProfile
    from cluster_generator.profiles.temperature import (
        RadialEntropyProfile,
        RadialTemperatureProfile,
    )

# Defining specific profile type hints
DensityProfile = Union["RadialDensityProfile"]
TemperatureProfile = Union["RadialTemperatureProfile"]
EntropyProfile = Union["RadialEntropyProfile"]
MassProfile = Union["RadialMassProfile"]

class ModelError(Exception):
    """
    Base exception for errors raised within the model.

    This exception is raised to indicate that an error has occurred during the
    execution of any model-related operation. It serves as a custom error class,
    distinguishing model-specific issues from generic exceptions.

    Developers can use this exception to catch errors specific to the
    ClusterModel operations and provide more contextual error handling.
    """
    pass


class ProfileDescriptor:
    """
    Descriptor class for managing profiles in the ClusterModel.

    This descriptor facilitates lazy loading and access to profile data stored in
    the grid manager. The primary responsibility of the ProfileDescriptor is to ensure
    that profiles are loaded from HDF5 storage only when accessed for the first time,
    thereby reducing unnecessary memory usage and improving performance.

    If a profile is not found in the `_profiles` dictionary (which acts as an in-memory cache),
    it attempts to load the profile from the corresponding HDF5 file. If the profile cannot be
    loaded, `None` is returned.

    Attributes
    ----------
    profile_name : str
        The name of the profile, such as 'density' or 'temperature'.
        This attribute is used to identify and locate the profile within the model's HDF5 file.
    group_name : str
        The uppercase version of `profile_name`. This is primarily used when interacting
        with HDF5, where profile names are usually stored in uppercase groups.

    Notes
    -----
    - This descriptor should only be used within `ClusterModel` or similar classes
      that manage physical profiles related to cluster modeling.
    - The lazy-loading nature ensures that profiles are only loaded from disk when
      requested, optimizing memory usage for larger models.

    Raises
    ------
    NotImplementedError
        Raised when attempting to directly assign a value to the profile descriptor.
        Profiles are read-only attributes and should not be modified directly via the descriptor.
    """

    def __init__(self, profile_name: str):
        """
        Initialize the ProfileDescriptor.

        Parameters
        ----------
        profile_name : str
            The name of the profile being managed by this descriptor.
        """
        self.profile_name = profile_name
        self.group_name = self.profile_name.upper()

    def __get__(self, instance: Optional['ClusterModel'], owner: Type['ClusterModel']) -> Union['Profile', 'ProfileDescriptor', None]:
        """
        Retrieve the profile from the ClusterModel instance.

        If the profile has not been loaded yet, it will attempt to load it from the HDF5
        file. If the profile is already loaded and cached in `_profiles`, it will return
        that cached version. If the profile cannot be found, `None` is returned.

        This method also supports being called without an instance, returning the descriptor
        itself, which can be useful for introspection or further manipulation.

        Parameters
        ----------
        instance : ClusterModel or None
            The instance of the `ClusterModel` where the profile is being accessed.
            If no instance is provided (i.e., the descriptor is accessed at the class level),
            the descriptor itself is returned.
        owner : Type[ClusterModel]
            The class type that owns this descriptor.

        Returns
        -------
        Profile or ProfileDescriptor or None
            The requested profile if it exists and is successfully loaded, or `None` if it
            is not available. If accessed without an instance, the descriptor is returned.
        """
        if instance is None:
            return self

        return instance.get_profile(self.profile_name)

    def __set__(self, instance: 'ClusterModel', value: 'Profile') -> None:
        """
        Prevent direct modification of the profile.

        Profile descriptors are read-only attributes within the `ClusterModel`. Therefore,
        any attempt to set or modify the profile directly will raise a `NotImplementedError`.

        Parameters
        ----------
        instance : ClusterModel
            The instance of `ClusterModel` where the profile would have been set.
        value : Profile
            The profile object (attempted to be set, but this operation is not allowed).

        Raises
        ------
        NotImplementedError
            Raised whenever there is an attempt to set a profile directly via the descriptor.
        """
        raise NotImplementedError(
            "ClusterModel profiles are read-only and cannot be set directly. "
            "If you need to update the profile, modify it at the HDF5 level."
        )


class ModelFieldDescriptor:
    """
    Descriptor for accessing fields in the grid manager's `Fields` and managing associated attributes
    such as `units` and `dtype`.

    This descriptor ensures that each field in the model is properly associated with its metadata
    (e.g., units and data type). If a field does not exist in the grid manager, a sentinel object
    (`UndefinedFieldSentinel`) is returned, which provides units and dtype information even for
    undefined fields. This prevents errors caused by missing fields and ensures that metadata is
    always accessible.

    Attributes
    ----------
    units : Optional[str]
        The units associated with the field (e.g., 'Msun/kpc**3').
    dtype : str
        The data type of the field (e.g., 'f8'). Defaults to 'f8'.

    Notes
    -----
    - This descriptor is primarily used within the `ClusterModel` class to provide structured
      access to fields stored in the grid manager.
    - Fields are read-only and cannot be modified directly via the descriptor. To modify
      a field, operations must be performed at the grid manager or HDF5 file level.

    Raises
    ------
    AttributeError
        Raised when there is an attempt to directly modify a field via the descriptor.
    """

    def __init__(self, units: Optional[str] = None, dtype: str = 'f8'):
        """
        Initialize the ModelFieldDescriptor.

        Parameters
        ----------
        units : Optional[str]
            The units associated with the field. Defaults to None.
        dtype : str
            The data type of the field. Defaults to 'f8' (64-bit floating point).
        """
        self.units = units
        self.dtype = dtype
        self.field_name: Optional[str] = None  # This will be set later by __set_name__

    def __set_name__(self, owner: Type, name: str) -> None:
        """
        Set the field name from the attribute name in the owning class.

        This method is automatically called when the descriptor is assigned to an attribute
        within a class. It assigns the attribute name to the `field_name` attribute, allowing
        the descriptor to reference the correct field within the grid manager.

        Parameters
        ----------
        owner : Type
            The class where the descriptor is defined (e.g., `ClusterModel`).
        name : str
            The name of the attribute being assigned the descriptor.
        """
        self.field_name = name

    def __get__(self, instance: Optional['ClusterModel'], owner: Type['ClusterModel']) -> Union['UndefinedFieldSentinel', None, 'ModelFieldDescriptor']:
        """
        Retrieve the field value from the `ClusterModel` instance.

        If the field does not exist in the grid manager, the `UndefinedFieldSentinel` is returned,
        which provides useful metadata such as units and dtype. If the field does exist, its value
        is returned directly from the grid manager.

        Parameters
        ----------
        instance : ClusterModel or None
            The instance of the `ClusterModel` class where the field is being accessed.
            If no instance is provided (i.e., the descriptor is accessed at the class level),
            the descriptor itself is returned.
        owner : Type[ClusterModel]
            The class type that owns this descriptor.

        Returns
        -------
        FieldIndex, UndefinedFieldSentinel, or None
            The requested field data if it exists, otherwise the sentinel object.
            If accessed without an instance, the descriptor is returned.
        """
        if instance is None:
            return self

        return instance.fields.get(self.field_name, )

    def __set__(self, instance: 'ClusterModel', value: 'FieldIndex') -> None:
        """
        Prevent direct modification of the field.

        Fields in the model are read-only and cannot be modified directly. Therefore, any
        attempt to set or modify the field via the descriptor will raise an `AttributeError`.

        Parameters
        ----------
        instance : ClusterModel
            The instance of `ClusterModel` where the field would have been set.
        value : FieldIndex
            The field value attempted to be set, but this operation is not allowed.

        Raises
        ------
        AttributeError
            Raised whenever there is an attempt to modify the field directly via the descriptor.
        """
        raise AttributeError(f"Cannot set the field '{self.field_name}' directly.")
