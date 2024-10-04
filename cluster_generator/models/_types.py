from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from cluster_generator.profiles._abc import Profile
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


# Exceptions
class ModelError(Exception):
    pass


# ============================== #
# Descriptors and meta classes   #
# ============================== #
class ProfileDescriptor:
    """
    Descriptor class for managing profiles in the ClusterModel.

    This class handles the loading and accessing of profiles. If a profile is not
    found in the `_profiles` dictionary, it attempts to load it from the file.
    If loading fails, the profile is set to None.
    """

    def __init__(self, profile_name: str):
        self.profile_name = profile_name
        self.group_name = self.profile_name.upper()

    def __get__(self, instance, owner) -> "Profile|ProfileDescriptor|None":
        """
        Get the profile from the instance.

        If the profile is not loaded, it attempts to load it from the file.
        If loading fails, it sets the profile to None.

        Parameters
        ----------
        instance : ClusterModel
            The instance of the ClusterModel.
        owner : type
            The owner class (ClusterModel).

        Returns
        -------
        Profile or None
            The profile if loaded successfully, otherwise None.
        """
        # If we aren't passed an instance at all, we want to return this descriptor
        # for access to its internal methods.
        if instance is None:
            return self

        # Otherwise, we determine if the profile is already loaded / if it's present
        # at all and proceed with the lazy-loading procedure.
        return instance.get_profile(self.profile_name)

    def __set__(self, instance, value):
        """
        Set the profile value in the instance's _profiles dictionary.

        Parameters
        ----------
        instance : ClusterModel
            The instance of the ClusterModel.
        value : Profile
            The profile to set.
        """
        raise NotImplementedError(
            "ClusterModel profiles cannot be set. If necessary, they may be changed at the"
            "HDF5 level; however, this should be done cautiously as it could break certain"
            "functionality."
        )


class ModelFieldDescriptor:
    """
    Descriptor for accessing fields in `grid_manager.Fields` and their associated attributes like `units` and `dtype`.

    Parameters
    ----------
    units : str
        The units associated with the field.
    dtype : type
        The data type of the field.
    """

    def __init__(self, units=None, dtype=None):
        self.units = units
        self.dtype = dtype
        self.field_name = (
            None  # The name of the field (deduced from the attribute name)
        )
        self._attr_flag = (
            None  # Internal flag to check if we're accessing 'units' or 'dtype'
        )

    def __set_name__(self, owner, name):
        """
        Automatically deduce the field name from the attribute name.

        Parameters
        ----------
        owner : type
            The owner class where the descriptor is being defined.
        name : str
            The name of the attribute where the descriptor is applied.
        """
        self.field_name = name

    def __get__(self, instance, owner):
        """
        Access the field, or specific attributes (units or dtype).

        Parameters
        ----------
        instance : object
            The instance of the owning class (i.e., ClusterModel).
        owner : type
            The owning class type.

        Returns
        -------
        FieldIndex or str or type or None
            The field data, or the units/dtype, or None if the field is not present.
        """
        if instance is None:
            return self

        # If the attribute flag is set, return the relevant value (units/dtype)
        if self._attr_flag == "units":
            self._reset_attr_flag()  # Reset after use
            return self.units
        elif self._attr_flag == "dtype":
            self._reset_attr_flag()  # Reset after use
            return self.dtype

        # Otherwise, return the actual field from instance.fields
        if self.field_name in instance.fields:
            return instance.fields[self.field_name]
        else:
            return None

    def __getattr__(self, item):
        """
        Handle dynamic access for 'units' and 'dtype'.
        If either is requested, set the internal flag accordingly.

        Parameters
        ----------
        item : str
            The attribute being accessed.

        Returns
        -------
        Any
            The corresponding attribute (either units or dtype).
        """
        if item == "units":
            self._attr_flag = "units"
            return self
        elif item == "dtype":
            self._attr_flag = "dtype"
            return self
        else:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute '{item}'"
            )

    def __set__(self, instance, value):
        """
        Prevent setting the field directly (read-only).

        Parameters
        ----------
        instance : object
            The instance of the owning class (i.e., ClusterModel).
        value : any
            The value being set (not allowed).

        Raises
        ------
        AttributeError
            Prevents modification of the field through this descriptor.
        """
        raise AttributeError(f"Cannot set the field '{self.field_name}' directly.")

    def _reset_attr_flag(self):
        """
        Reset the internal attribute flag after use, ensuring future accesses behave normally.
        """
        self._attr_flag = None
