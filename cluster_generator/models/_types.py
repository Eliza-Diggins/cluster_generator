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
