from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Union

from cluster_generator.geometry._abc import GeometryHandler
from cluster_generator.geometry.radial import SphericalGeometryHandler
from cluster_generator.grids.managers import GridManager
from cluster_generator.models._types import ProfileDescriptor
from cluster_generator.profiles._abc import Profile

if TYPE_CHECKING:
    pass


class ClusterModel(ABC):
    # Default factories
    DEFAULT_GEOMETRY = lambda: SphericalGeometryHandler()

    # Profile descriptors allow access to profiles
    # that are stored in the grid manager. Not all will
    # be present in a single model.
    temperature_profile = ProfileDescriptor("temperature")
    entropy_profile = ProfileDescriptor("entropy")
    density_profile = ProfileDescriptor("density")
    stellar_density_profile = ProfileDescriptor("stellar_density")

    def __init__(self, path: Union[Path, str]):
        # Construct the path and validate its existence.
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Failed to find model file {self.path}")

        # Load the grid manager associated with the file to form the
        # backing of the model
        self._load_grid_manager()

        # Load additional metadata that is not incorporated
        # in the standard grid manager.
        self._load_geometry()

        # Setup lazy-loading attributes.
        self._profiles = {}

    def _load_grid_manager(self):
        # Try to load the grid manager from the specified file
        # but catch any errors so that users have a clear idea what
        # went wrong.
        self.grid_manager = GridManager(self.path)

    def _load_geometry(self):
        """Load the geometry information from the GridManager."""
        self._geometry = GeometryHandler.from_hdf5(
            hdf5_file=self.grid_manager._handle.handle, group_path="GEOMETRY"
        )

    def __str__(self):
        return f"<ClusterModel: {str(self.path)}>"

    def __repr__(self):
        return f"<ClusterModel: {str(self.path)}, G={str(self._geometry)}>"

    @property
    def geometry(self):
        return self._geometry

    @property
    def fields(self):
        return self.grid_manager.Fields

    @property
    def levels(self):
        return self.grid_manager.Levels

    @property
    def bbox(self):
        return self.grid_manager.BBOX

    @property
    def block_size(self):
        return self.grid_manager.BLOCK_SIZE

    @property
    def profiles(self):
        return self._profiles

    def get_profile(self, profile_name: str):
        # Check if we already have it loaded. If not, we need
        # to seek it in the file and raise an error if
        # we cannot find it.
        if profile_name in self._profiles:
            return self._profiles[profile_name]
        else:
            try:
                profile = Profile.from_hdf5(
                    hdf5_file=self.grid_manager._handle.handle,
                    group_path=f"PROFILES/{profile_name}",
                    geometry=self.geometry,
                )
            except KeyError:
                # This means we just don't have this profile!
                profile = None
            except Exception as e:
                raise IOError(
                    f"Failed to load profile {profile_name} for unknown reasons: {e}."
                )

            # Store the profile in _profiles, even if it's None
            self._profiles[profile_name] = profile
            return profile

    def has_profile(self, profile_name: str):
        # We need to seek out the profile in both the HDF5 and
        # our _profiles dict. In some cases they may be in either / or
        return (profile_name in self._profiles) or (
            profile_name in self.grid_manager._handle["PROFILES"]
        )

    def add_profile(
        self,
        profile_name: str,
        profile: "Profile",
        create_field: bool = True,
        in_hdf5: bool = True,
        overwrite: bool = False,
        **kwargs,
    ):
        # Perform basic validation tasks. Ensure that the overwrite is managed and
        # figure out if we already have this profile loaded.
        hdf5_name = profile_name
        # Ensure that the profile shares our geometry.
        if profile.geometry_handler != self.geometry:
            raise ValueError(
                f"Cannot add {profile_name} to {self}. Profile had geometry {profile.geometry_handler} and Model"
                f" has geometry {self.geometry}. Profiles must have matching geometry to be added to a model."
            )

        if overwrite and self.has_profile(profile_name):
            # We need to remove in anticipation of adding anew.
            del self.grid_manager._handle["PROFILES"][hdf5_name]
        elif (not overwrite) and self.has_profile(profile_name):
            raise ValueError(
                f"Cannot add profile {profile_name}. This profile already exists and overwrite=False."
                f" If you intend to overwrite the profile, use overwrite=True."
            )
        else:
            pass

        # Add the profile.
        if in_hdf5:
            # This will automatically register the profile.
            profile.to_hdf5(
                self.grid_manager._handle.handle, group_path=f"PROFILES/{hdf5_name}"
            )
        else:
            pass

        self._profiles[profile_name] = profile

        # Manage field creation. In most cases, we want to dump the profile
        # to disk, which can be automatically done at this stage.
        if create_field:
            self.grid_manager.add_field_from_profile(
                profile, hdf5_name, overwrite=overwrite, **kwargs
            )

    def remove_profile(self, profile_name: str, remove_field: bool = True) -> None:
        """
        Remove a profile from the model.

        This method removes a profile from the model, ensuring that the profile is deleted from the in-memory
        `_profiles` dictionary and optionally from the HDF5 file and the associated field.

        Parameters
        ----------
        profile_name : str
            The name of the profile to remove.
        remove_field : bool, optional
            Whether to remove the associated field from the grid manager as well. Default is True.

        Raises
        ------
        KeyError
            If the profile does not exist in the model.
        ValueError
            If trying to remove a field that doesn't exist or an error occurs during removal.
        """
        # Validate the profile details.
        if (
            profile_name not in self._profiles
            and profile_name not in self.grid_manager._handle.get("PROFILES", {})
        ):
            raise KeyError(
                f"Profile '{profile_name}' does not exist in the model or HDF5 file."
            )

        # Remove the profile from our dictionary (if it is even there; it may not be due to lazy loading),
        # then remove the profile from the hdf5 file.
        if profile_name in self._profiles:
            del self._profiles[profile_name]

        if (
            "PROFILES" in self.grid_manager._handle
            and profile_name in self.grid_manager._handle["PROFILES"]
        ):
            try:
                del self.grid_manager._handle["PROFILES"][profile_name]
            except Exception as e:
                raise ValueError(
                    f"Failed to remove profile '{profile_name}' from HDF5: {e}"
                )

        # Optionally, we can also remove the universal field associated with the profile if
        # it exists.
        if remove_field:
            if profile_name in self.grid_manager.Fields:
                try:
                    self.grid_manager.remove_universal_field(
                        profile_name, unregister=True
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to remove field '{profile_name}' associated with the profile: {e}"
                    )

    def add_field_from_profile(self, profile: "Profile"):
        pass


# class ClusterModel(ABC):

#    # Default factories for geometry and equation of state (EOS)
#    DEFAULT_GEOMETRY = lambda: SphericalGeometryHandler()
#    DEFAULT_EOS = lambda: None

#    # Profile descriptors all us to store the actual profiles
#    # which allows for computation speed ups in the long run.
#    temperature_profile = ProfileDescriptor()
#    density_profile = ProfileDescriptor()
#    total_density_profile = ProfileDescriptor()
#    entropy_profile = ProfileDescriptor()


#    _field_meta_cache = {}

#    def __init__(self, filename: Union[str, Path]):
#        # Configure the filename and validate its existence.
#        self.filename = Path(filename)
#        if not self.filename.exists():
#            raise FileNotFoundError(f"File {self.filename} does not exist.")

#        mylog.info("Loading %s: %s...", self.__class__.__name__, self.filename)

#        # Load the grid manager and geometry
#        self._load_grid_manager()

#        # Load the profile and field dictionaries.
#        self._profiles = {}
#        self._fields = {}

#        # Load model properties (geometry, EOS)
#        self._load_geometry()
#        self._load_eos()


#    def _load_grid_manager(self):
#        """Load the GridManager from the specified file."""
#        try:
#            self.grid_manager = GridManager(self.filename)
#        except Exception as e:
#            raise ValueError(f"Failed to load GridManager from {self.filename}: {e}")

#    def _load_geometry(self):
#        """Load the geometry information from the GridManager."""
#        try:
#            self._geometry = GeometryHandler.from_hdf5(
#                hdf5_file=self.grid_manager._handle.handle,
#                group_path='GEOMETRY'
#            )
#        except Exception as e:
#            raise ModelError(f"Failed to load geometry from {self.filename}: {e}")

#    def _load_eos(self):
#        """Load the equation of state (EOS) from the model file if available."""
#        # Placeholder for EOS loading logic, if needed in the future.
#        pass

#    @classmethod
#    def _validate_and_generate_grid(cls, filename: Union[str, Path], geometry: 'GeometryHandler',
#                                    bbox: Any, block_size: Any, overwrite: bool = False) -> GridManager:
#        """
#        Validate input parameters and generate a GridManager instance.

#        Parameters
#        ----------
#        filename : str or Path
#            The path to the HDF5 file where the grid data will be saved or loaded.
#        geometry : GeometryHandler
#            Geometry handler defining symmetry and transformations.
#        bbox : Any
#            Bounding box, coerced into a valid BoundingBox.
#        block_size : Any
#            Block size, coerced into a valid DomainShape.
#        overwrite : bool, optional
#            Whether to overwrite an existing file (default is False).

#        Returns
#        -------
#        GridManager
#            An initialized GridManager object for the cluster model.

#        Raises
#        ------
#        ValueError
#            If parameters are invalid or incompatible.
#        """
#        filename = Path(filename)
#        bbox = coerce_to_bounding_box(bbox)
#        block_size = coerce_to_domain_shape(block_size)

#        # Validate that the dimensions align correctly
#        geometry_expected_dimensions = 3 - geometry.SYMMETRY_TYPE.dimension_reduction
#        if not (geometry_expected_dimensions == bbox.shape[1] == len(block_size)):
#            raise ValueError(f"Failed to generate ClusterModel. Geometry, bbox, and block_size dimensions do not match.")

#        # If file exists and overwrite is not allowed, raise an error
#        if filename.exists() and not overwrite:
#            raise ValueError(f"File {filename} exists, and overwrite=False. Set overwrite=True to proceed.")

#        # Otherwise, create a new GridManager
#        try:
#            mylog.debug("Creating a new GridManager at: %s", filename)
#            return GridManager.create(
#                path=filename,
#                BBOX=bbox,
#                BLOCK_SIZE=block_size,
#                AXES=geometry.SYMMETRY_TYPE.grid_axis_orders,
#                geometry=geometry,
#                overwrite=overwrite
#            )
#        except Exception as e:
#            raise ValueError(f"Failed to create GridManager at {filename}: {e}")

#    @classmethod
#    def _validate_input_profiles(cls, **profiles: 'Profile') -> 'GeometryHandler':
#        """
#        Validates that all provided profiles share the same geometry handler.

#        Parameters
#        ----------
#        profiles : Profile
#            Variable number of profiles to validate.

#        Returns
#        -------
#        GeometryHandler
#            The common geometry handler shared by all profiles.

#        Raises
#        ------
#        ValueError
#            If the profiles do not share the same geometry handler.
#        """
#        geometry_handlers = {profile.geometry_handler for profile in profiles.values()}
#        if len(geometry_handlers) != 1:
#            raise ValueError(
#                f"Expected all profiles to share the same geometry handler, but found {len(geometry_handlers)} different handlers: {geometry_handlers}. "
#                "All profiles must have the same geometry handler for model generation."
#            )
#        return geometry_handlers.pop()

#    @classmethod
#    def generate_cluster_model_from_profiles(
#            cls,
#            filename: Union[str, Path],
#            profiles: dict[str, 'Profile'],
#            bbox: Any,
#            block_size: Any,
#            overwrite: bool = False) -> 'ClusterModel':
#        """
#        Generate a ClusterModel from the given profiles.

#        Parameters
#        ----------
#        filename : str or Path
#            Path to the HDF5 file where the grid data will be saved or loaded.
#        profiles : dict[str, Profile]
#            Dictionary of profiles (e.g., density, temperature) to add to the model.
#        bbox : Any
#            Bounding box of the grid domain.
#        block_size : Any
#            Block size for grid discretization.
#        overwrite : bool, optional
#            Whether to overwrite an existing file (default is False).

#        Returns
#        -------
#        ClusterModel
#            A fully initialized ClusterModel object.

#        Raises
#        ------
#        ValueError
#            If input validation fails or grid manager creation fails.
#        """
#        mylog.info("Generating %s from profiles: %s",cls.__name__,list(profiles.keys()))
#        geometry = cls._validate_input_profiles(**profiles)

#        # Step 2: Validate and generate grid manager
#        grid_manager = cls._validate_and_generate_grid(
#            filename=filename,
#            geometry=geometry,
#            bbox=bbox,
#            block_size=block_size,
#            overwrite=overwrite
#        )

#        # Step 3: Write geometry and profiles to the HDF5 file
#        geometry.to_hdf5(grid_manager._handle.handle, "GEOMETRY")

#        for profile_name, profile in profiles.items():
#            mylog.info("Adding %s profile to model grid manager...", profile_name)

#            # Search for a corresponding field to determine units and name.
#            if hasattr(cls, profile_name):
#                units = getattr(cls,profile_name).units
#                hdf5_name = getattr(cls,profile_name).hdf5_name
#            else:
#                units = ""
#                hdf5_name = profile_name.upper()

#            profile.to_hdf5(grid_manager._handle.handle, f"PROFILES/{hdf5_name}")
#            try:
#                grid_manager.add_field_from_function(
#                    profile,
#                    field_name=hdf5_name,
#                    units=str(units),
#                    geometry=geometry
#                )
#            except Exception as e:
#                raise ModelError(f"Failed to add profile {hdf5_name} to coarse grid: {e}")

#        # Step 4: Return the model instance
#        return cls(filename)
