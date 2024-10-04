from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Union

from cluster_generator.geometry._abc import GeometryHandler
from cluster_generator.geometry.radial import SphericalGeometryHandler
from cluster_generator.grids.managers import GridManager
from cluster_generator.grids.utils import coerce_to_bounding_box, coerce_to_domain_shape
from cluster_generator.models._types import ModelFieldDescriptor, ProfileDescriptor
from cluster_generator.profiles._abc import Profile
from cluster_generator.utils import mylog

if TYPE_CHECKING:
    from cluster_generator.grids._types import BoundingBox, DomainShape


class ClusterModel(ABC):
    """
    ClusterModel represents a physical model of a galaxy cluster, providing access to
    profiles and fields stored in the grid manager. It also manages solvers and geometry.
    """

    DEFAULT_GEOMETRY = lambda: SphericalGeometryHandler()

    # Profile descriptors allow quick access to specific profiles stored in the grid manager
    temperature_profile = ProfileDescriptor("temperature")
    entropy_profile = ProfileDescriptor("entropy")
    density_profile = ProfileDescriptor("density")
    total_density_profile = ProfileDescriptor("total_density")
    stellar_density_profile = ProfileDescriptor("stellar_density")

    # Field Access
    # These descriptors provide both access to the FieldIndex objects of the grid manager,
    # but also handle the standard units and dtypes for different fields.
    # If a field is not in the list of available descriptors, it is still available through
    # self.fields.
    density = ModelFieldDescriptor(units="Msun/kpc**3")
    total_density = ModelFieldDescriptor(units="Msun/kpc**3")
    stellar_density = ModelFieldDescriptor(units="Msun/kpc**3")
    temperature = ModelFieldDescriptor(units="keV")
    potential = ModelFieldDescriptor(units="km**2/s**2")
    gravitational_field = ModelFieldDescriptor(units="km/s**2")
    pressure = ModelFieldDescriptor(units="Msun*kpc**3/Myr")

    def __init__(self, path: Union[Path, str], grid_manager=None):
        """
        Initialize the ClusterModel with a specified path to the model file.

        Parameters
        ----------
        path : Union[Path, str]
            Path to the model file.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Failed to find model file {self.path}")

        # Load the grid manager and geometry information
        if grid_manager is None:
            self._load_grid_manager()
        else:
            self.grid_manager = grid_manager

        self._load_geometry()

        # Initialize lazy-loaded profiles and solvers
        self._profiles = {}

    def _load_grid_manager(self):
        """Load the grid manager from the model file."""
        self.grid_manager = GridManager(self.path)

    def _load_geometry(self):
        """Load the geometry information from the GridManager."""
        self._geometry = GeometryHandler.from_hdf5(
            hdf5_file=self.grid_manager._handle.handle, group_path="GEOMETRY"
        )

    def __str__(self):
        return f"<ClusterModel: {str(self.path)}>"

    def __repr__(self):
        return f"<ClusterModel: {str(self.path)}, Geometry={str(self._geometry)}>"

    # --- Properties ---
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

    # --- Profile Management ---
    def get_profile(self, profile_name: str) -> Union["Profile", None]:
        """
        Retrieve a profile by its name, loading it from HDF5 if necessary.

        Parameters
        ----------
        profile_name : str
            The name of the profile to retrieve.

        Returns
        -------
        Profile or None
            The profile if found, or None if the profile is not available.
        """
        if profile_name in self._profiles:
            return self._profiles[profile_name]

        try:
            profile = Profile.from_hdf5(
                hdf5_file=self.grid_manager._handle.handle,
                group_path=f"PROFILES/{profile_name.upper()}",
                geometry=self.geometry,
            )
            self._profiles[profile_name] = profile
        except KeyError:
            profile = None
        except Exception as e:
            raise IOError(f"Failed to load profile '{profile_name}': {e}")

        return profile

    def has_profile(self, profile_name: str) -> bool:
        """
        Check if the model contains a specific profile.

        Parameters
        ----------
        profile_name : str
            Name of the profile to check.

        Returns
        -------
        bool
            True if the profile is found, False otherwise.
        """
        return (profile_name in self._profiles) or (
            profile_name.upper() in self.grid_manager._handle["PROFILES"].keys()
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
        """
        Add a profile to the model, optionally creating a field and saving it to HDF5.

        Parameters
        ----------
        profile_name : str
            Name of the profile to add.
        profile : Profile
            The profile object to add.
        create_field : bool, optional
            Whether to create a field from the profile. Default is True.
        in_hdf5 : bool, optional
            Whether to store the profile in the HDF5 file. Default is True.
        overwrite : bool, optional
            Whether to overwrite the profile if it already exists. Default is False.
        """
        if profile.geometry_handler != self.geometry:
            raise ValueError(
                f"Profile geometry '{profile.geometry_handler}' does not match the model geometry '{self.geometry}'"
            )

        if overwrite and self.has_profile(profile_name):
            del self.grid_manager._handle["PROFILES"][profile_name]
        elif not overwrite and self.has_profile(profile_name):
            raise ValueError(
                f"Profile '{profile_name}' already exists. Use overwrite=True to replace it."
            )

        if in_hdf5:
            profile.to_hdf5(
                self.grid_manager._handle.handle, group_path=f"PROFILES/{profile_name}"
            )

        self._profiles[profile_name] = profile

        if create_field:
            self.grid_manager.add_field_from_profile(
                profile, profile_name, overwrite=overwrite, **kwargs
            )

    def remove_profile(self, profile_name: str, remove_field: bool = True) -> None:
        """
        Remove a profile from the model.

        Parameters
        ----------
        profile_name : str
            Name of the profile to remove.
        remove_field : bool, optional
            Whether to remove the associated field from the grid manager as well. Default is True.
        """
        if not self.has_profile(profile_name):
            raise KeyError(f"Profile '{profile_name}' does not exist.")

        # Remove from in-memory profiles
        if profile_name in self._profiles:
            del self._profiles[profile_name]

        # Remove from HDF5
        if profile_name in self.grid_manager._handle.get("PROFILES", {}):
            del self.grid_manager._handle["PROFILES"][profile_name]

        # Optionally remove the field associated with the profile
        if remove_field and profile_name in self.grid_manager.Fields:
            self.grid_manager.remove_universal_field(profile_name, unregister=True)

    def add_field_from_profile(self, field_name: str, profile: "Profile", **kwargs):
        """Add a field from the provided profile to the grid manager."""
        self.grid_manager.add_field_from_profile(profile, field_name, **kwargs)

    @classmethod
    def _validate_and_generate_grid(
        cls,
        filename: Union[str, Path],
        geometry: "GeometryHandler",
        bbox: "BoundingBox",
        block_size: "DomainShape",
        overwrite: bool = False,
    ) -> GridManager:
        """
        Validate input parameters and generate a GridManager instance.

        Parameters
        ----------
        filename : str or Path
            The path to the HDF5 file where the grid data will be saved or loaded.
        geometry : GeometryHandler
            Geometry handler defining symmetry and transformations.
        bbox : Any
            Bounding box, coerced into a valid BoundingBox.
        block_size : Any
            Block size, coerced into a valid 'DomainShape'.
        overwrite : bool, optional
            Whether to overwrite an existing file (default is False).

        Returns
        -------
        GridManager
            An initialized GridManager object for the cluster model.

        Raises
        ------
        ValueError
            If parameters are invalid or incompatible.
        """
        filename = Path(filename)
        bbox = coerce_to_bounding_box(bbox)
        block_size = coerce_to_domain_shape(block_size)

        # Validate that the dimensions align correctly
        geometry_expected_dimensions = 3 - geometry.SYMMETRY_TYPE.dimension_reduction
        if not (geometry_expected_dimensions == bbox.shape[1] == len(block_size)):
            raise ValueError(
                "Failed to generate ClusterModel. Geometry, bbox, and block_size dimensions do not match."
            )

        # If file exists and overwrite is not allowed, raise an error
        if filename.exists() and not overwrite:
            raise ValueError(
                f"File {filename} exists, and overwrite=False. Set overwrite=True to proceed."
            )

        # Otherwise, create a new GridManager
        try:
            mylog.debug("\tCreating a new GridManager at: %s...", filename)
            return GridManager.create(
                path=filename,
                BBOX=bbox,
                BLOCK_SIZE=block_size,
                AXES=geometry.SYMMETRY_TYPE.grid_axis_orders,
                geometry=geometry,
                overwrite=overwrite,
            )
        except Exception as e:
            raise ValueError(f"Failed to create GridManager at {filename}: {e}")

    @classmethod
    def _validate_input_profiles(cls, **profiles: "Profile") -> "GeometryHandler":
        """
        Validates that all provided profiles share the same geometry handler.

        Parameters
        ----------
        profiles : Profile
            Variable number of profiles to validate.

        Returns
        -------
        GeometryHandler
            The common geometry handler shared by all profiles.

        Raises
        ------
        ValueError
            If the profiles do not share the same geometry handler.
        """
        geometry_handlers = {profile.geometry_handler for profile in profiles.values()}
        if len(geometry_handlers) != 1:
            raise ValueError(
                f"Expected all profiles to share the same geometry handler, but found {len(geometry_handlers)} different handlers: {geometry_handlers}. "
                "All profiles must have the same geometry handler for model generation."
            )
        return geometry_handlers.pop()

    @classmethod
    def generate_cluster_model_from_profiles(
        cls,
        filename: Union[str, Path],
        profiles: dict[str, "Profile"],
        bbox: "BoundingBox",
        block_size: "DomainShape",
        overwrite: bool = False,
    ) -> "ClusterModel":
        """
        Generate a ClusterModel from the given profiles.

        Parameters
        ----------
        filename : str or Path
            Path to the HDF5 file where the grid data will be saved or loaded.
        profiles : dict[str, Profile]
            Dictionary of profiles (e.g., density, temperature) to add to the model.
        bbox : Any
            Bounding box of the grid domain.
        block_size : Any
            Block size for grid discretization.
        overwrite : bool, optional
            Whether to overwrite an existing file (default is False).

        Returns
        -------
        ClusterModel
            A fully initialized ClusterModel object.

        Raises
        ------
        ValueError
            If input validation fails or grid manager creation fails.
        """
        mylog.info(
            "Instantiating %s from profiles: %s...", cls.__name__, list(profiles.keys())
        )

        # Perform validation steps. First we ensure the geometry is
        # self consistent and then produce the grid manager.
        geometry = cls._validate_input_profiles(**profiles)
        grid_manager = cls._validate_and_generate_grid(
            filename=filename,
            geometry=geometry,
            bbox=bbox,
            block_size=block_size,
            overwrite=overwrite,
        )

        # Add the geometry to the HDF5 file. This is required for
        # a valid Model HDF5 file.
        geometry.to_hdf5(grid_manager._handle.handle, "GEOMETRY")

        # For each of the profiles, we add the profile to
        # disk, add a field corresponding to that profile
        # and add an injection solver for the field.
        for field_name, profile in profiles.items():
            mylog.info("\tAdding %s profile to model...", field_name)
            profile.to_hdf5(
                grid_manager._handle.handle, f"PROFILES/{field_name.upper()}"
            )

            # Fetch the field descriptor from the class. This allows
            # for lookup of standard dtype and units. If we don't have it,
            # we assume no units and f8.
            if hasattr(cls, field_name):
                units, dtype = (
                    getattr(cls, field_name).units,
                    getattr(cls, field_name).dtype,
                )
            else:
                units, dtype = "", "f8"

            # Add the injection solver to our solver array so that
            # refinement can take advantage of the available profile.
            # SKIPPED.

            # Now add the field to the file.
            try:
                grid_manager.add_field_from_profile(
                    profile, field_name, units=units, dtype=dtype
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to add field {field_name} to coarse grid: {e}"
                )

        # Step 4: Return the model instance
        grid_manager.commit_changes()
        return cls(filename, grid_manager=grid_manager)

    @classmethod
    def from_dens_and_temp(
        cls,
        filename: Union[str, Path],
        density: Profile,
        temperature: Profile,
        bbox: "BoundingBox",
        block_size: "DomainShape",
        overwrite: bool = False,
    ):
        pass

    @classmethod
    def from_dens_and_tden(
        cls,
        filename: Union[str, Path],
        density: Profile,
        total_density: Profile,
        bbox: "BoundingBox",
        block_size: "DomainShape",
        overwrite: bool = False,
    ):
        pass

    def no_gas(self):
        pass


if __name__ == "__main__":
    from cluster_generator.profiles.density import NFWDensityProfile

    GridManager.logger.setLevel("CRITICAL")
    p = NFWDensityProfile()
    q = NFWDensityProfile()

    g = ClusterModel.generate_cluster_model_from_profiles(
        "test.hdf5",
        {"density": p, "totl_density": q},
        [0, 5000],
        [1000],
        overwrite=True,
    )

    print(g.get_profile("totl_density"))

    print(g.profiles)
