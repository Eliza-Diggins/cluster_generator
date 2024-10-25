from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Union

from cluster_generator.geometry.abc import GeometryHandler
from cluster_generator.geometry.radial import SphericalGeometryHandler
from cluster_generator.grids.managers import GridManager
from cluster_generator.grids.utils import coerce_to_bounding_box, coerce_to_domain_shape
from cluster_generator.models._types import ProfileDescriptor
from cluster_generator.profiles._abc import Profile
from cluster_generator.utilities.logging import mylog
from cluster_generator.pipelines.abc import Pipeline

if TYPE_CHECKING:
    from cluster_generator.grids._types import BoundingBox, DomainShape


class ClusterModel(ABC):
    # Implementing default factories.
    @staticmethod
    def DEFAULT_GEOMETRY():
        """
        Return the default geometry handler (SphericalGeometryHandler).
        """
        return SphericalGeometryHandler()

    # Profile descriptors allow quick access to specific profiles stored in the grid manager.
    temperature_profile = ProfileDescriptor("temperature")
    entropy_profile = ProfileDescriptor("entropy")
    density_profile = ProfileDescriptor("density")
    total_density_profile = ProfileDescriptor("total_density")
    stellar_density_profile = ProfileDescriptor("stellar_density")

    def __init__(
            self,
            path: Union[Path, str],
            grid_manager: Union[GridManager, None] = None,
            geometry_handler: Union[GeometryHandler, None] = None,
            pipeline: Union['Pipeline', None] = None,
    ):
        """
        ClusterModel represents a physical model of a galaxy cluster, providing access to profiles
        and fields stored in the grid manager. It also manages pipelines, geometry, and pipelines.

        Parameters
        ----------
        path : Union[Path, str]
            Path to the model file, which stores the cluster's data.
        grid_manager : GridManager or None, optional
            Pre-existing `GridManager` to be used for handling the cluster's grid data. If not provided,
            the grid manager is loaded from the model file at the given path.
        geometry_handler : GeometryHandler or None, optional
            Custom geometry handler to define the geometry of the cluster model. If not provided,
            the default geometry handler (`SphericalGeometryHandler`) will be used.
        pipeline : SolverPipeline or None, optional
            An instance of a `Pipeline` used to handle solver pipelines. If not provided,
            attempts to load from file or defaults to None.

        Raises
        ------
        FileNotFoundError
            If the provided path does not exist.
        ValueError
            If an invalid path type is provided.
        RuntimeError
            If there is a failure while loading the grid manager or geometry handler.
        """
        # Validate and set the path
        self.path = self._validate_path(path)
        mylog.info("Initializing ClusterModel from path: %s", self.path)

        # Initialize attributes with provided arguments or load defaults if not provided
        self.grid_manager = grid_manager or self._load_grid_manager()
        mylog.debug("\t%-15s: %s",'Grid Manager', self.grid_manager)

        self.geometry_handler = geometry_handler or self._load_geometry()
        mylog.debug("\t%-15s: %s",'Geometry', self.geometry_handler)

        self._pipeline = self._initialize_pipeline()
        if pipeline is not None:
            self.set_pipeline(pipeline,overwrite=True)
        mylog.debug("\t%-15s: %s",'Pipeline', self._pipeline)

        self._profiles = {}  # Initialize lazy-loaded profiles
        self.Fields = self.grid_manager.Fields

    @property
    def pipeline(self):
        return self._pipeline

    def set_pipeline(self, pipeline: Pipeline, overwrite: bool = False) -> None:
        # Validate pipeline overwrite
        if self._pipeline is not None and not overwrite:
            # We don't have overwrite permission.
            raise ValueError("Pipeline already set; use overwrite=True to replace.")
        elif self._pipeline is not None and overwrite:
            self.remove_pipeline(delete_data = True,delete_pipeline = True)

        # Assign new pipeline and configure it within model context
        self._pipeline = pipeline
        self._pipeline._model = self

        # Set up HDF5 handle and call configuration
        handle = self.grid_manager._handle.handle.require_group('PIPELINE')
        self._pipeline.to_hdf5(handle, set_handle=True)

    def remove_pipeline(self, delete_data = False, delete_pipeline = False):
        if self._pipeline is None:
            raise IOError("Cannot unlink pipeline. There is no pipeline to unlink.")

        # Delete the data from disk first. If we do so, the handle gets reset to
        # ensure the pipeline knows it lost its data.
        if delete_data:
            del self._pipeline._handle
            self._pipeline._handle = None

        # Now remove the model reference in the pipeline and delete the pipeline if needed.
        self._pipeline._model = None
        if delete_pipeline:
            del self._pipeline

        self._pipeline = None

    @staticmethod
    def _validate_path(path: Union[Path, str]) -> Path:
        """Validates and returns the model path as a `Path` object."""
        if not isinstance(path, (str, Path)):
            raise ValueError(f"Invalid path type: {type(path)}. Expected str or Path.")
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Failed to find model file at {path}")
        return path

    def _load_grid_manager(self) -> GridManager:
        """
        Load and return the GridManager from the model file specified by `self.path`.

        Raises
        ------
        RuntimeError
            If the GridManager fails to load for any reason.

        Returns
        -------
        GridManager
            Loaded GridManager instance.
        """
        try:
            return GridManager(self.path)
        except Exception as e:
            raise RuntimeError(f"Failed to load GridManager from {self.path}: {e}")

    def _load_geometry(self) -> GeometryHandler:
        """
        Load and return the GeometryHandler from the GridManager.

        This method assumes that geometry is stored within the HDF5 file in the
        'GEOMETRY' group. The geometry handler must match the geometry defined
        in the GridManager.

        Raises
        ------
        IOError
            If the geometry group is missing from the HDF5 file or an error occurs while reading.
        RuntimeError
            If the geometry handler cannot be loaded.

        Returns
        -------
        GeometryHandler
            Loaded GeometryHandler instance.
        """
        try:
            if "GEOMETRY" not in self.grid_manager._handle:
                raise IOError(f"Geometry group not found in {self.path}.")

            return GeometryHandler.from_hdf5(
                hdf5_file=self.grid_manager._handle.handle, group_path="GEOMETRY"
            )
        except IOError as e:
            raise IOError(f"Failed to locate or load geometry from {self.path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load GeometryHandler: {e}")

    def _initialize_pipeline(self) -> Pipeline|None:
        """Initialize or load Pipeline from HDF5."""
        if "PIPELINE" in self.grid_manager._handle:
            # We don't need to setup the model connection, we're loading from our
            # own HDF5.
            pipeline = Pipeline.from_hdf5(self.grid_manager._handle["PIPELINE"])
            return pipeline
        else:
            return None

    def __str__(self):
        return f"<ClusterModel: {str(self.path)}>"

    def __repr__(self):
        return f"<ClusterModel: {str(self.path)}, Geometry={str(self.geometry_handler)}>"

    # --- Properties ---
    @property
    def geometry(self) -> GeometryHandler:
        """
        Get the geometry of the cluster model.

        The geometry handler defines the model's symmetry and transformation properties.

        Returns
        -------
        GeometryHandler
            The geometry handler associated with the cluster model.

        Examples
        --------
        .. code-block:: python

            model = ClusterModel("path/to/model/file")
            geometry = model.geometry
            print(geometry)
        """
        return self.geometry_handler

    @property
    def levels(self) -> dict:
        """
        Get the refinement levels from the GridManager.

        Levels refer to different grid resolutions in the Adaptive Mesh Refinement (AMR) structure.

        Returns
        -------
        dict
            A dictionary where keys are level numbers and values represent corresponding
            grid refinement levels.

        Examples
        --------
        .. code-block:: python

            model = ClusterModel("path/to/model/file")
            levels = model.levels
            print(levels)
        """
        return self.grid_manager.Levels

    @property
    def bbox(self) -> list:
        """
        Get the bounding box of the grid domain.

        The bounding box defines the spatial limits of the model's grid.

        Returns
        -------
        list
            A list representing the spatial limits (bounding box) of the grid domain.

        Examples
        --------
        .. code-block:: python

            model = ClusterModel("path/to/model/file")
            bbox = model.bbox
            print(bbox)
        """
        return self.grid_manager.BBOX

    @property
    def block_size(self) -> list:
        """
        Get the block size for each grid.

        Block size represents the number of cells in each dimension of the grid blocks.

        Returns
        -------
        list
            A list representing the block size for the grids.

        Examples
        --------
        .. code-block:: python

            model = ClusterModel("path/to/model/file")
            block_size = model.block_size
            print(block_size)
        """
        return self.grid_manager.BLOCK_SIZE

    # --- Profile Management ---
    def list_profiles(self) -> list[str]:
        """
        List all available profiles in the model.

        This method returns a list of all profiles currently available in the model,
        either loaded in memory (from `_profiles`) or stored in the HDF5 file.

        Returns
        -------
        list of str
            A list containing the names of all profiles available in the model.
            This includes profiles that have been loaded into memory and those that
            are stored on disk but not yet loaded.

        Examples
        --------
        .. code-block:: python

            model = ClusterModel("path/to/model/file")
            profiles = model.list_profiles()
            print(profiles)  # ['density', 'temperature', ...]

        Notes
        -----
        Profiles are either loaded into memory during the model's operation or stored
        in the HDF5 file on disk. This method combines both sources to give a full list
        of available profiles.
        """
        # Get profiles loaded in memory (from _profiles)
        memory_profiles = list(self._profiles.keys())

        # Get profiles stored in HDF5 (from the 'PROFILES' group)
        if "PROFILES" in self.grid_manager._handle:
            hdf5_profiles = [k.lower() for k in list(self.grid_manager._handle["PROFILES"].keys())]
        else:
            hdf5_profiles = []

        # Combine and return unique profiles from both sources
        all_profiles = set(memory_profiles + hdf5_profiles)
        return sorted(all_profiles)

    def get_profile(self, field_name: str) -> Union["Profile", None]:
        """
        Retrieve a profile by its field name, loading it from the HDF5 file if necessary.

        This method first checks if the profile has already been loaded and cached in
        the `_profiles` attribute. If not, it attempts to load the profile from the HDF5
        file associated with the grid manager. If loading fails due to a missing profile,
        it returns `None`.

        Parameters
        ----------
        field_name : str
            The name of the field for which the profile is requested.

        Returns
        -------
        Profile or None
            The corresponding profile object if successfully loaded or already cached,
            otherwise returns `None` if the profile is not found or cannot be loaded.

        Raises
        ------
        IOError
            If loading the profile from HDF5 fails due to unexpected issues (other than
            a missing profile).

        Examples
        --------
        .. code-block:: python

            model = ClusterModel("path/to/model/file")
            density_profile = model.get_profile("density")
            if density_profile is not None:
                print("Density profile loaded successfully!")
            else:
                print("Failed to load density profile.")

        Notes
        -----
        Profiles are lazily loaded, meaning they are only read from disk if they have not
        been previously accessed. This ensures that memory usage is minimized when not all
        profiles are needed simultaneously.
        """

        # Return the profile if it is already loaded in memory
        if field_name in self._profiles:
            return self._profiles[field_name]

        # Attempt to load the profile from the HDF5 file
        try:
            profile = Profile.from_hdf5(
                hdf5_file=self.grid_manager._handle.handle,
                group_path=f"PROFILES/{field_name.upper()}",
                geometry=self.geometry,
            )
            # Cache the loaded profile
            self._profiles[field_name] = profile
        except KeyError:
            # Profile not found in HDF5, return None
            profile = None
        except Exception as e:
            # An unexpected error occurred during loading
            raise IOError(f"Failed to load profile '{field_name}': {e}")

        return profile

    def has_profile(self, profile_name: str) -> bool:
        """
        Check if a specific profile is available either in memory or on disk.

        This method first checks if the profile has already been loaded into memory
        (i.e., stored in the `_profiles` attribute). If not, it attempts to check the
        HDF5 file associated with the grid manager to determine whether the profile is
        available on disk.

        Parameters
        ----------
        profile_name : str
            The name of the profile to check.

        Returns
        -------
        bool
            Returns `True` if the profile exists either in memory or in the HDF5 file,
            `False` otherwise.

        Examples
        --------
        .. code-block:: python

            model = ClusterModel("path/to/model/file")
            if model.has_profile("temperature"):
                print("Temperature profile is available.")
            else:
                print("Temperature profile is not available.")

        Notes
        -----
        This method first checks the cached profiles in memory for faster access. If the
        profile has not been loaded yet, it then checks the HDF5 file on disk.
        """
        # Check if the profile is loaded in memory
        if profile_name in self._profiles:
            return True

        # Check if the profile exists in the HDF5 file
        return profile_name.upper() in self.grid_manager._handle["PROFILES"].keys()

    def add_profile(
            self,
            profile_name: str,
            profile: "Profile",
            create_field: bool = True,
            in_hdf5: bool = True,
            overwrite: bool = False,
            **kwargs,
    ) -> None:
        """
        Add a profile to the model, optionally creating a field and saving it to HDF5.

        Parameters
        ----------
        profile_name : str
            Name of the profile to add.
        profile : Profile
            The profile object to add. This profile contains the relevant data for the model.
        create_field : bool, optional
            Whether to create a field in the grid manager based on the profile. Default is True.
        in_hdf5 : bool, optional
            Whether to store the profile in the HDF5 file. If False, the profile will only be stored
            in memory. Default is True.
        overwrite : bool, optional
            Whether to overwrite an existing profile in memory or HDF5. Default is False.
        **kwargs : dict
            Additional keyword arguments passed to the field creation process (if applicable).

        Raises
        ------
        ValueError
            If the profile's geometry does not match the model's geometry, or if the profile already
            exists and overwrite is False.

        Notes
        -----
        This method ensures that the profile geometry aligns with the model's geometry, handles overwriting
        and creation of associated fields, and optionally writes the profile to HDF5 storage.

        Examples
        --------
        .. code-block:: python

            profile = DensityProfile(...)
            model.add_profile("density", profile, create_field=True, in_hdf5=True)

        See Also
        --------
        :py:meth:`ClusterModel.remove_profile`

        """
        profile_name = profile_name.lower()
        # Ensure profile geometry matches model geometry
        if profile.geometry_handler != self.geometry:
            raise ValueError(
                f"Profile geometry '{profile.geometry_handler}' does not match the model geometry '{self.geometry}'"
            )

        # Handle overwriting conditions
        if overwrite and self.has_profile(profile_name):
            del self.grid_manager._handle["PROFILES"][profile_name.upper()]
        elif not overwrite and self.has_profile(profile_name):
            raise ValueError(
                f"Profile '{profile_name}' already exists. Use overwrite=True to replace it."
            )

        # Save the profile to HDF5 if requested
        if in_hdf5:
            profile.to_hdf5(
                self.grid_manager._handle.handle, group_path=f"PROFILES/{profile_name.upper()}"
            )

        # Add the profile to the in-memory profile cache
        self._profiles[profile_name] = profile

        # Optionally create the corresponding field in the grid manager
        if create_field:
            self.grid_manager.add_field_from_profile(
                profile, profile_name, overwrite=overwrite, **kwargs
            )

    def remove_profile(self, profile_name: str, remove_field: bool = True) -> None:
        """
        Remove a profile from the model and optionally remove the corresponding field.

        Parameters
        ----------
        profile_name : str
            The name of the profile to remove from the model.
        remove_field : bool, optional
            If True, also removes the associated field from the grid manager. Default is True.

        Raises
        ------
        KeyError
            If the specified profile does not exist in the model.

        Notes
        -----
        This method will remove both in-memory and on-disk (HDF5) representations of the profile,
        ensuring that the model remains consistent. If `remove_field` is set to True, the associated
        field (if any) will also be removed from the grid manager.

        Examples
        --------
        .. code-block:: python

            model.remove_profile("density")

        See Also
        --------
        :py:meth:`ClusterModel.add_profile`

        """
        profile_name = profile_name.lower()
        # Check if the profile exists in memory or HDF5
        if not self.has_profile(profile_name):
            raise KeyError(f"Profile '{profile_name}' does not exist.")

        # Remove the profile from the in-memory cache
        if profile_name in self._profiles:
            del self._profiles[profile_name]

        # Remove the profile from the HDF5 file
        if profile_name in self.grid_manager._handle.get("PROFILES", {}):
            del self.grid_manager._handle["PROFILES"][profile_name.upper()]

        # Optionally remove the associated field from the grid manager
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
        mylog.info("Generating model from profiles...")
        mylog.debug("\tProfiles = %s.",list(profiles.keys()))

        # Perform validation steps. First we ensure the geometry is
        # self consistent and then produce the grid manager.
        geometry = cls._validate_input_profiles(**profiles)
        mylog.debug("\tGeometry = %s.",geometry)
        grid_manager = cls._validate_and_generate_grid(
            filename=filename,
            geometry=geometry,
            bbox=bbox,
            block_size=block_size,
            overwrite=overwrite,
        )

        # Add the geometry to the HDF5 file. This is required for
        # a valid Model HDF5 file.
        from cluster_generator.pipelines.solvers.common import ProfileInjectionSolver
        geometry.to_hdf5(grid_manager._handle.handle, "GEOMETRY")

        # Create the model instance and proceed with populating
        # the relevant pipeline.
        model = cls(filename,grid_manager=grid_manager)
        pipeline = Pipeline()

        _previous_field = "start"
        for field_name,profile in profiles.items():
            profile.to_hdf5(grid_manager._handle.handle, f"PROFILES/{field_name.upper()}")
            pipeline.tasks.add_task(field_name,ProfileInjectionSolver(field_name))
            pipeline.procedure.add_edge(_previous_field, field_name,lambda _,__,___: True)
            _previous_field = field_name

        pipeline.procedure.add_edge(_previous_field,'end',lambda _,__,___: True)
        model.set_pipeline(pipeline)

        return model

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
        from cluster_generator.pipelines.common import DensityTemperaturePipeline
        mylog.info("Generating model from density and temperature...")

        # Validating.
        profiles = {'temperature': temperature,'density': density}
        geometry = cls._validate_input_profiles(**profiles)
        mylog.debug("\tGeometry = %s.",geometry)
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

        # Create the model instance and proceed with populating
        # the relevant pipeline.
        model = cls(filename,grid_manager=grid_manager)
        pipeline = DensityTemperaturePipeline()

        for field_name,profile in profiles.items():
            profile.to_hdf5(grid_manager._handle.handle, f"PROFILES/{field_name.upper()}")

        model.set_pipeline(pipeline)

        return model



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


if __name__ == '__main__':
    from cluster_generator.profiles.density import NFWDensityProfile
    from cluster_generator.profiles.temperature import VikhlininTemperatureProfile
    import h5py


    Pipeline.logger.setLevel('TRACE')
    GridManager.logger.setLevel('INFO')
    a = NFWDensityProfile()
    b = VikhlininTemperatureProfile(T_0=12)

    h = ClusterModel.from_dens_and_temp('test.hdf5',a,b,[0,1000],block_size=[10000],overwrite=True)

    h.pipeline.validate_pipeline(require_setup=True,force=True,validate_tasks=True)
    h.pipeline(h.grid_manager[0,0])
    h.grid_manager.add_level(10)
    h.grid_manager[1].add_grid(0)
    h.grid_manager[1].add_grid(1)
    h.grid_manager[1].add_grid(2)
    h.grid_manager[1].add_grid(3)
    h.grid_manager[1].add_grid(4)
    h.grid_manager[1].add_grid(5)
    for i in range(6):
        h.pipeline(h.grid_manager[1,i])


    import matplotlib.pyplot as plt
    r = h.grid_manager[0,0].get_coordinates().ravel()
    p = h.grid_manager[0,0]['potential'][...]
    plt.loglog(r, -p)
    for i in range(6):
        r1 = h.grid_manager[1,i].get_coordinates().ravel()
        p1 = h.grid_manager[1,i]['potential'][...]

#

        plt.semilogy(r1,-p1)
    plt.show()
