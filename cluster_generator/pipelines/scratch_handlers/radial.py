from cluster_generator.pipelines.scratch_handlers.abc import ScratchSpaceHandler
from typing import TYPE_CHECKING, Tuple, Union
import h5py
import numpy as np
from cluster_generator.grids.grids import Grid
from .abc import ScratchSpaceHandler
from ._except import ScratchHandlerException
from numpy.typing import NDArray
from scipy.interpolate import InterpolatedUnivariateSpline

if TYPE_CHECKING:
    from cluster_generator.pipelines.abc import Pipeline
    from cluster_generator.geometry.radial import RadialGeometryHandler

class RadialScratchLevel:
    def __init__(self, scratch_handler: 'RadialScratchHandler', level_id: int):

        # Set generic attributes.
        self.scratch = scratch_handler
        self.pipeline = self.scratch.pipeline
        self.level_id = level_id

        self.handle = self._init_handle()
        self.radii = self._initialize_radii()
        self.profiles = self._initialize_profiles()


    def _init_handle(self) -> h5py.Group:
        # Construct the handle name and ensure that the
        # handle exists.
        handle_name = f'LEVEL_{self.level_id}'
        handle = self.scratch.handle.require_group(handle_name)

        # Ensure that the DELTA attribute is present.
        if 'DELTA' not in handle.attrs:
            handle.attrs['DELTA'] = self.scratch.compute_delta_r(self.level_id)

        # Return the handler
        return handle

    def _initialize_radii(self) -> h5py.Dataset:
        """Initialize or retrieve the RADII dataset in the level group."""
        if 'RADII' not in self.handle:
            return self.handle.create_dataset('RADII', shape=(0,), maxshape=(None,),
                                          dtype='float64')
        else:
            return self.handle['RADII']

    def _initialize_profiles(self) -> dict:
        """Initialize a dictionary for profile datasets in the level group."""
        return {k: self.handle[k] for k in self.handle.keys() if k != 'RADII'}

    def __getitem__(self,key: str):
        return self.profiles[key]

    def __delitem__(self, key: str):
        try:
            del self.handle[key]
        except KeyError:
            raise ScratchHandlerException(f'Failed to remove {key} from {self} because it doesn\'t exist.')

    def __setitem__(self, key: str, value: h5py.Dataset|NDArray):
        if isinstance(value, h5py.Dataset):
            data = value[...]
        elif isinstance(value, np.ndarray):
            data = value
        else:
            raise TypeError(f"Cannot set {key} to {value} in {self} because it has type {type(value)} which is not"
                            f" a dataset or an array.")

        if key not in self.handle:
            # We are creating a new dataset.
            self.handle.create_dataset(key,data=data)
        else:
            self.handle[key][...] = data

    def __contains__(self, item: str):
        return item in self.profiles

    def contains_radii(self, radii: np.ndarray, tolerance: float = None) -> Tuple[bool, NDArray[int] | None]:
        """
        Check if the provided radii are present within the stored radii, with a tolerance.

        Parameters
        ----------
        radii : np.ndarray
            Array of radii to check.
        tolerance : float, optional
            Tolerance for matching radii. Defaults to `delta_r / 2` if not provided.

        Returns
        -------
        Tuple[bool, NDArray[int] | None]
            Whether all radii are present, and the corresponding indices if they are.

        Raises
        ------
        ValueError
            If the input radii array is empty.
        """
        if radii.size == 0:
            raise ValueError("Input radii array is empty.")

        current_radii = self.radii[:]
        tolerance = self.handle.attrs['DELTA'] / 2 if tolerance is None else tolerance

        if current_radii.size == 0:
            return False, None

        # Find the closest indices in current_radii for each radius in input radii
        radii = np.sort(radii)
        indices = np.searchsorted(current_radii, radii)
        indices = np.clip(indices, 0, len(current_radii) - 1)
        right_distances = np.abs(current_radii[indices] - radii)
        left_indices = np.maximum(indices - 1, 0)
        left_distances = np.abs(current_radii[left_indices] - radii)
        min_distances = np.minimum(right_distances, left_distances)

        # Check if all minimum distances are within tolerance
        within_tolerance = min_distances <= tolerance

        if np.all(within_tolerance):
            # Select indices corresponding to minimum distances
            selected_indices = np.where(right_distances <= left_distances, indices, left_indices)
            return True, selected_indices
        return False, None

    def add_radii(self, radii: np.ndarray):
        """
        Add new radii to the RADII dataset, placing them in the correct positions based on `delta_r`.

        Parameters
        ----------
        radii : np.ndarray
            Array of new radii values to add.
        """
        # Loading the current radius array and fetching the tolerance.
        current_radii = self.radii[:]
        delta_r = self.handle.attrs['DELTA']


        # Round and filter new radii for uniqueness
        new_radii_rounded = np.round(radii / delta_r) * delta_r
        new_radii_rounded = np.unique(new_radii_rounded)  # Ensure no duplicates

        # Find insertion indices and filter out radii already present
        insertion_indices = np.searchsorted(current_radii, new_radii_rounded)
        mask_new_radii = ~np.isclose(new_radii_rounded[:, None], current_radii, atol=delta_r / 2).any(axis=1)
        unique_new_radii = new_radii_rounded[mask_new_radii]

        if unique_new_radii.size > 0:
            # Update the radii array
            updated_radii = np.insert(current_radii, insertion_indices[mask_new_radii], unique_new_radii)
            self.radii.resize(updated_radii.shape)
            self.radii[...] = updated_radii

            # Extend each profile with NaNs for the added radii
            for profile in self.profiles.values():
                profile_data = profile[...]
                updated_profile = np.insert(profile_data, insertion_indices[mask_new_radii], np.nan)
                profile.resize((self.radii.size,))
                profile[...] = updated_profile

    def remove_radii(self, radii: np.ndarray):
        """
        Remove specific radii from the RADII dataset and associated profiles.

        Parameters
        ----------
        radii : np.ndarray
            Array of radii values to remove.

        Raises
        ------
        ValueError
            If no matching radii are found to remove.
        """
        # Load current radii
        current_radii = self.radii[:]
        delta_r = self.handle.attrs['DELTA']

        # Round the radii to remove to the nearest multiple of delta_r
        radii_to_remove_rounded = np.round(radii / delta_r) * delta_r

        # Find indices of the radii to remove
        removal_indices = np.isclose(current_radii[:, None], radii_to_remove_rounded, atol=delta_r / 2).any(axis=1)

        if not removal_indices.any():
            unmatched_radii = radii[
                ~np.isclose(current_radii[:, None], radii_to_remove_rounded, atol=delta_r / 2).any(axis=1)]
            raise ValueError(f"No matching radii found to remove. Unmatched radii: {unmatched_radii}")

        # Remove radii from the dataset
        remaining_radii = current_radii[~removal_indices]
        self.radii.resize(remaining_radii.shape)
        self.radii[...] = remaining_radii

        # Remove corresponding values from each profile
        for profile in self.profiles.values():
            profile_data = profile[...]
            updated_profile = profile_data[~removal_indices]
            profile.resize(updated_profile.shape)
            profile[...] = updated_profile
    def get_radii(self,radii: np.ndarray, tolerance: float = None):
        status,indices = self.contains_radii(radii,tolerance=tolerance)

        if status:
            return self.radii[indices]
        else:
            raise ScratchHandlerException(f"Failed to find all radii to within selected tolerance {tolerance}.")

    def contains_radii_range(self, min_radius: float, max_radius: float, tolerance: float = None) -> Tuple[bool, slice | None]:
        """
        Check if the radii array contains all radii within a specified range, and return the corresponding slice.

        Parameters
        ----------
        min_radius : float
            The minimum radius of the range.
        max_radius : float
            The maximum radius of the range.
        tolerance : float, optional
            Tolerance for matching radii. Defaults to `delta_r / 2` if not provided.

        Returns
        -------
        Tuple[bool, slice | None]
            A tuple where the first element is True if all radii in the range are found,
            and the second element is a slice object representing the start and end indices of the matched range.
            If the range is not found, the first element is False and the second element is None.
        """
        # Generate the array of radii for the requested range
        delta_r = self.handle.attrs['DELTA']
        radii_range = np.arange(min_radius, max_radius + delta_r, delta_r)

        # Use the existing contains_radii method to check if these radii are within the dataset
        status, indices = self.contains_radii(radii_range, tolerance)

        if status:
            # If the radii are contained within the tolerance, convert the indices to a slice
            start_idx = indices[0]
            end_idx = indices[-1] + 1  # End is inclusive in slice, so +1
            return True, slice(start_idx, end_idx)

        # If not all radii are contained, return False
        return False, None

    def add_radii_range(self,min_radius: float, max_radius: float):
        # Generate the array of radii for the requested range
        delta_r = self.handle.attrs['DELTA']
        radii_range = np.arange(min_radius, max_radius + delta_r, delta_r)

        self.add_radii(radii_range)

    def remove_radii_range(self,min_radius: float, max_radius: float):
        # Generate the array of radii for the requested range
        delta_r = self.handle.attrs['DELTA']
        radii_range = np.arange(min_radius, max_radius + delta_r, delta_r)

        self.remove_radii(radii_range)

    def get_radii_range(self, min_radius: float, max_radius: float, tolerance: float = None):
        status, slc = self.contains_radii_range(min_radius,max_radius,tolerance=tolerance)

        if status:
            return self.radii[slc]
        else:
            raise ScratchHandlerException(f"Failed to find all radii to within selected tolerance {tolerance}.")

    def get_radii_slice(self, min_radius: float, max_radius: float, tolerance: float = None) -> slice:
        status, slc = self.contains_radii_range(min_radius,max_radius,tolerance=tolerance)

        if status:
            return slc
        else:
            raise ScratchHandlerException(f"Failed to find all radii to within selected tolerance {tolerance}.")

    def contains_radii_for_grid(self, grid: "Grid", buffer: int = 3, tolerance: float = None) -> Tuple[
        bool, slice | None]:
        """
        Check if all radii for the grid's bounding box, with optional buffer, are present.

        Parameters
        ----------
        grid : Grid
            The grid object with a bounding box (BBOX) attribute.
        buffer : int, optional
            The additional buffer to add around the grid radii. Default is 3.
        tolerance : float, optional
            Tolerance for matching radii. Defaults to `delta_r / 2`.

        Returns
        -------
        bool
            True if all required radii are present, False otherwise.

        Raises
        ------
        ValueError
            If the grid object is invalid or has no bounding box attribute.
        """
        if not hasattr(grid, "BBOX"):
            raise ValueError(f"Invalid grid object: {grid}. Bounding box attribute is required.")

        min_r, max_r = self.scratch.geometry.get_min_max_radii_from_grid(grid)

        # Apply the buffer
        delta_r = self.handle.attrs['DELTA']
        min_r = max(min_r - buffer * delta_r, delta_r)  # Ensure we don't go below zero radii
        max_r += buffer * delta_r
        return self.contains_radii_range(min_r, max_r, tolerance=tolerance)

    def add_radii_from_grid(self, grid: "Grid", buffer: int = 3):
        """
        Add radii for a grid's bounding box, with an optional buffer.

        Parameters
        ----------
        grid : Grid
            The grid object with a bounding box (BBOX) attribute.
        buffer : int, optional
            The buffer (in delta_r units) to add around the grid's radii range. Default is 3.
        """
        min_r, max_r = self.scratch.geometry.get_min_max_radii_from_grid(grid)

        # Apply the buffer
        delta_r = self.handle.attrs['DELTA']
        min_r = max(min_r - buffer * delta_r, 0)  # Ensure we don't go below zero radii
        max_r += buffer * delta_r

        self.add_radii_range(min_r, max_r)

    def remove_radii_from_grid(self,grid,buffer:int=3):
        min_r, max_r = self.scratch.geometry.get_min_max_radii_from_grid(grid)

        # Apply the buffer
        delta_r = self.handle.attrs['DELTA']
        min_r = max(min_r - buffer * delta_r, 0)  # Ensure we don't go below zero radii
        max_r += buffer * delta_r

        return self.remove_radii_range(min_r,max_r)

    def get_radii_from_grid(self, grid: "Grid", buffer: int = 3, tolerance: float = None):
        """
        Retrieve the slice of radii for a grid's bounding box, with optional buffer.

        Parameters
        ----------
        grid : Grid
            The grid object with a bounding box (BBOX) attribute.
        buffer : int, optional
            The buffer (in delta_r units) to add around the grid's radii range. Default is 3.
        tolerance : float, optional
            Tolerance for matching radii. Defaults to `delta_r / 2`.

        Returns
        -------
        np.ndarray
            The radii slice within the bounding box of the grid, adjusted by the buffer.

        Raises
        ------
        ScratchHandlerException
            If radii cannot be found within the selected tolerance.
        """
        status, slc = self.contains_radii_for_grid(grid, buffer, tolerance=tolerance)

        if status:
            return self.radii[slc]
        else:
            raise ScratchHandlerException(f"Failed to find all radii within selected tolerance {tolerance}.")

    def get_slc_from_grid(self, grid: "Grid", buffer: int = 3, tolerance: float = None):

        status, slc = self.contains_radii_for_grid(grid, buffer, tolerance=tolerance)

        if status:
            return slc
        else:
            raise ScratchHandlerException(f"Failed to find all radii within selected tolerance {tolerance}.")

    def get_radii_and_slc_from_grid(self, grid: "Grid", buffer: int = 3, tolerance: float = None):
        status, slc = self.contains_radii_for_grid(grid, buffer, tolerance=tolerance)

        if status:
            return self.radii[slc], slc
        else:
            raise ScratchHandlerException(f"Failed to find all radii within selected tolerance {tolerance}.")

    def get_profile_from_radii(self, profile: str, radii: np.ndarray, tolerance: float = None) -> np.ndarray:
        """
        Retrieve the profile values corresponding to the given radii.

        Parameters
        ----------
        profile : str
            The name of the profile to retrieve data from.
        radii : np.ndarray
            Array of radii for which the profile data is requested.
        tolerance : float, optional
            Tolerance for matching radii. Defaults to `delta_r / 2` if not provided.

        Returns
        -------
        np.ndarray
            The profile values corresponding to the requested radii. If some radii
            are not found within the tolerance, NaN will be returned for those values.

        Raises
        ------
        KeyError
            If the profile does not exist.
        ValueError
            If radii cannot be matched within the given tolerance.
        """
        # Check if the requested profile exists
        if profile not in self.profiles:
            raise KeyError(f"Profile '{profile}' does not exist in level {self.level_id}.")

        # Load the current radii and profile data
        current_radii = self.radii[:]
        profile_data = self.profiles[profile][:]
        tolerance = self.handle.attrs['DELTA'] / 2 if tolerance is None else tolerance

        # If no current radii exist, we cannot match anything
        if current_radii.size == 0:
            raise ValueError("No radii available to match.")

        # Sort input radii
        radii = np.sort(radii)

        # Find the closest indices in current_radii for each radius in input radii
        indices = np.searchsorted(current_radii, radii)
        indices = np.clip(indices, 0, len(current_radii) - 1)

        # Check if the radii are within tolerance
        within_tolerance = np.abs(current_radii[indices] - radii) <= tolerance

        # Prepare the result array, filling unmatched radii with NaNs
        result = np.full(radii.shape, np.nan)
        result[within_tolerance] = profile_data[indices[within_tolerance]]

        return result

    def get_profile_from_range(self, profile: str,r_min,r_max,tolerance: float = None) -> np.ndarray:
        """
        Retrieve the profile values corresponding to the radii within a specified range.

        This method retrieves profile data that falls between a minimum and maximum radius.
        It will return the values associated with the specified profile for radii within the
        range `[r_min, r_max]`. If radii in the range cannot be matched, a `ValueError` is raised.

        Parameters
        ----------
        profile : str
            The name of the profile to retrieve data from.
        r_min : float
            The minimum radius of the range.
        r_max : float
            The maximum radius of the range.
        tolerance : float, optional
            Tolerance for matching radii. Defaults to `delta_r / 2` if not provided.

        Returns
        -------
        np.ndarray
            The profile values corresponding to the requested radii slice within the range.

        Raises
        ------
        KeyError
            If the profile does not exist in the level.
        ValueError
            If no current radii exist to match, or the radii cannot be matched within the given tolerance.
        """
        if profile not in self.profiles:
            raise KeyError(f"Profile '{profile}' does not exist in level {self.level_id}.")

        # Load the current radii and profile data
        current_radii = self.radii[:]

        # If no current radii exist, we cannot match anything
        if current_radii.size == 0:
            raise ValueError("No radii available to match.")

        _,slc = self.get_radii_slice(r_min,r_max,tolerance=tolerance)

        return self.profiles[profile][slc]

    def get_profile_from_grid(self, profile: str, grid: "Grid", buffer: int = 3, tolerance: float = None) -> np.ndarray:
        """
        Retrieve the profile values for a grid's bounding box, with an optional buffer.

        Parameters
        ----------
        profile : str
            The name of the profile to retrieve data from.
        grid : Grid
            The grid object with a bounding box (BBOX) attribute.
        buffer : int, optional
            The buffer (in delta_r units) to add around the grid's radii range. Default is 3.
        tolerance : float, optional
            Tolerance for matching radii. Defaults to `delta_r / 2`.

        Returns
        -------
        np.ndarray
            The profile values corresponding to the requested radii slice within the grid.

        Raises
        ------
        KeyError
            If the profile does not exist.
        ValueError
            If radii cannot be matched within the given tolerance.
        """
        if profile not in self.profiles:
            raise KeyError(f"Profile '{profile}' does not exist in level {self.level_id}.")

        # Get the radii slice from the grid's bounding box
        slc = self.get_slc_from_grid(grid, buffer, tolerance=tolerance)

        return self.profiles[profile][slc]

    def get_profile_values(self, profile_name: str, indices: np.ndarray) -> np.ndarray:
        """
        Retrieve values from a specific profile at specified indices, handling NaNs for unfilled regions.
        """
        if profile_name not in self.profiles:
            raise KeyError(f"Profile '{profile_name}' not found.")
        return self.profiles[profile_name][indices]

    def add_profile(self, profile_name: str):
        """
        Add a new profile to match the RADII dataset, initialized with NaNs.
        """
        if profile_name in self.profiles:
            raise ValueError(f"Profile '{profile_name}' already exists.")

        # Initialize new profile dataset with NaNs for existing radii length
        profile = self.handle.create_dataset(profile_name, shape=self.radii.shape,
                                             maxshape=(None,), dtype='float64')
        profile[...] = np.nan  # Fill initial values with NaN
        self.profiles[profile_name] = profile

    @staticmethod
    def _check_for_nans(profile_data: np.ndarray):
        """
        Helper method to check if the profile data contains NaNs.

        Parameters
        ----------
        profile_data : np.ndarray
            The profile data to check.

        Raises
        ------
        ValueError
            If NaNs are found in the profile data.
        """
        if np.isnan(profile_data).any():
            raise ValueError("Profile data contains NaNs, cannot create interpolation.")

    def get_spline_from_radii(self, profile: str, radii: np.ndarray, tolerance: float = None,**kwargs):
        """
        Get an InterpolatedUnivariateSpline object for the given radii and profile, with optional buffer.

        Parameters
        ----------
        profile : str
            The name of the profile to retrieve data from.
        radii : np.ndarray
            Array of radii to use for interpolation.
        tolerance : float, optional
            Tolerance for matching radii. Defaults to `delta_r / 2`.
        kwargs:
            Additional keyword arguments to pass to InterpolatedUnivariateSpline.

        Returns
        -------
        InterpolatedUnivariateSpline
            The spline object created from the profile data and radii.

        Raises
        ------
        ValueError
            If NaNs are found in the profile data or if radii cannot be matched.
        """
        # Get the profile and radii data
        profile_data = self.get_profile_from_radii(profile, radii, tolerance=tolerance)
        self._check_for_nans(profile_data)

        # Create and return the InterpolatedUnivariateSpline
        return InterpolatedUnivariateSpline(radii, profile_data, **kwargs)

    def get_spline_from_range(self, profile: str, r_min: float, r_max: float, tolerance: float = None,**kwargs):
        """
        Get an InterpolatedUnivariateSpline object for a range of radii and profile.

        Parameters
        ----------
        profile : str
            The name of the profile to retrieve data from.
        r_min : float
            The minimum radius.
        r_max : float
            The maximum radius.
        tolerance : float, optional
            Tolerance for matching radii. Defaults to `delta_r / 2`.
        kwargs:
            Additional keyword arguments to pass to InterpolatedUnivariateSpline.

        Returns
        -------
        InterpolatedUnivariateSpline
            The spline object created from the profile data within the range.

        Raises
        ------
        ValueError
            If NaNs are found in the profile data or if radii cannot be matched.
        """
        # Get the radii and profile data for the range
        profile_data = self.get_profile_from_range(profile, r_min, r_max, tolerance=tolerance)
        radii = self.get_radii_range(r_min,r_max,tolerance=tolerance)

        return InterpolatedUnivariateSpline(radii, profile_data, k=3,**kwargs)

    def get_spline_from_grid(self, profile: str, grid: "Grid", buffer: int = 3, tolerance: float = None,**kwargs):
        """
        Get an InterpolatedUnivariateSpline object for a grid's bounding box and profile.

        Parameters
        ----------
        profile : str
            The name of the profile to retrieve data from.
        grid : Grid
            The grid object with a bounding box (BBOX) attribute.
        buffer : int, optional
            The buffer (in delta_r units) to add around the grid's radii range. Default is 3.
        tolerance : float, optional
            Tolerance for matching radii. Defaults to `delta_r / 2`.
        kwargs:
            Additional keyword arguments to pass to InterpolatedUnivariateSpline.

        Returns
        -------
        InterpolatedUnivariateSpline
            The spline object created from the profile data within the grid's bounding box.

        Raises
        ------
        ValueError
            If NaNs are found in the profile data or if radii cannot be matched.
        """
        # Get the profile and radii data for the grid's bounding box
        profile_data = self.get_profile_from_grid(profile, grid, buffer=buffer, tolerance=tolerance)

        # Check for NaNs in the profile data
        self._check_for_nans(profile_data)

        # Get the radii slice for the grid
        radii_slice = self.get_slc_from_grid(grid, buffer, tolerance=tolerance)

        return InterpolatedUnivariateSpline(self.radii[radii_slice], profile_data, **kwargs)

    def set_profile_from_function(self,profile: str,function, grid: 'Grid', **kwargs):
        radii_slice = self.get_slc_from_grid(grid,**kwargs)
        radii = self.radii[radii_slice]

        profile_values = function(radii)
        self.profiles[profile][radii_slice] = profile_values


class RadialScratchHandler(ScratchSpaceHandler):
    """
    RadialScratchHandler manages the levels within the radial scratch space for a given pipeline.
    It handles adding, removing, and overwriting levels, and provides utility methods for accessing
    and managing the radial data at different levels.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline associated with this handler.
    """

    def __init__(self, pipeline: 'Pipeline'):
        """
        Initialize the RadialScratchHandler with a given pipeline.
        The levels dictionary is used to track different radial scratch levels.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to which this handler is associated.
        """
        super().__init__(pipeline)
        self.levels = {}

    @property
    def geometry(self) -> 'RadialGeometryHandler':
        """
        Return the RadialGeometryHandler associated with the current model's geometry.

        Returns
        -------
        RadialGeometryHandler
            The geometry handler for this pipeline.
        """
        return self.pipeline.model.geometry

    def compute_delta_r(self, level_id: int) -> float:
        """
        Compute the delta_r (spacing between radii) for a specific level.
        This method queries the RadialGeometryHandler to get the minimum radii spacing
        for the grid associated with the given level.

        Parameters
        ----------
        level_id : int
            The ID of the level for which delta_r is being computed.

        Returns
        -------
        float
            The delta_r value.

        Raises
        ------
        ScratchHandlerException
            If an error occurs during delta_r computation, or if the computed delta_r is invalid.
        """
        try:
            delta_r = self.geometry.get_min_radii_spacing(self.pipeline.model.grid_manager[level_id])
            if delta_r <= 0:
                raise ValueError(f"Invalid delta_r value computed for level {level_id}: {delta_r}")
        except Exception as e:
            raise ScratchHandlerException(f'Failed to compute delta_r for level {level_id} of {self.pipeline}: {e}')

        return delta_r

    def add_level(self, level_id: int) -> RadialScratchLevel:
        """
        Add a new level to the RadialScratchHandler. If the level already exists, raise an error.

        Parameters
        ----------
        level_id : int
            The ID of the level to add.

        Returns
        -------
        RadialScratchLevel
            The newly created RadialScratchLevel.

        Raises
        ------
        ScratchHandlerException
            If the level already exists.
        """
        if self.pipeline.model is None:
            raise ScratchHandlerException(f"Cannot add level {level_id} because there is no model.")
        if level_id in self.levels:
            raise ScratchHandlerException(f"Level {level_id} already exists in {self}.")


        level = RadialScratchLevel(self, level_id)
        self.levels[level_id] = level
        return level

    def remove_level(self, level_id: int):
        """
        Remove a level from the RadialScratchHandler. If the level does not exist, raise an error.

        Parameters
        ----------
        level_id : int
            The ID of the level to remove.

        Raises
        ------
        ScratchHandlerException
            If the level does not exist.
        """
        if level_id not in self.levels:
            raise ScratchHandlerException(f"Level {level_id} does not exist in {self}.")

        # Remove the level's scratch data
        del self.levels[level_id]

    def overwrite_level(self, level_id: int) -> RadialScratchLevel:
        """
        Overwrite an existing level in the RadialScratchHandler. If the level exists,
        it will be removed and recreated. If it does not exist, it will be created.

        Parameters
        ----------
        level_id : int
            The ID of the level to overwrite.

        Returns
        -------
        RadialScratchLevel
            The newly created (or recreated) RadialScratchLevel.
        """
        if level_id in self.levels:
            self.remove_level(level_id)
        return self.add_level(level_id)

    def get_level(self, level_id: int) -> RadialScratchLevel:
        """
        Retrieve a specific level from the RadialScratchHandler.

        Parameters
        ----------
        level_id : int
            The ID of the level to retrieve.

        Returns
        -------
        RadialScratchLevel
            The RadialScratchLevel corresponding to the given level ID.

        Raises
        ------
        ScratchHandlerException
            If the level does not exist.
        """
        if level_id not in self.levels:
            raise ScratchHandlerException(f"Level {level_id} does not exist in {self}.")
        return self.levels[level_id]

    def has_level(self, level_id: int) -> bool:
        """
        Check if a specific level exists in the RadialScratchHandler.

        Parameters
        ----------
        level_id : int
            The ID of the level to check.

        Returns
        -------
        bool
            True if the level exists, False otherwise.
        """
        return level_id in self.levels

    def list_levels(self) -> list:
        """
        List all levels currently stored in the RadialScratchHandler.

        Returns
        -------
        list
            A list of level IDs.
        """
        return list(self.levels.keys())

    def clear_all_levels(self):
        """
        Clear all levels in the RadialScratchHandler, removing all stored radial data.

        Raises
        ------
        ScratchHandlerException
            If an error occurs while clearing the levels.
        """
        try:
            self.levels.clear()
        except Exception as e:
            raise ScratchHandlerException(f"Failed to clear all levels: {e}")


