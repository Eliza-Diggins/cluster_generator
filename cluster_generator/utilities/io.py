import h5py
from typing import Union, Any

class HDF5FileHandler:
    """
    A handler for managing datasets and groups within an HDF5 file or group.
    Supports dynamic creation, resizing, and handling of temporary data storage.
    """
    def __init__(self, handle: Union[h5py.File, h5py.Group, str], mode: str = "r+"):
        """
        Initialize the handler with an HDF5 file, group handle, or file path.

        Parameters
        ----------
        handle : Union[h5py.File, h5py.Group, str]
            The HDF5 file/group handle or file path to open.
        mode : str, optional
            The mode for opening the file (default is 'r+').
        """
        self._is_file = isinstance(handle, str)
        self.filename = handle if self._is_file else handle.file.filename

        if isinstance(handle, str):
            self.handle = h5py.File(handle, mode=mode)
        elif isinstance(handle, (h5py.File, h5py.Group)):
            self.handle = handle
        else:
            raise TypeError("Handle must be a file path, an instance of h5py.File, or h5py.Group")

        self.mode = mode

    def __getitem__(self, key: str) -> Any:
        """
        Access a dataset or group by key.

        Parameters
        ----------
        key : str
            The key to access the dataset or group.

        Returns
        -------
        Any
            The dataset or group at the specified key.
        """
        return self.handle[key]

    def __delitem__(self, key: str):
        """
        Delete a dataset or group by key.

        Parameters
        ----------
        key : str
            The key of the dataset or group to delete.
        """
        if key in self.handle:
            del self.handle[key]

    def __contains__(self, item: str) -> bool:
        """
        Check if a dataset or group exists.

        Parameters
        ----------
        item : str
            The key to check for existence.

        Returns
        -------
        bool
            True if the item exists, False otherwise.
        """
        return item in self.handle

    def __len__(self) -> int:
        """
        Return the number of datasets and groups.

        Returns
        -------
        int
            The number of items in the file/group.
        """
        return len(self.handle)

    @property
    def attrs(self) -> h5py.AttributeManager:
        """
        Return the attributes of the HDF5 file or group.

        Returns
        -------
        h5py.AttributeManager
            The attributes of the file or group.
        """
        return self.handle.attrs

    def get(self, item: str, default: Any = None) -> Any:
        """
        Retrieve a dataset or group by key, or return default if not found.

        Parameters
        ----------
        item : str
            The key of the dataset or group to retrieve.
        default : Any, optional
            The value to return if the item is not found (default is None).

        Returns
        -------
        Any
            The dataset or group if found, otherwise the default.
        """
        return self.handle.get(item, default)

    def switch_mode(self, mode: str):
        """
        Switch the mode of the HDF5 file.

        Parameters
        ----------
        mode : str
            The new mode to open the file with.

        Raises
        ------
        ValueError
            If trying to switch mode for a non-file handle.
        """
        if self._is_file:
            self.handle.close()
            self.handle = h5py.File(self.filename, mode=mode)
            self.mode = mode
        else:
            raise ValueError("Cannot switch mode for an HDF5 group handle.")

    def keys(self) -> list:
        """
        Return the keys (dataset and group names) in the file/group.

        Returns
        -------
        list
            A list of keys in the file/group.
        """
        return list(self.handle.keys())

    def items(self) -> list:
        """
        Return the items (datasets and groups) in the file/group.

        Returns
        -------
        list
            A list of items in the file/group.
        """
        return list(self.handle.items())

    def write_data(self, dataset_name: str, data: Any, overwrite: bool = True):
        """
        Write or overwrite a dataset in the file/group.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to write.
        data : Any
            The data to write.
        overwrite : bool, optional
            Whether to overwrite the dataset if it exists (default is True).
        """
        if dataset_name in self.handle:
            if overwrite:
                del self.handle[dataset_name]  # Overwrite existing dataset
            else:
                raise ValueError(f"Dataset '{dataset_name}' already exists.")
        self.handle.create_dataset(dataset_name, data=data)

    def flush(self):
        """
        Flush any changes to disk.
        """
        if self.handle is not None:
            self.handle.flush()

    def create_group(self, name: str, **kwargs) -> h5py.Group:
        """
        Create a new group in the HDF5 file/group.

        Parameters
        ----------
        name : str
            The name of the group to create.

        Returns
        -------
        h5py.Group
            The newly created group.
        """
        return self.handle.create_group(name, **kwargs)

    def close(self):
        """
        Close the HDF5 file if it was opened as a file handle.
        """
        if self._is_file and isinstance(self.handle, h5py.File):
            self.handle.close()

    def __enter__(self) -> 'HDF5FileHandler':
        """
        Enter the runtime context for the handler.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context, ensuring the file is closed if applicable.
        """
        self.close()

    def __str__(self) -> str:
        """
        Return a string representation of the HDF5FileHandler.

        Returns
        -------
        str
            String representation of the handler.
        """
        return f"HDF5FileHandler for '{self.handle.name}' with {len(self)} items."

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the HDF5FileHandler.

        Returns
        -------
        str
            Detailed string representation of the handler.
        """
        return f"<HDF5FileHandler(handle='{self.handle.name}', num_items={len(self)})>"

    def __del__(self):
        """
        Ensure the HDF5 file is closed properly when the object is deleted.
        """
        try:
            self.close()
        except Exception as e:
            # Log or handle any cleanup errors if needed
            print(f"Error closing the file: {e}")

    def delete(self):
        """
        Completely remove the HDF5 file from disk if it was opened from a file path,
        or remove the group if it was initialized with a group handle.

        Raises
        ------
        ValueError
            If the handler was created with neither a file nor a group.
        OSError
            If there is an issue removing the file from the filesystem.
        """
        import os
        if self._is_file:
            # If the handle represents a file, close it and delete the file.
            self.close()
            try:
                os.remove(self.filename)
                print(f"File '{self.filename}' has been deleted.")
            except OSError as e:
                raise OSError(f"Error deleting the file '{self.filename}': {e}")
        elif isinstance(self.handle, h5py.Group):
            # If the handle represents a group, attempt to delete the group.
            group_name = self.handle.name
            parent = self.handle.parent
            self.close()
            try:
                del parent[group_name]
                print(f"Group '{group_name}' has been deleted from parent '{parent.name}'.")
            except KeyError:
                raise ValueError(f"Group '{group_name}' could not be found in the parent group.")
        else:
            raise ValueError(
                "Cannot delete unknown handle type. Only file-based or group-based handlers can be deleted.")