import logging
from abc import ABC, abstractmethod

import h5py
import numpy as np

from cluster_generator.utils import cgparams


class LogDescriptor(ABC):
    def __get__(self, instance, owner) -> logging.Logger:
        # check owner for an existing logger:
        logger = logging.getLogger(owner.__name__)

        self.configure_logger(logger)

        return logger

    @abstractmethod
    def configure_logger(self, logger):
        pass


class GridManagerLogDescriptor(LogDescriptor):
    def configure_logger(self, logger):
        _handler = logging.StreamHandler()
        _handler.setFormatter(
            logging.Formatter(cgparams["system"]["logging"]["main"]["format"])
        )
        if len(logger.handlers) == 0:
            logger.addHandler(_handler)


class HDF5FileHandler:
    """
    A simplified HDF5 file handler that always opens the file in read/write (r+) or append (a) mode,
    allowing both reading and writing without switching modes.
    """

    def __init__(self, filename, mode="r+"):
        """
        Initialize the HDF5 file handler.

        Parameters
        ----------
        filename : str
            The path to the HDF5 file.
        mode : str, optional
            The mode to open the file in ('r+' for read/write, 'a' for append), by default 'r+'.
        """
        self.filename = filename
        self.mode = mode
        self.handle = h5py.File(self.filename, mode=mode)

    def __getitem__(self, key):
        """
        Access a dataset by key.
        """
        return self.handle[key]

    def __delitem__(self, key):
        del self.handle[key]

    def __contains__(self, item):
        """
        Check if a dataset exists in the file.
        """
        return item in self.handle

    def __len__(self):
        """
        Return the number of items in the file.
        """
        return len(self.handle)

    @property
    def attrs(self):
        """
        Return the attributes of the HDF5 file.
        """
        return self.handle.attrs

    def get(self, item, default=None):
        return self.handle.get(item, default)

    def switch_mode(self, mode: str):
        """Switch the HDF5 file's mode dynamically."""
        if self.handle:
            self.handle.close()
        self.handle = h5py.File(self.filename, mode=mode)

    def keys(self):
        """
        Return a list of dataset keys in the file.
        """
        return list(self.handle.keys())

    def items(self):
        """
        Return a list of dataset items in the file.
        """
        return list(self.handle.items())

    def write_data(self, dataset_name, data):
        """
        Write or overwrite a dataset in the file.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to write.
        data : array-like
            The data to write to the dataset.
        """
        if dataset_name in self.handle:
            del self.handle[dataset_name]  # Overwrite the existing dataset
        self.handle.create_dataset(dataset_name, data=data)

    def flush(self):
        """
        Flush changes to disk without closing the file.
        """
        if self.handle is not None:
            self.handle.flush()

    def create_group(self, *args, **kwargs):
        self.handle.create_group(*args, **kwargs)

    def close(self):
        """
        Close the HDF5 file.
        """
        if self.handle is not None:
            self.handle.close()

    def __enter__(self):
        """
        Enter the runtime context and return the file handler.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context, ensuring the file is closed.
        """
        self.close()


def coerce_to_bounding_box(bbox):
    """
    Coerce the input into a bounding box represented as an ndarray of shape (2, DIM).

    Parameters
    ----------
    bbox : Union[list, tuple, ndarray]
        A representation of two coordinate points in space, e.g., [(x1, y1, z1), (x2, y2, z2)] for 3D
        or [x1, x2] for 1D.

    Returns
    -------
    ndarray
        A numpy array of shape (2, DIM), where DIM is the dimensionality of the bounding box.

    Raises
    ------
    ValueError
        If the input cannot be coerced into a valid bounding box format.
    """
    bbox = np.asarray(bbox)  # Convert to ndarray if it's a list or tuple

    # Handle the 1D case where bbox might be a single list or tuple, e.g., [x1, x2]
    if bbox.ndim == 1 and len(bbox) == 2:
        return bbox.reshape(2, 1)  # Return as (2, 1) for 1D

    # Ensure that bbox is of shape (2, DIM)
    if bbox.ndim == 2 and bbox.shape[0] == 2:
        return bbox  # Already in the desired shape

    raise ValueError(
        f"Invalid bounding box format: {bbox}. Expected two coordinate points in space."
    )


def coerce_to_domain_shape(domain_shape):
    """
    Coerce the input into a domain shape represented as an ndarray of integers with shape (DIM,).

    Parameters
    ----------
    domain_shape : Union[list, tuple, int]
        A representation of the domain shape. For example, (x, y, z) for 3D, or a single integer for 1D.

    Returns
    -------
    ndarray
        A numpy array of integers representing the domain shape with shape (DIM,).

    Raises
    ------
    ValueError
        If the input cannot be coerced into a valid domain shape format.
    """
    # Handle single integer input (1D case)
    if isinstance(domain_shape, int):
        return np.array([domain_shape], dtype=int)

    # Convert to ndarray if it's a list or tuple
    domain_shape = np.asarray(domain_shape, dtype=int)

    # Ensure it's a 1D array of integers
    if domain_shape.ndim == 1 and domain_shape.dtype == int:
        return domain_shape

    raise ValueError(
        f"Invalid domain shape format: {domain_shape}. Expected a list or tuple of integers."
    )
