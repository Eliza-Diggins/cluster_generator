import logging

import numpy as np

from cluster_generator.utilities.config import cgparams
from cluster_generator.utilities.logging import LogDescriptor


class GridManagerLogDescriptor(LogDescriptor):
    def configure_logger(self, logger):
        _handler = logging.StreamHandler()
        _handler.setFormatter(
            logging.Formatter(cgparams["logging"]["mylog"]["format"])
        )
        if len(logger.handlers) == 0:
            logger.addHandler(_handler)

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
