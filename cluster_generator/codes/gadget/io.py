"""Module for handling Gadget file types for initial conditions.

.. rubric:: Developer Notes

- This module provides a comprehensive implementation of standard Gadget-2 file formats for initial conditions.
  The module allows for configurable behaviors and recognized fields, enabling easy extension for descendant file classes.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import Field, dataclass, field, fields
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    ClassVar,
    Literal,
    Self,
    Type,
    TypeVar,
    Union,
)

import h5py
import numpy as np
import unyt
from numpy.typing import NDArray

from cluster_generator.particles import ClusterParticles
from cluster_generator.utilities.logging import devlog, mylog
from cluster_generator.utilities.utils import reverse_dict

if TYPE_CHECKING:
    from yt.utilities.cosmology import Cosmology

GIC = TypeVar("GIC", bound="Gadget2_ICFile")
GICB = TypeVar("GICB", bound="Gadget2_Binary")
GICH = TypeVar("GICH", bound="Gadget2_HDF5")

# Mapping of Gadget particle types to cluster generator equivalent particle types
_GADGET_PARTICLES: dict[int, str | None] = {
    0: "gas",
    1: "dm",
    2: None,
    3: "tracer",
    4: "star",
    5: "black_hole",
}

# Default units and naming conventions for expected fields in Gadget files.
# !Ordering matters for blocks in binary.
_GADGET_FIELDS: OrderedDict[
    str, tuple[str, str, str, str, list[int] | None, int]
] = OrderedDict(
    {
        "Coordinates": ("particle_position", "POS", "kpc", "f4", None, 3),
        "Velocities": ("particle_velocity", "VEL", "km/s", "f4", None, 3),
        "ParticleIDs": ("particle_index", "ID", None, "i4", None, 1),
        "Masses": ("particle_mass", "MASS", "1e10*Msun", "f4", None, 1),
        "InternalEnergy": ("thermal_energy", "U", "km**2/s**2", "f4", [0], 1),
        "Density": ("density", "DEN", "1e10*Msun/kpc**3", "f4", [0], 1),
        "Potential": ("potential_energy", "POT", "km**2/s**2", "f4", None, 1),
        "MagneticField": (
            "magnetic_field",
            "MAG",
            "1e5*sqrt(Msun)*km/s/(kpc**1.5)",
            "f4",
            [0],
            3,
        ),
        "PassiveScalars": ("passive_scalars", "PASS", "", "f4", [0], 1),
    }
)

# Mapping between header fields in binary and HDF5 Gadget formats
_GADGET_HEADERS: dict[str, str] = {
    "Npart": "NumPart_ThisFile",
    "Massarr": "MassTable",
    "Time": "Time",
    "Redshift": "Redshift",
    "FlagSfr": "Flag_Sfr",
    "FlagFeedback": "Flag_Feedback",
    "Nall": "NumPart_Total",
    "FlagCooling": "Flag_Cooling",
    "NumFiles": "NumFilesPerSnapshot",
    "BoxSize": "BoxSize",
    "Omega0": "Omega0",
    "OmegaLambda": "OmegaLambda",
    "HubbleParam": "HubbleParam",
    "FlagAge": "Flag_StellarAge",
    "FlagMetals": "Flag_Metals",
    "NallHW": "NumPart_Total_HW",
    "flag_entr_ics": "Flag_Entropy_ICs",
}
# Default float type for Gadget files
_GADGET_DEFAULT_FLOAT_TYPE: str = "f4"
# Standard group prefix in HDF5.
_GADGET_GROUP_PREFIX: str = "Type"

# We utilize our own implementation of the standard binary Fortran file because
# scipy.FortranFile has severe limitations in its ability to move within the file
# to read / write interchangeably, etc.


def write_record(file: BinaryIO, record: NDArray) -> None:
    """Write a record to a binary file in Fortran-style format.

    Parameters
    ----------
    file : BinaryIO
        A file-like object opened in binary mode (e.g., 'wb', 'rb+', 'ab').
    record : NDArray
        The record to write, which should be a NumPy array with a specific dtype.

    Notes
    -----
    This function writes the record size in bytes before and after the record data,
    as required by the Fortran binary format.
    """
    record_size = record.nbytes
    record_size_bytes = np.array(record_size, dtype=np.uint32).tobytes()

    # writing the data, including the leader and follower.
    file.write(record_size_bytes)
    file.write(record.tobytes())
    file.write(record_size_bytes)


def read_record(file: BinaryIO, dtype: np.dtype) -> NDArray:
    """Read a record from a binary file formatted in Fortran-style.

    Parameters
    ----------
    file : BinaryIO
        A file-like object opened in binary mode (e.g., 'rb').
    dtype : np.dtype
        The numpy dtype object representing the structure of the data to read.

    Returns
    -------
    NDArray
        The data read from the record, converted to the specified dtype.

    Raises
    ------
    EOFError
        If the end of the file is reached unexpectedly.
    ValueError
        If the leading and trailing size markers do not match.
    """
    # Read the leading int32 to determine the record size.
    leading_size_bytes = file.read(4)
    if len(leading_size_bytes) < 4:
        raise EOFError("Unexpected end of file while reading the leading size marker.")

    record_size = int(np.frombuffer(leading_size_bytes, dtype=np.uint32)[0])

    # We can now read the record.
    record_data = file.read(record_size)
    if len(record_data) < record_size:
        raise EOFError("Unexpected end of file while reading the record data.")

    # We now read the follower to prepare for the next read.
    trailing_size_bytes = file.read(4)
    if len(trailing_size_bytes) < 4:
        raise EOFError("Unexpected end of file while reading the trailing size marker.")

    # Convert the trailing size marker to an integer
    trailing_size = int(np.frombuffer(trailing_size_bytes, dtype=np.uint32)[0])

    # Verify that the leading and trailing size markers match
    if record_size != trailing_size:
        raise ValueError("Mismatch between leading and trailing size markers.")

    # Convert the raw data to a numpy array with the specified dtype
    data = np.frombuffer(record_data, dtype=dtype)
    return data


def skip_record(file: BinaryIO) -> int:
    """Skip a record in a binary file formatted in Fortran-style.

    Parameters
    ----------
    file : BinaryIO
        A file-like object opened in binary mode (e.g., 'rb').

    Raises
    ------
    EOFError
        If the end of the file is reached unexpectedly.
    """
    # Read the leading size marker (4 bytes for Fortran record size)
    record_size_bytes = file.read(4)
    if len(record_size_bytes) < 4:
        raise EOFError("Unexpected end of file while reading the record size.")

    record_size = int(np.frombuffer(record_size_bytes, dtype=np.uint32)[0])

    # Skip the record data and the trailing size marker
    file.seek(
        record_size, 1
    )  # Skip forward by `record_size` bytes (relative to current position)

    trailing_size_bytes = file.read(4)
    if len(trailing_size_bytes) < 4:
        raise EOFError("Unexpected end of file while reading the trailing size marker.")

    return record_size


def read_block_specifier(file: BinaryIO) -> str:
    """Reads the block specifier from a generic binary file-like object.

    Parameters
    ----------
    file : BinaryIO
        A file-like object opened in binary mode that supports reading.

    Returns
    -------
    str
        The decoded block specifier.

    Raises
    ------
    IOError
        If there is an issue reading from the file.
    """
    try:
        block_specifier = read_record(file, np.dtype("S4"))[0]

        # Decode and return the block specifier
        return block_specifier.decode("utf-8").rstrip()

    except (OSError, ValueError) as e:
        raise IOError(f"Failed to read block specifier: {e}")


@dataclass
class Gadget2_Header:
    """Represents the header of a Gadget-2 binary or HDF5 file.

    The header contains essential metadata and configuration details about the simulation,
    such as the number of particles, the box size, cosmological parameters, and flags
    indicating the presence of various physical processes. This class provides functionality
    for reading and writing headers in both binary and HDF5 formats and converting between
    different representations.

    Notes
    -----
    - This class is essential for managing the metadata required to correctly interpret and process
      Gadget-2 files, both in binary and HDF5 formats.
    - The header is critical in ensuring consistency across different parts of the simulation data,
      including particle counts, simulation box size, and cosmological parameters.
    - From a development standpoint, this class can be easily subclassed with changes to various class variables
      to implement just about any variation of the standard header scheme, including adding new flags or changing
      datatypes.
    """

    HEADER_FLOAT_TYPE: ClassVar[str | np.dtype] = np.float64
    """dtype: The default float type for header values."""
    HEADER_UINT_TYPE: ClassVar[str | np.dtype] = np.uint32
    """dtype: The default uint type for header values."""
    HEADER_INT_TYPE: ClassVar[str | np.dtype] = np.int32
    """dtype: The default int type for header values."""
    HEADER_ORDER: ClassVar[list[str]] = list(_GADGET_HEADERS.keys())
    """List[str]: The order of header keys and values.

    This is only necessary in binary format where the order of the header matters.
    """
    HEADER_SIZE: ClassVar[int] = 256
    """int: The number of bytes this header takes up."""
    N_PARTICLE_TYPES: ClassVar[int] = 6
    """int: The number of particle types recognized by this header."""

    @staticmethod
    def _generate_header_field(
        ftype: np.dtype, utype: np.dtype, itype: np.dtype
    ) -> Callable[[...], Field]:
        """Generates a dataclass field for a Gadget-2 header.

        Parameters
        ----------
        ftype : np.dtype
            The float type to use for floating-point fields.
        utype : np.dtype
            The unsigned integer type to use for unsigned integer fields.
        itype : np.dtype
            The integer type to use for integer fields.

        Returns
        -------
        Callable[..., Field]
            A function that generates a dataclass field for the header.
        """

        def wrapper(*args, **kwargs):
            metadata = kwargs.pop("metadata", {})
            metadata["dtype"] = args[0]

            if args[0] == "f":
                metadata["dtype"] = ftype
            elif args[0] == "u":
                metadata["dtype"] = utype
            elif args[0] == "i":
                metadata["dtype"] = itype
            else:
                pass

            metadata["binary_flag"] = args[1]
            metadata["size"] = kwargs.pop("size", 1)

            default = kwargs.pop("default", None)
            default_factory = kwargs.pop("default_factory", None)

            if default is not None:
                default = metadata["dtype"](default)
                return field(*args[2:], metadata=metadata, default=default, **kwargs)
            elif default_factory is not None:
                default_factory = lambda df=default_factory: np.asarray(
                    df(), dtype=metadata["dtype"]
                )
                return field(
                    *args[2:],
                    metadata=metadata,
                    default_factory=default_factory,
                    **kwargs,
                )
            else:
                return field(*args[2:], metadata=metadata, **kwargs)

        return wrapper

    HEADER_FIELD: ClassVar[Callable[[...], Field]] = _generate_header_field(
        HEADER_FLOAT_TYPE, HEADER_UINT_TYPE, HEADER_INT_TYPE
    )

    # Defining the expected headers
    NumPart_ThisFile: NDArray[HEADER_UINT_TYPE] = HEADER_FIELD("u", "Npart", size=6)
    """array of uint: The number of particles of each type in this file."""
    NumPart_Total: NDArray[HEADER_UINT_TYPE] = HEADER_FIELD("i", "Nall", size=6)
    """array of uint: The total number of particles of each type across the entire simulation."""
    BoxSize: HEADER_FLOAT_TYPE = HEADER_FIELD("f", "BoxSize")
    """float: The size of the simulation box in comoving units."""
    MassTable: NDArray[HEADER_FLOAT_TYPE] = HEADER_FIELD(
        "f", "Massarr", size=6, default_factory=lambda: np.zeros(6)
    )
    """array of float: The fixed mass of each particle type. If 0, then the masses are stored in a separate block."""
    Time: HEADER_FLOAT_TYPE = HEADER_FIELD("f", "Time", default=0.0)
    """float: The time of this snapshot."""
    Redshift: HEADER_FLOAT_TYPE = HEADER_FIELD("f", "Redshift", default=0.0)
    """float: The redshift of this snapshot."""
    Flag_Sfr: HEADER_INT_TYPE = HEADER_FIELD("i", "FlagSfr", default=0)
    """int: Star formation flag (1 if star formation is included, 0 otherwise)."""
    Flag_Feedback: HEADER_INT_TYPE = HEADER_FIELD("i", "FlagFeedback", default=0)
    """int: Feedback flag (1 if feedback is included, 0 otherwise)."""
    Flag_Cooling: HEADER_INT_TYPE = HEADER_FIELD("i", "FlagCooling", default=0)
    """int: Cooling flag (1 if cooling is included, 0 otherwise)."""
    NumFilesPerSnapshot: HEADER_INT_TYPE = HEADER_FIELD("i", "NumFiles", default=1)
    """int: Number of files in which this snapshot is split."""
    Omega0: HEADER_FLOAT_TYPE = HEADER_FIELD("f", "Omega0", default=0.0)
    """float: The matter density parameter at redshift 0."""
    OmegaLambda: HEADER_FLOAT_TYPE = HEADER_FIELD("f", "OmegaLambda", default=0.0)
    """float: The cosmological constant density parameter at redshift 0."""
    HubbleParam: HEADER_FLOAT_TYPE = HEADER_FIELD("f", "HubbleParam", default=0.0)
    """float: The Hubble constant in units of 100 km/s/Mpc."""
    Flag_StellarAge: HEADER_INT_TYPE = HEADER_FIELD("i", "FlagAge", default=0)
    """int: Stellar age flag (1 if stellar ages are tracked, 0 otherwise)."""
    Flag_Metals: HEADER_INT_TYPE = HEADER_FIELD("i", "FlagMetals", default=0)
    """int: Metals flag (1 if metallicity is tracked, 0 otherwise)."""
    NumPart_Total_HW: NDArray[HEADER_UINT_TYPE] = HEADER_FIELD(
        "i", "NallHW", size=6, default_factory=lambda: np.zeros(6, dtype=np.uint32)
    )
    """array of uint: The high word of the total number of particles of each type."""
    Flag_Entropy_ICs: HEADER_INT_TYPE = HEADER_FIELD("i", "flag_entr_ics", default=0)
    """int: Entropy flag (1 if entropy is used as an initial condition, 0 otherwise)."""

    # Generate HDF5 to binary and binary to HDF5 name mappings from dataclass fields
    HDF5_TO_BINARY: ClassVar[dict[str, str]] = {}
    """Dict[str, str]: A mapping from HDF5 field names to their corresponding binary
    field names."""

    BINARY_TO_HDF5: ClassVar[dict[str, str]] = {}
    """Dict[str, str]: A mapping from binary field names to their corresponding HDF5
    field names."""

    @classmethod
    def generate_field_mapping(cls) -> dict[str, str]:
        """Generates a mapping from HDF5 name to binary name and vice versa.

        Returns
        -------
        dict[str, str]
            A dictionary mapping HDF5 field names to binary field names.
        """
        return {
            f.metadata["binary_flag"]: f.name
            for f in fields(cls)
            if "binary_flag" in f.metadata
        }

    def __repr__(self) -> str:
        """Return a detailed string representation of the Gadget2_Header.

        Provides a verbose summary of all fields in the header, useful for debugging and detailed inspections.

        Returns
        -------
        str
            A string representation of the Gadget2_Header instance.
        """
        header_details = []
        for fld in fields(self):
            field_value = getattr(self, fld.name)
            header_details.append(f"{fld.name}: {field_value}")
        return "Gadget2_Header(\n  " + "\n  ".join(header_details) + "\n)"

    def __str__(self) -> str:
        """Provide a one-line summary of the Gadget2_Header.

        Returns
        -------
        str
            A concise summary of the header, including key information such as the number of particle types and box size.
        """
        num_particles = np.sum(self.NumPart_Total)
        return f"Gadget2_Header: {num_particles} particles, BoxSize = {self.BoxSize}"

    def print_summary(self) -> None:
        """Print a nicely formatted table summarizing the header fields.

        This method attempts to use the `tabulate` library to output a table that details
        each field's properties, such as the field name and its value. If `tabulate` is not installed,
        it falls back to printing a less formatted summary using the `__repr__` method.

        Notes
        -----
        - The method gracefully handles the absence of the `tabulate` library, ensuring
          compatibility in environments where external dependencies cannot be installed.
        """
        try:
            from tabulate import tabulate

            # Prepare the data for tabulation
            headers = ["Field Name", "Value"]
            table_data = []

            for fld in fields(self):
                field_value = getattr(self, fld.name)
                table_data.append([fld.name, field_value])

            # Print the table using tabulate
            print(tabulate(table_data, headers=headers, tablefmt="grid"))

        except ImportError:
            print("Tabulate library not found. Falling back to default representation.")
            print(self.__repr__())

    @classmethod
    def _get_header_struct(cls: Type["Gadget2_Header"]) -> np.dtype:
        """Constructs the numpy struct to read the binary header data.

        Returns
        -------
        np.dtype
            The numpy dtype representing the structure of the binary header.

        Raises
        ------
        TypeError
            If the class definition is inconsistent with expected struct fields.
        ValueError
            If the struct size exceeds the allowed header size.
        """
        dtypes: dict[str, tuple[np.dtype, int]] = {}
        total_size: int = 0
        # Iterate over the fields and pull the necessary datatypes and names.
        for fld in fields(cls):
            try:
                dtype = fld.metadata["dtype"]
                binary_flag = fld.metadata["binary_flag"]
                size = fld.metadata.get("size", 1)
            except KeyError:
                raise TypeError(
                    f"Failed to load {fld.name} struct because class definition is inconsistent."
                )

            dtypes[binary_flag] = (dtype, size)
            total_size += np.dtype(dtype).itemsize * size

        if total_size > cls.HEADER_SIZE:
            raise ValueError(
                f"Struct has size {total_size} but header can only be {cls.HEADER_SIZE} bytes."
            )
        elif total_size < cls.HEADER_SIZE:
            dtypes["unused"] = (np.dtype(f"a{cls.HEADER_SIZE - total_size}"), 1)
        else:
            pass

        # Create the struct and return
        return np.dtype([(k, *v) if v[1] > 1 else (k, v[0]) for k, v in dtypes.items()])

    @classmethod
    def load_header(cls, path: str | Path, format: Literal["binary", "hdf5"]) -> Self:
        """Load the header of a Gadget-2 file from disk.

        This method reads the header information from a Gadget-2 file located at the specified path.

        The method supports both binary and HDF5 formats of Gadget-2 files. Depending on the format
        specified, it delegates the task to the appropriate private method (`_load_header_binary` or
        `_load_header_hdf5`) to handle the specific format's intricacies.

        Parameters
        ----------
        path : str or Path
            The path to the file from which the header is to be read. This should point to a valid
            Gadget-2 initial conditions file in either binary or HDF5 format.
        format : Literal["binary", "hdf5"]
            The format of the file. Must be either "binary" for binary Gadget-2 files or "hdf5" for
            HDF5 Gadget-2 files. This argument determines which underlying method is called to
            interpret the file format correctly.

        Returns
        -------
        Gadget2_Header
            An instance of `Gadget2_Header` containing the metadata read from the file header. This
            object can then be used to access various properties of the simulation or further manipulate
            the file.

        Raises
        ------
        ValueError
            If the specified format is neither "binary" nor "hdf5". This ensures that only supported
            formats are processed and helps prevent errors caused by incorrect format specifications.

        Notes
        -----
        The `load_header` method is a high-level interface for reading Gadget-2 headers, simplifying the
        process by abstracting the format-specific details away from the user. Internally, it uses the
        appropriate method to handle the format and ensures that the data is read correctly into a
        structured format that the rest of the codebase can use effectively.

        Example
        -------

        .. code-block:: python

            >>> header = Gadget2_Header.load_header("simulation_file.g2", format="binary")
            >>> print(header.BoxSize)
            100000.0
        """
        path = Path(path)
        mylog.info("Loading HEADER from %s (format=%s)", path, format)
        devlog.debug("Attempting to load HEADER from %s with format %s", path, format)

        if format == "binary":
            return cls._load_header_binary(path)
        elif format == "hdf5":
            return cls._load_header_hdf5(path)
        else:
            raise ValueError(f"Arg `format` must be binary or hdf5, not {format}.")

    def write_header(
        self,
        path: str | Path,
        format: Literal["binary", "hdf5"],
        overwrite: bool = False,
        **kwargs,
    ):
        """Write the header of a Gadget-2 file to disk. This method writes the header
        information, which contains critical metadata about the simulation (such as the
        number of particles, cosmological parameters, box size, and various flags), to a
        Gadget-2 file. The header can be written in either binary or HDF5 format,
        depending on the specified file format.

        Parameters
        ----------
        path : str or Path
            The file path to which the header is to be written. This should point to a valid location
            where a Gadget-2 initial conditions file in either binary or HDF5 format is or will be stored.
        format : Literal["binary", "hdf5"]
            The format of the file to be written. Can be either "binary" for binary Gadget-2 files or
            "hdf5" for HDF5 Gadget-2 files. This argument determines which underlying method is called
            to handle the specific format's intricacies.
        overwrite : bool, optional
            If set to True, the method will overwrite the existing file at the specified path, if any.
            If False (the default), an error will be raised if the file already exists.
        **kwargs : dict, optional
            Additional keyword arguments that can be passed to handle specific writing configurations
            or options. Currently, only ``length_unit`` is used and determines how the ``BoxSize`` parameter is
            coverted (if at all).

        Raises
        ------
        ValueError
            If the file already exists and `overwrite` is set to False.
            If the `format` argument is neither "binary" nor "hdf5".

        Notes
        -----
        - This method provides a high-level interface for writing header data, abstracting the complexities
          of format-specific details from the user.
        - The method determines the correct procedure based on the format and ensures that the header is
          written in a way that adheres to the Gadget-2 file specifications.

        Example
        -------

        .. code-block:: python

            >>> header = Gadget2_Header(...)
            >>> header.write_header("simulation_file.g2", format="binary", overwrite=True)
        """
        path = Path(path)
        mylog.info("Writing HEADER to %s (format=%s)", path, format)
        devlog.debug(
            "Preparing to write HEADER to %s (format=%s, overwrite=%s)",
            path,
            format,
            overwrite,
        )

        # Setup the path, generate parents, delete if overwrite is true, etc.
        if not path.parents[0].exists():
            path.parents[0].mkdir(parents=True, exist_ok=True)

        if path.exists() and not overwrite:
            raise ValueError(f"{path} exists and overwrite = False.")
        elif path.exists() and overwrite:
            path.unlink()  # Remove the file.
        else:
            pass

        # Generate the header values in a ready-to-write dict.
        length_unit = kwargs.pop("length_unit", None)
        header_dict = self._get_field_values(length_unit=length_unit, format=format)

        # Generate
        if format == "binary":
            self._write_header_binary(path, header_dict)
        elif format == "hdf5":
            self._write_header_hdf5(path, header_dict)
        else:
            raise ValueError(f"Arg `format` must be binary or hdf5, not {format}.")

        mylog.info("Header successfully written to %s (format=%s)", path, format)

    @classmethod
    def _load_header_binary(cls, path: Path) -> "Gadget2_Header":
        """Load the header from a binary format.

        Parameters
        ----------
        path : Path
            The path to the binary file from which to load the header.

        Returns
        -------
        Gadget2_Header
            The header loaded from the binary file.
        """
        # Fetch the header dtype and use it to read the binary struct
        header_dtype = cls._get_header_struct()
        _header_dict: dict[str, Any] = {}

        with open(path, "rb") as f:
            # We need to read the header chars first, then proceed to the actual header
            block_name = read_block_specifier(f)
            assert (
                block_name == "HEAD"
            ), f"Encountered block {block_name}, expected HEAD."

            # Now read the actual header.
            try:
                header = read_record(f, header_dtype)
            except (EOFError, IOError) as e:
                raise IOError(f"Failed to read header of {path}: {e.__str__()}.")

        # Process the header data to populate the header dictionary
        for hkey in header_dtype.names:
            if hkey == "unused":
                continue
            _header_dict[cls.BINARY_TO_HDF5[hkey]] = header[hkey][0]
            devlog.debug(
                "\t [HEADER] - %s = %s.",
                cls.BINARY_TO_HDF5[hkey],
                _header_dict[cls.BINARY_TO_HDF5[hkey]],
            )

        return cls(**_header_dict)

    @classmethod
    def _load_header_hdf5(cls, path: Path) -> "Gadget2_Header":
        """Fetch the head from the hdf5 file.

        Parameters
        ----------
        path : Path
            The path to the HDF5 file from which to load the header.

        Returns
        -------
        Gadget2_Header
            The header loaded from the HDF5 file.
        """
        _header_dict: dict[str, Any] = {}

        with h5py.File(path, "r") as f:
            for k in f["Header"].attrs.keys():
                _header_dict[k] = f["Header"].attrs[k]
                devlog.debug("\t [HEADER] - %s = %s.", k, _header_dict[k])

        return cls(**_header_dict)

    def _get_field_values(
        self, length_unit: str = None, format: str = "binary"
    ) -> dict[str, Any]:
        """Prepare header values for writing.

        Parameters
        ----------
        length_unit : str, optional
            The unit of length to use when writing the header.
        format : str, optional
            The format of the file, either ``binary`` or ``hdf5``.

        Returns
        -------
        dict[str, Any]
            The dictionary of header field values ready for writing.
        """
        # Setting the default length
        if not length_unit:
            length_unit = "kpc"

        # Prepare a dictionary to write.
        header_dict: dict[str, Any] = {}
        for fld in fields(self.__class__):
            value = self.__getattribute__(fld.name)
            dtype = fld.metadata["dtype"]
            size = fld.metadata["size"]
            if fld.name == "BoxSize":
                # We need to deal with boxsize special because it may have units or may not. Either way,
                # we convert and then write.
                if isinstance(value, unyt.unyt_quantity):
                    value = value.to_value(length_unit)

            header_dict[fld.name] = value

            # Enforce the size and dtype.
            if size > 1:
                try:
                    header_dict[fld.name] = np.reshape(header_dict[fld.name], (size,))
                except Exception:
                    raise ValueError(
                        f"Field {fld.name} could not be coerced to have shape ({size},)."
                    )

            header_dict[fld.name] = np.asarray(header_dict[fld.name], dtype=dtype)

        # Ensuring that the labels are correct
        if format == "hdf5":
            return header_dict
        elif format == "binary":
            return {self.__class__.HDF5_TO_BINARY[k]: v for k, v in header_dict.items()}
        else:
            raise ValueError(f"Format {format} not supported.")

    @staticmethod
    def _write_header_hdf5(path: Path, header_dict: dict[str, Any]):
        """Writes the header to an HDF5 file.

        Parameters
        ----------
        path : Path
            The path to the HDF5 file.
        header_dict : dict[str, Any]
            The dictionary of header values to write.
        """
        with h5py.File(path, "w") as fio:
            fio.create_group("Header")
            for k, v in header_dict.items():
                fio["Header"].attrs[k] = v

    def _write_header_binary(self, path: Path, header_dict: dict[str, Any]):
        """Writes the header to a binary file.

        Parameters
        ----------
        path : Path
            The path to the binary file.
        header_dict : dict[str, Any]
            The dictionary of header values to write.
        """
        header_dtype = self._get_header_struct()

        header_array = np.zeros(1, dtype=header_dtype)
        for field_name in header_dtype.names:
            if field_name == "unused":
                continue
            header_array[field_name][0] = header_dict[field_name]

        # Open the file in read/write mode ('rb+'), so we don't truncate the file
        with open(path, "wb+") as f:
            f.seek(0)  # Ensure we are at the start of the file

            # Write the `HEAD` block as a separate record
            head_size = 4  # `HEAD` is 4 bytes
            f.write(np.array(head_size, dtype=np.int32).tobytes())  # Write start marker
            f.write(b"HEAD")  # Write `HEAD` block name
            f.write(np.array(head_size, dtype=np.int32).tobytes())  # Write end marker

            # Write the actual header data as a separate record
            f.write(
                np.array(self.__class__.HEADER_SIZE, dtype=np.int32).tobytes()
            )  # Write start marker
            f.write(header_array.tobytes())  # Write header data
            f.write(
                np.array(self.__class__.HEADER_SIZE, dtype=np.int32).tobytes()
            )  # Write end marker

    @classmethod
    def from_particles(
        cls: Type["Gadget2_Header"],
        particles: ClusterParticles,
        particle_registry: "Gadget2_ParticleRegistry",
        cosmology: "Cosmology" = None,
        boxsize_buffer: float = 0.25,
        **kwargs,
    ) -> "Gadget2_Header":
        """Create a `Gadget2_Header` instance from particle data and a particle
        registry.

        This method constructs a `Gadget2_Header` by analyzing particle data provided in a
        `ClusterParticles` instance. It uses the particle registry to map particle types and,
        optionally, cosmological information to populate the header fields. The method can
        also calculate the simulation box size based on the particle positions.

        Parameters
        ----------
        particles : ClusterParticles
            An instance of `ClusterParticles` containing the particle data. This includes
            positions, velocities, masses, and other properties of each particle type that
            are relevant for setting up the simulation header.
        particle_registry : Gadget2_ParticleRegistry
            An instance of `Gadget2_ParticleRegistry` that maps Gadget particle types to
            internal types. It is used to identify the types of particles present and how
            they should be counted and managed in the header.
        cosmology : yt.utilities.cosmology.Cosmology, optional
            An optional `Cosmology` object that provides cosmological parameters like
            matter density (`Omega0`), dark energy density (`OmegaLambda`), and the Hubble
            constant (`HubbleParam`). These parameters are necessary if the simulation
            is cosmological.
        boxsize_buffer : float, optional
            A buffer factor to expand the calculated simulation box size beyond the minimum
            bounding box of the particle positions. This buffer is a fraction (default is 0.25)
            that increases the box size to accommodate any boundary conditions or to avoid
            particles being too close to the edges.
        **kwargs : dict, optional
            Additional keyword arguments that can be passed to the `Gadget2_Header` constructor.
            These may include specific header fields or overrides for default values.

        Returns
        -------
        Gadget2_Header
            A new instance of `Gadget2_Header` populated with data derived from the particle
            information, registry mappings, and any cosmological parameters provided.

        Notes
        -----
        - The particle counts (`NumPart_ThisFile` and `NumPart_Total`) are automatically
          computed based on the particle data. If masses are assigned per particle type,
          the counts are updated accordingly.
        - If a `BoxSize` is not explicitly provided in `kwargs`, it is computed as the
          smallest bounding box that contains all particles, expanded by the `boxsize_buffer`.
        - Cosmological parameters are only set if a `Cosmology` instance is provided. If not,
          these fields remain at their default values.
        - This method is typically used in preparing the header for a new simulation or for
          converting between different file formats while retaining accurate metadata.

        Example
        -------

        .. code-block:: python

            # Assuming `particles` is an instance of ClusterParticles with relevant data
            # and `particle_registry` is an instance of Gadget2_ParticleRegistry:

            header = Gadget2_Header.from_particles(
                particles,
                particle_registry,
                cosmology=my_cosmology,
                boxsize_buffer=0.3
            )
            print(header.BoxSize)  # Output the calculated box size with buffer
        """
        # Initialize particle counts to zero
        num_part_this_file = np.zeros(cls.N_PARTICLE_TYPES, dtype=cls.HEADER_UINT_TYPE)
        num_part_total = np.zeros(cls.N_PARTICLE_TYPES, dtype=cls.HEADER_INT_TYPE)

        # Populate NumPart_ThisFile and NumPart_Total
        for ptype, pname in particle_registry.map.items():
            if pname and (pname, "particle_mass") in particles.fields:
                # ! We always assign masses to each particle. If that changes, this needs to
                # ! change to reflect it because the particle_mass field may be absent.
                num_part_this_file[ptype] = particles[pname, "particle_mass"].size
                num_part_total[ptype] = num_part_this_file[ptype]

        if "BoxSize" not in kwargs:
            # Calculate the BoxSize based on particle positions with a 0.25 buffer
            min_coords = np.full(3, np.inf)
            max_coords = np.full(3, -np.inf)

            # Iterate over all particle types and find the extents
            for ptype in particles.particle_types:
                if (ptype, "particle_position") in particles.fields:
                    positions = particles.fields[(ptype, "particle_position")].to_value(
                        "kpc"
                    )
                    min_coords = np.minimum(min_coords, positions.min(axis=0))
                    max_coords = np.maximum(max_coords, positions.max(axis=0))

            # Calculate the extent of the box and add buffer space
            extent = max_coords - min_coords
            kwargs["BoxSize"] = unyt.unyt_quantity(
                np.max(extent) * (1 + (2 * boxsize_buffer)), "kpc"
            )
        else:
            pass

        # Prepare header kwargs
        header_kwargs = kwargs.copy()
        header_kwargs.update(
            {
                "NumPart_ThisFile": num_part_this_file,
                "NumPart_Total": num_part_total,
            }
        )

        # If cosmology is provided, set cosmological parameters
        if cosmology:
            header_kwargs.update(
                {
                    "Omega0": cosmology.omega_matter,
                    "OmegaLambda": cosmology.omega_lambda,
                    "HubbleParam": cosmology.hubble_constant
                    / 100.0,  # convert H0 to 100 km/s/Mpc units
                }
            )

        return cls(**header_kwargs)


Gadget2_Header.BINARY_TO_HDF5 = Gadget2_Header.generate_field_mapping()
Gadget2_Header.HDF5_TO_BINARY = {v: k for k, v in Gadget2_Header.BINARY_TO_HDF5.items()}


class Gadget2_ParticleRegistry:
    """Registry for managing particle types in Gadget-2 files.

    This class provides a mapping between Gadget-2 particle type IDs (0-5) and their
    corresponding names in the cluster generator context. It allows for easy retrieval
    and updating of particle type information, supporting operations such as adding new
    types, removing existing ones, and retrieving particle type names or IDs.
    """

    DEFAULT_PARTICLE_MAP: dict[int, str | None] = _GADGET_PARTICLES.copy()
    """Dict[int, str | None]: Default mapping of Gadget-2 particle types to their
    corresponding names.

    This map supports up to 6 particle types (IDs 0 through 5).
    """

    def __init__(self, particle_map: dict[int, str]):
        """Initializes the particle registry for Gadget-2 files.

        Parameters
        ----------
        particle_map : dict[int, str]
            Mapping of Gadget-2 particle types to cluster generator equivalent particle types.

        Raises
        ------
        AssertionError
            If the length of the particle map exceeds 6, since Gadget-2 format supports only 6 particle species.
        """
        self._map: dict[int, str | None] = self.__class__.DEFAULT_PARTICLE_MAP.copy()

        for k, v in particle_map.items():
            self._map[k] = v

        for i in range(6):
            if i not in self._map:
                self._map[i] = None

        assert (
            len(self._map) == 6
        ), "Gadget style particle files only support 6 particle species."

    def __repr__(self) -> str:
        """Return a string representation of the particle registry.

        Returns
        -------
        str
            String representation of the particle registry.
        """
        return f"Gadget2_ParticleRegistry({self._map})"

    def __getitem__(self, particle_id: int) -> str:
        """Enable dictionary-like access to particle names by ID.

        Parameters
        ----------
        particle_id : int
            The particle type ID (0-5) in the Gadget-2 format.

        Returns
        -------
        str
            The name of the particle type.

        Raises
        ------
        ValueError
            If the particle ID is not recognized.
        """
        return self.get_particle_name(particle_id)

    def __setitem__(self, particle_id: int, particle_name: str) -> None:
        """Enable dictionary-like assignment of particle names by ID.

        Parameters
        ----------
        particle_id : int
            The particle type ID (0-5) in the Gadget-2 format.
        particle_name : str
            The name of the particle type to assign.
        """
        self.update_particle_map(particle_id, particle_name)

    def __delitem__(self, particle_id: int) -> None:
        """Enable dictionary-like deletion of particle names by ID.

        Parameters
        ----------
        particle_id : int
            The particle type ID (0-5) in the Gadget-2 format.
        """
        self.remove_particle(particle_id)

    def __len__(self) -> int:
        """Return the number of particle types currently in the registry.

        Returns
        -------
        int
            The number of particle types.
        """
        return len([ptype for ptype in self._map.values() if ptype is not None])

    def __contains__(self, particle_name: str) -> bool:
        """Check if a particle name is present in the registry.

        Parameters
        ----------
        particle_name : str
            The name of the particle type to check.

        Returns
        -------
        bool
            True if the particle name is in the registry, False otherwise.
        """
        return particle_name in self.rmap

    @property
    def map(self) -> dict[int, str | None]:
        """Returns the internal particle mapping.

        Returns
        -------
        dict[int, str | None]
            The internal mapping of particle types.
        """
        return self._map

    @property
    def rmap(self) -> dict[str, int]:
        """Returns the reverse particle mapping.

        Returns
        -------
        dict[str, int]
            The reverse mapping of particle types.
        """
        return {v: k for k, v in self._map.items() if v is not None}

    def get_particle_name(self, particle_id: int) -> str:
        """Get the particle name corresponding to the provided particle ID.

        Parameters
        ----------
        particle_id : int
            The particle type ID (0-5) in the Gadget-2 format.

        Returns
        -------
        str
            The name of the particle type.

        Raises
        ------
        ValueError
            If the particle ID is not recognized.
        """
        if particle_id not in self._map:
            raise ValueError(
                f"Particle ID {particle_id} is not recognized. Valid IDs are 0 to 5."
            )
        return self._map[particle_id]

    def get_particle_id(self, particle_name: str) -> int:
        """Get the particle ID corresponding to the provided particle name.

        Parameters
        ----------
        particle_name : str
            The name of the particle type.

        Returns
        -------
        int
            The particle type ID (0-5) in the Gadget-2 format.

        Raises
        ------
        ValueError
            If the particle name is not recognized.
        """
        if particle_name not in self.rmap:
            raise ValueError(
                f"Particle name '{particle_name}' is not recognized. Available names are: {list(self.rmap.keys())}."
            )
        return self.rmap[particle_name]

    def update_particle_map(self, particle_id: int, particle_name: str) -> None:
        """Update the particle registry with a new particle type name.

        Parameters
        ----------
        particle_id : int
            The particle type ID (0-5) in the Gadget-2 format.
        particle_name : str
            The name of the particle type to assign.

        Raises
        ------
        ValueError
            If the particle ID is out of the valid range (0-5).
        """
        if particle_id < 0 or particle_id > 5:
            raise ValueError("Particle ID must be between 0 and 5.")

        self._map[particle_id] = particle_name

    def remove_particle(self, particle_id: int) -> None:
        """Remove a particle type from the registry by ID.

        Parameters
        ----------
        particle_id : int
            The particle type ID (0-5) to remove from the registry.

        Raises
        ------
        ValueError
            If the particle ID is not recognized.
        """
        if particle_id not in self._map:
            raise ValueError(
                f"Particle ID {particle_id} is not recognized. Valid IDs are 0 to 5."
            )

        self._map[particle_id] = None

    def list_particles(self) -> dict[int, str]:
        """List all particle types in the registry.

        Returns
        -------
        dict[int, str]
            A dictionary mapping particle type IDs (0-5) to their names.
        """
        return self._map.copy()


class Gadget2_FieldRegistry:
    """Registry for managing field properties in Gadget-2 files.

    This class maintains a mapping of field names to their properties, such as cluster generator
    style names, binary format names, units, data types, particle IDs, and dimensions. It facilitates
    operations like adding, removing, updating, and retrieving field properties, providing an interface
    for managing data fields in Gadget-2 files.

    Notes
    -----

    **Mapping Structure and Field Properties**

    The `Gadget2_FieldRegistry` class relies on an `OrderedDict` to maintain the mapping of field names to their properties.
    Each field in this mapping is associated with a dictionary that contains several key attributes:

    - `binary_name`: The name of the field as it appears in the binary file.
    - `cluster_name`: The equivalent name of the field in the cluster generator style.
    - `units`: The units associated with the field, represented as a string or `unyt.Unit` object. If no units are specified, this value can be `None`.
    - `dtype`: The data type of the field, specified as a string or `np.dtype`.
    - `pid`: The list of particle IDs that the field applies to. If `None`, the field applies to all particle types.
    - `size`: The dimensionality of the field per particle (e.g., 1 for scalar fields, 3 for vector fields).

    **Importance of Ordering for Binary File Types**

    The use of an `OrderedDict` is critical for maintaining the correct order of fields, especially when dealing with binary Gadget-2 files.
    In binary formats, the order of fields in the file must match the order expected by the reading/writing routines to ensure data integrity.
    If fields are out of order, this can lead to misinterpretation of the binary data, potentially causing errors or corrupting the file.

    .. hint::

        If you're writing a file to the binary gadget-2 format, this really really matters. Ensure that your field registry
        is ordered the same way the software expects the read the data. If the blocks are not in the correct order, you will
        encounter IO errors.

    **Conventions for Missing Fields**

    - If a field is not present in the registry but is found in the file, it will be ignored but a warning is raised.
    - When writing, only fields in the registry are written back to disk.

    **Handling Custom Field Configurations**

    Users can extend the field registry to include additional fields or modify existing ones to fit specific needs.
    This flexibility allows the `Gadget2_FieldRegistry` to accommodate custom Gadget-2 formats or simulation-specific
     requirements without altering the core class structure.
    """

    DEFAULT_FIELD_MAP: OrderedDict[str, dict[str, Any]] = OrderedDict(
        {
            key: {
                "cluster_name": value[0],
                "binary_name": value[1],
                "units": value[2],
                "dtype": value[3],
                "pid": value[4],
                "size": value[5],
            }
            for key, value in _GADGET_FIELDS.items()
        }
    )
    """OrderedDict[str, dict[str, Any]]: Default field properties for Gadget-2 files."""

    def __init__(self, field_map: dict[str, dict[str, Any]]):
        """Initializes the field registry for Gadget-2 style files.

        Parameters
        ----------
        field_map : OrderedDict[str, dict[str, Any]]
            A dictionary mapping field names to their properties. Each field name maps to a dictionary
            containing the following keys:
            - ``binary_name``: ``str``, the field name in binary format.
            - ``cluster_name``: ``str``, the field name in cluster generator style.
            - ``units``: ``str``, ``unyt.Unit``, or ``None``, the units of the field.
            - ``dtype``: ``str`` or ``np.dtype``, the data type of the field.
            - ``pid``: ``list[int]`` or None, the particle IDs for which this field is expected.
            - ``size``: ``int``, the number of dimensions this field should have per particle. Vectors should have 3, scalars 1.

        Notes
        -----
        Only fields not present in :py:attr:`Gadget2_FieldRegistry.DEFAULT_FIELD_MAP` need to be specified. If a
        field is in both the default map and the ``field_map``, the ``field_map`` takes precedence so that
        the user may override defaults without having to redesign the class.
        """
        # Setting the map. We first use the default field map to set the defaults, then update with the new field map.
        self._map: OrderedDict[str, dict[str, Any]] = self.__class__.DEFAULT_FIELD_MAP

        for k, v in field_map.items():
            if k not in self._map:
                self._map[k] = v
            else:
                # update only specific items.
                for _kk, _vv in v.items():
                    self._map[k][_kk] = _vv

        # Validate the field map
        for field_name, properties in self._map.items():
            self._validate_field_properties(field_name, properties)

    @property
    def HDF_CG(self) -> dict[str, str]:
        """Mapping of HDF5 field names to cluster generator names.

        Returns
        -------
        dict[str, str]
            A dictionary mapping HDF5 field names to cluster generator names.
        """
        return {k: v["cluster_name"] for k, v in self._map.items()}

    @property
    def HDF_BINARY(self) -> dict[str, str]:
        """Mapping of HDF5 field names to binary names.

        Returns
        -------
        dict[str, str]
            A dictionary mapping HDF5 field names to binary names.
        """
        return {k: v["binary_name"] for k, v in self._map.items()}

    @property
    def CG_BINARY(self) -> dict[str, str]:
        """Mapping of cluster generator names to binary names.

        Returns
        -------
        dict[str, str]
            A dictionary mapping cluster generator names to binary names.
        """
        return {v["cluster_name"]: v["binary_name"] for _, v in self._map.items()}

    @property
    def CG_HDF(self) -> dict[str, str]:
        """Reverse mapping of HDF5 field names to cluster generator names.

        Returns
        -------
        dict[str, str]
            A dictionary mapping cluster generator names to HDF5 field names.
        """
        return reverse_dict(self.HDF_CG)

    @property
    def BINARY_HDF(self) -> dict[str, str]:
        """Reverse mapping of HDF5 field names to binary names.

        Returns
        -------
        dict[str, str]
            A dictionary mapping binary names to HDF5 field names.
        """
        return reverse_dict(self.HDF_BINARY)

    @property
    def BINARY_CG(self) -> dict[str, str]:
        """Reverse mapping of cluster generator names to binary names.

        Returns
        -------
        dict[str, str]
            A dictionary mapping binary names to cluster generator names.
        """
        return reverse_dict(self.CG_BINARY)

    @staticmethod
    def _validate_field_properties(fieldname: str, properties: dict[str, Any]) -> None:
        """Validates the properties of a field.

        Parameters
        ----------
        fieldname : str
            The name of the field to validate.
        properties : dict[str, Any]
            The properties of the field to validate.

        Raises
        ------
        ValueError
            If any required property is missing or invalid.
        """
        required_keys = {"binary_name", "cluster_name", "units", "dtype", "pid", "size"}
        missing_keys = required_keys - properties.keys()
        if missing_keys:
            raise ValueError(
                f"Field '{fieldname}' is missing required properties: {missing_keys}"
            )

    def get_binary_name(self, field_name: str) -> str:
        """Get the binary field name corresponding to the provided field name.

        Parameters
        ----------
        field_name : str
            The name of the field in any recognized format.

        Returns
        -------
        str
            The binary field name.

        See Also
        --------
        get_cluster_name : Get the cluster generator style name corresponding to the provided field name.
        """
        return self._map[field_name]["binary_name"]

    def get_cluster_name(self, field_name: str) -> str:
        """Get the cluster generator style name corresponding to the provided field
        name.

        Parameters
        ----------
        field_name : str
            The name of the field in any recognized format.

        Returns
        -------
        str
            The cluster generator style name.

        See Also
        --------
        get_binary_name : Get the binary field name corresponding to the provided field name.
        """
        return self._map[field_name]["cluster_name"]

    def get_units(self, field_name: str) -> str:
        """Get the units of the provided field.

        Parameters
        ----------
        field_name : str
            The name of the field in any recognized format.

        Returns
        -------
        str
            The units of the field.
        """
        return self._map[field_name]["units"]

    def get_dtype(self, field_name: str) -> np.dtype:
        """Get the dtype of the provided field.

        Parameters
        ----------
        field_name : str
            The name of the field in any recognized format.

        Returns
        -------
        np.dtype
            The dtype of the field.
        """
        return np.dtype(self._map[field_name]["dtype"])

    def get_pid(self, field_name: str) -> list[int]:
        """Get the particle IDs associated with the provided field.

        Parameters
        ----------
        field_name : str
            The name of the field in any recognized format.

        Returns
        -------
        list[int]
            The list of particle IDs for the field, or a list of all particle IDs if not specified.
        """
        return (
            self._map[field_name]["pid"]
            if self._map[field_name]["pid"] is not None
            else list(range(6))
        )

    def get_size(self, field_name: str) -> int:
        """Get the size (number of dimensions per particle) of the provided field.

        Parameters
        ----------
        field_name : str
            The name of the field in any recognized format.

        Returns
        -------
        int
            The size of the field (number of dimensions per particle).
        """
        return self._map[field_name]["size"]

    def update_field(self, field_name: str, properties: dict[str, Any]) -> None:
        """Update the properties of a field in the registry.

        Parameters
        ----------
        field_name : str
            The name of the field to update.
        properties : dict[str, Any]
            The updated properties for the field.

        Raises
        ------
        ValueError
            If the field properties are not valid.
        """
        self._validate_field_properties(field_name, properties)

        if field_name not in self._map:
            self._map[field_name] = properties
        else:
            self._map[field_name].update(properties)

    def remove_field(self, field_name: str) -> None:
        """Remove a field from the registry.

        Parameters
        ----------
        field_name : str
            The name of the field to remove.

        Raises
        ------
        ValueError
            If the field is not found in the registry.
        """
        if field_name not in self._map:
            raise ValueError(f"Field '{field_name}' not found in registry.")

        _ = self._map.pop(field_name)

    def list_fields(self) -> dict[str, dict[str, Any]]:
        """List all fields in the registry with their properties.

        Returns
        -------
        dict[str, dict[str, Any]]
            A dictionary mapping field names to their properties.
        """
        return self._map.copy()

    def get_block_dtype(self, hdf5_field_id: str) -> np.dtype:
        """Constructs the correct numpy dtype for a given HDF5 field ID.

        Parameters
        ----------
        hdf5_field_id : str
            The HDF5 field identifier for which to construct the dtype.

        Returns
        -------
        np.dtype
            The numpy dtype representing the structure of the block for the given field.

        Raises
        ------
        ValueError
            If the field is not found in the registry.
        """
        if hdf5_field_id not in self._map:
            raise ValueError(f"Field '{hdf5_field_id}' not found in registry.")

        field_info = self._map[hdf5_field_id]
        dtype = field_info["dtype"]
        size = field_info["size"]

        # Determine the shape based on the size (e.g., vectors have size 3, scalars have size 1)
        if size > 1:
            block_dtype = np.dtype([self.HDF_BINARY[hdf5_field_id], dtype, size])
        else:
            block_dtype = np.dtype([self.HDF_BINARY[hdf5_field_id], dtype])

        return block_dtype

    def __repr__(self) -> str:
        """Return a formatted string representation of the field registry for
        readability.

        Returns
        -------
        str
            Formatted string representation of the field registry, showing field names
            and their associated properties in a tabular format.
        """
        # Define column widths for a more readable tabular format
        col_widths = {
            "Field": 15,
            "Cluster Name": 15,
            "Binary Name": 15,
            "Units": 10,
            "Data Type": 10,
            "PID": 15,
            "Size": 5,
        }

        # Header for the output table
        header = (
            f"{'Field':<{col_widths['Field']}} "
            f"{'Cluster Name':<{col_widths['Cluster Name']}} "
            f"{'Binary Name':<{col_widths['Binary Name']}} "
            f"{'Units':<{col_widths['Units']}} "
            f"{'Data Type':<{col_widths['Data Type']}} "
            f"{'PID':<{col_widths['PID']}} "
            f"{'Size':<{col_widths['Size']}}"
        )

        # Separator line for the table
        separator = "-" * len(header)

        # Rows containing the field data
        rows = []
        for fld, props in self._map.items():
            row = (
                f"{fld:<{col_widths['Field']}} "
                f"{props['cluster_name']:<{col_widths['Cluster Name']}} "
                f"{props['binary_name']:<{col_widths['Binary Name']}} "
                f"{str(props['units']):<{col_widths['Units']}} "
                f"{str(props['dtype']):<{col_widths['Data Type']}} "
                f"{str(props['pid']):<{col_widths['PID']}} "
                f"{str(props['size']):<{col_widths['Size']}}"
            )
            rows.append(row)

        # Combine everything into a single formatted string
        return "\n".join([header, separator] + rows)

    def __str__(self) -> str:
        """Return a concise, one-line string representation of the field registry.

        Returns
        -------
        str
            A one-line summary indicating the number of fields managed by the registry
            and a brief overview of the field properties.
        """
        num_fields = len(self._map)
        field_names = ", ".join(list(self._map.keys())[:3])  # Show up to 3 field names
        if num_fields > 3:
            field_names += ", ..."

        return f"Gadget2_FieldRegistry with {num_fields} fields: [{field_names}]"

    def __getitem__(self, field_name: str) -> dict[str, Any]:
        """Enable dictionary-like access to field properties by field name.

        Parameters
        ----------
        field_name : str
            The name of the field.

        Returns
        -------
        dict[str, Any]
            The properties of the field.

        Raises
        ------
        KeyError
            If the field is not found in the registry.
        """
        if field_name not in self._map:
            raise KeyError(f"Field '{field_name}' not found in registry.")
        return self._map[field_name]

    def __setitem__(self, field_name: str, properties: dict[str, Any]) -> None:
        """Enable dictionary-like assignment of field properties by field name.

        Parameters
        ----------
        field_name : str
            The name of the field.
        properties : dict[str, Any]
            The properties to assign to the field.

        Raises
        ------
        ValueError
            If the field properties are not valid.
        """
        self.update_field(field_name, properties)

    def __delitem__(self, field_name: str) -> None:
        """Enable dictionary-like deletion of fields by field name.

        Parameters
        ----------
        field_name : str
            The name of the field.

        Raises
        ------
        ValueError
            If the field is not found in the registry.
        """
        self.remove_field(field_name)

    def __len__(self) -> int:
        """Return the number of fields currently in the registry.

        Returns
        -------
        int
            The number of fields.
        """
        return len(self._map)

    def __contains__(self, field_name: str) -> bool:
        """Check if a field name is present in the registry.

        Parameters
        ----------
        field_name : str
            The name of the field to check.

        Returns
        -------
        bool
            True if the field name is in the registry, False otherwise.
        """
        return field_name in self._map

    def print_summary(self) -> None:
        """Print a nicely formatted table summarizing the fields in the registry.

        This method attempts to use the `tabulate` library to output a table that details
        each field's properties, such as the field name, cluster name, binary name, units,
        data type, particle IDs, and size. If `tabulate` is not installed, it falls back
        to printing a less formatted summary using the `__repr__` method.

        Notes
        -----
        - The method gracefully handles the absence of the `tabulate` library, ensuring
          compatibility in environments where external dependencies cannot be installed.
        """
        try:
            from tabulate import tabulate

            # Prepare the data for tabulation
            headers = [
                "Field Name",
                "Cluster Name",
                "Binary Name",
                "Units",
                "Data Type",
                "Particle IDs",
                "Size",
            ]
            table_data = []

            for field_name, properties in self._map.items():
                table_data.append(
                    [
                        field_name,
                        properties["cluster_name"],
                        properties["binary_name"],
                        properties["units"]
                        if properties["units"] is not None
                        else "None",
                        properties["dtype"],
                        properties["pid"] if properties["pid"] is not None else "All",
                        properties["size"],
                    ]
                )

            # Print the table using tabulate
            print(tabulate(table_data, headers=headers, tablefmt="grid"))

        except ImportError:
            print("Tabulate library not found. Falling back to default representation.")
            print(self.__repr__())


class Gadget2_ICFile(ABC):
    """Abstract base class for handling Gadget-2 initial condition (IC) files.

    This class provides a framework for reading and writing Gadget-2 IC files in various formats (binary or HDF5).
    It manages the file path, header, particle data, and registries for fields and particles. Subclasses should implement
    format-specific behavior for reading and writing particle data.

    Notes
    -----
    **File Format and Abstract Methods**

    The `Gadget2_ICFile` class is an abstract base class designed to support various file formats used in Gadget-2 simulations,
    such as binary and HDF5. Subclasses should define the `_format` class variable to specify their format and implement the
    abstract methods `load_particles` and `_write_particle_data` to handle format-specific reading and writing of particle data.

    **Handling Registries and Headers**

    - The `particle_registry` and `field_registry` properties manage particle types and field properties, respectively.
      They provide a standardized way to access and modify the metadata required to interpret the simulation data correctly.
    - The `header` property provides access to the file's metadata, such as the number of particles, cosmological parameters,
      and simulation box size, ensuring that all essential information is available and can be modified if needed.

    **File Overwriting and Safety**

    - The `from_particles` method provides a mechanism for creating Gadget-2 files from particle data. It includes safety
      checks to prevent accidental overwriting of existing files unless explicitly allowed by the user.

    **Subclasses and Extensibility**

    - This class is designed to be subclassed for specific file formats. Subclasses should implement format-specific logic for
      reading and writing particle data while leveraging the base class's infrastructure for handling headers, registries,
      and other metadata.

    Examples
    --------
    Creating a Gadget-2 binary file from particle data:

    .. code-block:: python

        particles = ClusterParticles(...)
        path = "simulation_file.g2"
        Gadget2_Binary.from_particles(particles, path, overwrite=True)

    Loading particle data from an existing Gadget-2 file:

    .. code-block:: python

        ic_file = Gadget2_Binary("simulation_file.g2")
        particles = ic_file.particles
    """

    _format: Literal[
        "hdf5", "binary"
    ] = None  # Placeholder for the format to be defined by subclasses

    def __init__(
        self,
        path: Union[str, Path],
        particle_registry: Gadget2_ParticleRegistry = None,
        field_registry: Gadget2_FieldRegistry = None,
    ):
        """Initialize a Gadget2_ICFile instance.

        Parameters
        ----------
        path : str or Path
            The path to the Gadget-2 initial condition file.
        particle_registry : Gadget2_ParticleRegistry, optional
            An instance of Gadget2_ParticleRegistry to manage particle types.
        field_registry : Gadget2_FieldRegistry, optional
            An instance of Gadget2_FieldRegistry to manage field properties.
        """
        # Setup the path, force as Path() so that we can easily check existence.
        self.path: Path = Path(path)

        # Setting up the property base variables.
        self._header: Union[Gadget2_Header, None] = None
        self._particle_registry: Union[
            Gadget2_ParticleRegistry, None
        ] = particle_registry
        self._field_registry: Union[Gadget2_FieldRegistry, None] = field_registry
        self._particles: Union[ClusterParticles, None] = None

    def __len__(self) -> int:
        """Return the number of particles in the Gadget-2 file.

        Returns
        -------
        int
            The total number of particles across all types.
        """
        return np.sum(self.header.NumPart_ThisFile)

    def __str__(self) -> str:
        """Provide a user-friendly string representation of the Gadget2_ICFile object.

        Returns
        -------
        str
            A string summarizing the file path and format.
        """
        return f"Gadget2_ICFile(path='{self.path}')"

    def __repr__(self) -> str:
        """Provide a detailed string representation of the Gadget2_ICFile object for
        debugging.

        Returns
        -------
        str
            A detailed string representation of the object.
        """
        return (
            f"Gadget2_ICFile(path={self.path}, "
            f"particle_registry={repr(self._particle_registry)}, "
            f"field_registry={repr(self._field_registry)}, "
            f"header={repr(self._header)}, "
            f"particles={repr(self._particles)})"
        )

    @property
    def particles(self) -> ClusterParticles:
        """Retrieve the particle data from the Gadget-2 file.

        Returns
        -------
        ClusterParticles
            The particle data loaded from the file.

        Notes
        -----
        This property lazily loads the particle data if it hasn't been loaded already.
        """
        if self._particles is None:
            self.load_particles()
        return self._particles

    @property
    def header(self) -> Gadget2_Header:
        """Retrieve the header of the Gadget-2 file.

        Returns
        -------
        Gadget2_Header
            The header containing metadata about the Gadget-2 file.

        Notes
        -----
        This property lazily loads the header if it hasn't been loaded already.
        """
        if self._header is None:
            self._header = Gadget2_Header.load_header(self.path, self.__class__._format)
        return self._header

    def write_header(self, overwrite: bool = False, **kwargs):
        """Write the header to the Gadget-2 file.

        Parameters
        ----------
        overwrite : bool, optional
            If True, allows overwriting of the existing file. Default is False.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the header's `write_header` method.

        Raises
        ------
        ValueError
            If the file already exists and `overwrite` is set to False.

        Notes
        -----
        This method uses the format-specific write method defined in the header class.
        """
        self.header.write_header(self.path, self.__class__._format, overwrite, **kwargs)

    @property
    def particle_registry(self) -> Gadget2_ParticleRegistry:
        """Retrieve the particle registry.

        Returns
        -------
        Gadget2_ParticleRegistry
            The particle registry managing particle types.

        Notes
        -----
        This property initializes the registry if it hasn't been set already.
        """
        if self._particle_registry is None:
            self._particle_registry = Gadget2_ParticleRegistry({})
        return self._particle_registry

    @particle_registry.setter
    def particle_registry(self, value: Gadget2_ParticleRegistry):
        """Set the particle registry.

        Parameters
        ----------
        value : Gadget2_ParticleRegistry
            The new particle registry to assign.
        """
        self._particle_registry = value

    @property
    def field_registry(self) -> Gadget2_FieldRegistry:
        """Retrieve the field registry.

        Returns
        -------
        Gadget2_FieldRegistry
            The field registry managing field properties.

        Notes
        -----
        This property initializes the registry if it hasn't been set already.
        """
        if self._field_registry is None:
            self._field_registry = Gadget2_FieldRegistry({})
        return self._field_registry

    @field_registry.setter
    def field_registry(self, value: Gadget2_FieldRegistry):
        """Set the field registry.

        Parameters
        ----------
        value : Gadget2_FieldRegistry
            The new field registry to assign.
        """
        self._field_registry = value

    @abstractmethod
    def load_particles(self, inplace: bool = True):
        """Load particle data from the Gadget-2 file.

        Parameters
        ----------
        inplace : bool, optional
            If True, updates the instance's particle data directly. Default is True.

        Notes
        -----
        This method must be implemented by subclasses to handle format-specific particle loading.
        """
        pass

    def reload(self):
        """Reload the header and particle data from the Gadget-2 file.

        Notes
        -----
        This method resets the header and particle data, forcing a reload from the file.
        """
        self._header, self._particles = None, None
        _, _ = self.header, self.particles  # Forces reload

    @classmethod
    def from_particles(
        cls: Type["Gadget2_ICFile"],
        particles: ClusterParticles,
        path: Union[str, Path],
        field_registry: Gadget2_FieldRegistry = None,
        particle_registry: Gadget2_ParticleRegistry = None,
        header: Gadget2_Header = None,
        overwrite: bool = False,
        **kwargs,
    ):
        """Create a Gadget-2 file from particle data, registries, and header
        information.

        Parameters
        ----------
        particles : ClusterParticles
            The particle data to be written to the file.
        path : str or Path
            The file path where the Gadget-2 file will be created.
        field_registry : Gadget2_FieldRegistry, optional
            The field registry for managing field properties. Default is None.
        particle_registry : Gadget2_ParticleRegistry, optional
            The particle registry for managing particle types. Default is None.
        header : Gadget2_Header, optional
            The header containing metadata for the Gadget-2 file. Default is None.
        overwrite : bool, optional
            If True, allows overwriting of the existing file. Default is False.
        **kwargs : dict, optional
            Additional keyword arguments for header creation.

        Raises
        ------
        ValueError
            If the file already exists and `overwrite` is set to False.

        Notes
        -----
        This method handles the creation of a new Gadget-2 file, including writing the header and particle data.
        """
        # Enforcing overwriting rules
        path = Path(path)

        # setup the path, generate parents, delete if overwrite is true, etc.
        if not path.parents[0].exists():
            path.parents[0].mkdir(parents=True, exist_ok=True)

        if path.exists() and not overwrite:
            raise ValueError(f"{path} exists and overwrite = False.")
        elif path.exists() and overwrite:
            path.unlink()  # remove the file.
        else:
            pass

        # Setting the field / particle registry and the header.
        # field and particle registries just inherit the defaults, the header can be built
        # from the registries and particles.
        if field_registry is None:
            field_registry = Gadget2_FieldRegistry({})

        if particle_registry is None:
            particle_registry = Gadget2_ParticleRegistry({})

        if header is None:
            header = Gadget2_Header.from_particles(
                particles, particle_registry, **kwargs
            )

        # Write the header to file.
        header.write_header(path, format=cls._format, overwrite=overwrite)

        # writing the actual particle data.
        cls._write_particle_data(
            path, particles, field_registry, particle_registry, header
        )

    @classmethod
    @abstractmethod
    def _write_particle_data(
        cls,
        path: Union[str, Path],
        particles: ClusterParticles,
        field_registry: Gadget2_FieldRegistry,
        particle_registry: Gadget2_ParticleRegistry,
        header: Gadget2_Header,
    ):
        """Write particle data to a Gadget-2 file.

        Parameters
        ----------
        path : str or Path
            The file path where the Gadget-2 file is located.
        particles : ClusterParticles
            The particle data to be written.
        field_registry : Gadget2_FieldRegistry
            The field registry for managing field properties.
        particle_registry : Gadget2_ParticleRegistry
            The particle registry for managing particle types.
        header : Gadget2_Header
            The header containing metadata for the Gadget-2 file.

        Notes
        -----
        This method must be implemented by subclasses to handle format-specific particle writing.
        """
        pass

    def save(self, overwrite: bool = False, **kwargs):
        """Save both the header and particle data to the Gadget-2 file.

        Parameters
        ----------
        overwrite : bool, optional
            If True, allows overwriting of the existing file. Default is False.
        **kwargs : dict, optional
            Additional keyword arguments for saving.

        Raises
        ------
        ValueError
            If the file already exists and `overwrite` is set to False.
        """
        return self.from_particles(
            self.particles,
            self.path,
            self.field_registry,
            self.particle_registry,
            self.header,
            overwrite=overwrite,
            **kwargs,
        )


class Gadget2_Binary(Gadget2_ICFile):
    """Class for handling binary Gadget-2 initial condition files.

    This class provides methods to read and write particle data in the binary format
    used by Gadget-2, a common N-body and hydrodynamics code for cosmological
    simulations. It supports reading and writing particle data and headers, ensuring
    compatibility with Gadget-2's binary file format.

    Attributes
    ----------
    _format : str
        Specifies the file format handled by this class, which is "binary".

    See Also
    --------
    Gadget2_ICFile : Abstract base class for Gadget-2 file handling.
    Gadget2_HDF5 : Class for handling Gadget-2 initial condition files in HDF5 format.
    """

    _format = "binary"

    def load_particles(self, inplace: bool = True) -> "ClusterParticles":
        """Load particle data from a binary Gadget-2 file.

        Parameters
        ----------
        inplace : bool, optional
            If True, updates the instance's particle data directly. Default is True.

        Returns
        -------
        ClusterParticles
            The loaded particle data as a ClusterParticles object.

        Raises
        ------
        AssertionError
            If the binary file does not start with the 'HEAD' block.
        EOFError, IOError
            If the file ends unexpectedly or cannot be read properly.

        Examples
        --------

        .. code-block:: python

            >>> gadget_binary = Gadget2_Binary('path/to/gadget_file.g2')
            >>> particles = gadget_binary.load_particles()
            >>> print(np.sum(particles.num_particles.values()))  # Output: Number of particles loaded
        """
        mylog.info("Loading particle data from %s.", self)
        devlog.info("Loading particle data from %s.", self)
        fields = OrderedDict()

        with open(self.path, "rb") as f:
            # Skip the header
            head = read_block_specifier(f)
            assert head == "HEAD", f"Binary file {self.path} doesn't start with 'HEAD'."
            devlog.debug("\tLoading block HEAD (header)...")
            # If we haven't already read the header, this is a good opportunity to do so.
            # And we need the information to proceed with the reading process anyways.
            header_struct = Gadget2_Header._get_header_struct()

            if self._header is None:
                header_data = read_record(f, header_struct)
                header_dict = {
                    Gadget2_Header.BINARY_TO_HDF5[name]: header_data[name][0]
                    for name in header_struct.names
                    if name != "unused"
                }
                self._header = Gadget2_Header(**header_dict)
                devlog.debug(
                    "\t\tLoaded header data from binary file: %s", self._header
                )
            else:
                _ = skip_record(f)
                devlog.debug("\t\tSkipped header record as it is already loaded.")

            while True:
                try:
                    block_name = read_block_specifier(f)
                    if block_name not in self.field_registry.BINARY_HDF:
                        # Skip unrecognized block
                        record_size = skip_record(f)
                        mylog.warning(
                            "\tSkipped block %s (not in registry). [Size = %s bytes]",
                            block_name,
                            record_size,
                        )
                        devlog.warning(
                            "\tSkipped block %s (not in registry). [Size = %s bytes]",
                            block_name,
                            record_size,
                        )
                        continue

                        # Load recognized data block
                    self._load_data_block(f, block_name, fields)
                except (EOFError, IOError):
                    break

        particle_types = list(set([k[0] for k in fields.keys()]))
        particles = ClusterParticles(particle_types, fields)

        if inplace:
            self._particles = particles
            devlog.debug("Particle data updated in place for %s.", self)

        return particles

    def _load_data_block(
        self, file: BinaryIO, block_name: str, fields: OrderedDict
    ) -> None:
        """Load a data block from the binary file into the fields dictionary.

        Parameters
        ----------
        file : BinaryIO
            The binary file object opened for reading data.
        block_name : str
            The name of the data block to load.
        fields : OrderedDict
            The dictionary to store loaded field data.

        Raises
        ------
        EOFError
            If the end of file is reached unexpectedly.
        ValueError
            If there is an issue with reading the data block or mismatched block structure.
        """
        hdf_name, cg_name = (
            self.field_registry.BINARY_HDF[block_name],
            self.field_registry.BINARY_CG[block_name],
        )
        mylog.debug("\tLoading block %s (%s)...", block_name, cg_name)
        devlog.debug("\tLoading block %s (%s)...", block_name, cg_name)

        # Determine the data structure for the block
        struct_dtype = self._get_block_struct(block_name)
        struct_data = read_record(file, struct_dtype)

        for ptype in struct_dtype.names:
            devlog.debug(
                "\t\tProcessing particle type %s with dtype %s.",
                ptype,
                struct_dtype[ptype].subdtype,
            )
            cg_particle_name = self.particle_registry.get_particle_name(
                int(ptype.replace("Type", ""))
            )

            if cg_particle_name is not None:
                fields[(cg_particle_name, cg_name)] = unyt.unyt_array(
                    struct_data[ptype][0], units=self.field_registry.get_units(hdf_name)
                )

    @classmethod
    def _write_particle_data(
        cls: Type["Gadget2_Binary"],
        path: str | Path,
        particles: "ClusterParticles",
        field_registry: "Gadget2_FieldRegistry",
        particle_registry: "Gadget2_ParticleRegistry",
        header: "Gadget2_Header",
    ) -> None:
        """Write particle data to a binary Gadget-2 file.

        This method writes the particle data to a binary Gadget-2 file without rewriting the header.
        It assumes the header has already been written and appends the particle data in binary format.

        Parameters
        ----------
        path : str | Path
            The file path to write the particle data to.
        particles : ClusterParticles
            The particle data to be written.
        field_registry : Gadget2_FieldRegistry
            The field registry containing mapping and properties of the fields to be written.
        particle_registry : Gadget2_ParticleRegistry
            The particle registry to map Gadget particle types to internal types.
        header : Gadget2_Header
            The header object containing metadata about the file format.
        """
        path = Path(path)
        devlog.info("Writing particle data to binary file: %s", path)
        mylog.info("Writing particle data to binary file: %s", path)

        # Open the file below the header which should already be written to disk.
        with open(path, "ab") as fio:
            for field_name, binary_name in field_registry.HDF_BINARY.items():
                cg_field_name = field_registry.HDF_CG[field_name]

                # Are there any particles with this field?
                if not any(i[1] == cg_field_name for i in particles.fields.keys()):
                    # We failed to find any particles with this field.
                    devlog.debug(
                        "\tNo particles have implemented block %s. Skipping...",
                        binary_name,
                    )
                    continue

                block_dtype = cls.get_block_struct(binary_name, field_registry, header)

                if not block_dtype:
                    devlog.warning(
                        "\tNo block structure for %s. Skipping...", binary_name
                    )
                    continue

                block_specifier = np.array(binary_name, dtype="S4")
                write_record(fio, block_specifier)
                devlog.debug("\tWriting block %s.", binary_name)
                mylog.debug("\tWriting block %s.", binary_name)

                # filling the data
                block_data = np.zeros(1, dtype=block_dtype)

                for ptype in block_dtype.names:
                    cg_particle_name = particle_registry.get_particle_name(
                        int(ptype.replace("Type", ""))
                    )
                    if (
                        cg_particle_name is None
                        or (cg_particle_name, field_registry.BINARY_CG[binary_name])
                        not in particles.fields
                    ):
                        continue

                    units = field_registry.get_units(field_name)

                    arr = particles[
                        cg_particle_name, field_registry.BINARY_CG[binary_name]
                    ]

                    if units is not None:
                        arr = arr.to_value(units)
                        block_data[ptype] = arr

                    else:
                        arr = arr.d if isinstance(arr, unyt.unyt_array) else arr
                        block_data[ptype] = arr

                    devlog.debug(
                        "\t\tWrote data for particle type %s in block %s.",
                        ptype,
                        binary_name,
                    )

                write_record(fio, block_data)

    def _get_block_struct(self, block_name: str) -> np.dtype:
        """Construct the block struct based on header information and field registry
        data.

        Parameters
        ----------
        block_name : str
            The name of the block to construct the struct for.

        Returns
        -------
        np.dtype
            The numpy dtype representing the full structure of the block.
        """
        return self.__class__.get_block_struct(
            block_name, self.field_registry, self.header
        )

    @classmethod
    def get_block_struct(
        cls,
        block_name: str,
        registry: "Gadget2_FieldRegistry",
        header: "Gadget2_Header",
    ) -> np.dtype:
        """Construct the block struct based on header information and field registry
        data.

        Parameters
        ----------
        block_name : str
            The name of the block to construct the struct for.
        registry : Gadget2_FieldRegistry
            The field registry to determine field mappings and properties.
        header : Gadget2_Header
            The header object containing metadata about the file format.

        Returns
        -------
        np.dtype
            The numpy dtype representing the full structure of the block, or None if not applicable.

        Examples
        --------

        .. code-block:: python

            >>> dtype = Gadget2_Binary.get_block_struct('POS ', field_registry, header)
            >>> print(dtype)  # Output: dtype with structure corresponding to the POS block.
        """
        field_name = registry.BINARY_HDF[block_name]
        field_info = registry.list_fields()[field_name]
        field_dtype = field_info["dtype"]
        field_size = field_info["size"]

        num_particles = np.asarray(header.NumPart_ThisFile)
        pid_list = field_info["pid"]

        if pid_list is None:
            pid_list = list(range(header.N_PARTICLE_TYPES))

        dtypes = []

        for particle_type in range(header.N_PARTICLE_TYPES):
            n_part = num_particles[particle_type]
            if (particle_type in pid_list) and (n_part > 0):
                ptype_name = f"Type{particle_type}"
                dtype = (
                    (ptype_name, field_dtype, (n_part, field_size))
                    if field_size > 1
                    else (ptype_name, field_dtype, n_part)
                )
                dtypes.append(dtype)

        return np.dtype(dtypes) if dtypes else None

    def __str__(self) -> str:
        """Provide a user-friendly string representation of the Gadget2_Binary object.

        Returns
        -------
        str
            A string summarizing the file path and format.
        """
        return f"Gadget2_Binary(path='{self.path}')"

    def __repr__(self) -> str:
        """Provide a detailed string representation of the Gadget2_Binary object for
        debugging.

        Returns
        -------
        str
            A detailed string representation of the object.
        """
        return (
            f"Gadget2_Binary(path={self.path}, "
            f"particle_registry={repr(self._particle_registry)}, "
            f"field_registry={repr(self._field_registry)}, "
            f"header={repr(self._header)}, "
            f"particles={repr(self._particles)})"
        )

    def __len__(self) -> int:
        """Return the number of particles in the Gadget-2 file.

        Returns
        -------
        int
            The total number of particles across all types.
        """
        return sum(self.header.NumPart_ThisFile)


class Gadget2_HDF5(Gadget2_ICFile):
    _format = "hdf5"
    _group_prefix = _GADGET_GROUP_PREFIX

    def __init__(self, *args, **kwargs):
        super(Gadget2_HDF5, self).__init__(*args, **kwargs)

        # adding the secret group prefix variable so it can be set at
        # read.
        self._group_prefix = None

    def load_particles(self, inplace: bool = True):
        """Load the particle data from HDF5.

        Parameters
        ----------
        inplace: bool, optional
            If ``True``, then the data of this instance will be automatically updated.
            Default is ``True``.

        Returns
        -------
        ClusterParticles
            The resulting data of the gadget file.
        """
        import re

        mylog.info("Loading particle data from HDF5 file: %s", self.path)
        devlog.debug("Loading particle data from HDF5 file: %s", self.path)

        # Manage the group prefix pattern so that it can be determined based on
        # the matches.
        if self._group_prefix is not None:
            group_pattern = rf"^({self._group_prefix})(\d+)$"
        else:
            group_pattern = r"^(PartType|Part|Type)(\d+)$"

        # Figure out all of the fields that are going to be read.
        field_names = {}

        with h5py.File(self.path, "r") as fio:
            # Select only those group names which meet our match criteria.
            particle_groups = [
                u for u in [re.match(group_pattern, k) for k in fio.keys()] if u
            ]

            if not len(particle_groups):
                raise ValueError(
                    f"Failed to load particles from {self.path}. No particles groups found."
                )

            # grab fields from each particle type.
            devlog.info("\tFound %d particle groups in file.", len(particle_groups))
            for part_group in particle_groups:
                # use the RE match to determine particle id and group name. Look up the corresponding
                # cluster generator name for the particle.
                ptype, prefix, pid = (
                    part_group.group(0),
                    part_group.group(1),
                    int(part_group.group(2)),
                )

                # Do we even recognize this particle type.
                if self.particle_registry.map.get(pid, None) is None:
                    devlog.debug("\tSkipped particle type %s, not in registry.", ptype)
                    continue

                # set the group prefix.
                if self._group_prefix is None:
                    self._group_prefix = prefix
                    devlog.debug(
                        "\tGroup prefix for %s is %s.", self, self._group_prefix
                    )

                for field_name in fio[ptype].keys():
                    if field_name in self.field_registry.list_fields():
                        # The field is in our registry, the particle is in our registry. We read.
                        field_names[ptype, field_name] = (
                            self.particle_registry.get_particle_name(pid),
                            self.field_registry.HDF_CG[field_name],
                        )

        # We've now go the mappings for all of the fields we want to read. The reading process now
        # proceeds.
        pfields = OrderedDict({})

        for k, v in field_names.items():
            # grab all of the field properties we've got.
            ptype_hdf5, field_hdf5 = k
            ptype_cg, field_cg = v
            mylog.debug(
                "\tLoading (%s,%s) - [(%s,%s)]",
                ptype_hdf5,
                field_hdf5,
                ptype_cg,
                field_cg,
            )
            field_properties = self.field_registry.list_fields()[field_hdf5]

            if field_properties["units"] is None:
                with h5py.File(self.path, "r") as f:
                    pfields[(ptype_cg, field_cg)] = f[ptype_hdf5][field_hdf5][:]
            else:
                a = unyt.unyt_array.from_hdf5(
                    str(self.path), dataset_name=field_hdf5, group_name=ptype_hdf5
                )

                a = a.to(field_properties["units"]).astype(field_properties["dtype"])
                pfields[ptype_cg, field_cg] = a

        cg_particle_types = list(set([str(k[0]) for k in pfields]))
        particles = ClusterParticles(cg_particle_types, pfields)

        if inplace:
            self._particles = particles

        return particles

    @classmethod
    def _write_particle_data(
        cls,
        path: str | Path,
        particles: "ClusterParticles",
        field_registry: "Gadget2_FieldRegistry",
        particle_registry: "Gadget2_ParticleRegistry",
        header: "Gadget2_Header",
    ) -> None:
        """Write particle data to an HDF5 Gadget-2 file.

        This method writes the particle data to an HDF5 Gadget-2 file. It creates
        the necessary groups and datasets to store particle information according
        to the Gadget-2 HDF5 format conventions.

        Parameters
        ----------
        path : str or Path
            The file path to write the particle data to.
        particles : ClusterParticles
            The particle data to be written.
        field_registry : Gadget2_FieldRegistry
            The field registry containing mapping and properties of the fields to be written.
        particle_registry : Gadget2_ParticleRegistry
            The particle registry to map Gadget particle types to internal types.
        header : Gadget2_Header
            The header object containing metadata about the file format.
        """
        path = Path(path)
        devlog.info("Writing particle data to HDF5 file: %s", path)
        mylog.info("Writing particle data to HDF5 file: %s", path)

        # Open the file below the header which should already be written to disk.

        for particle_id, cg_particle_name in particle_registry.map.items():
            if (cg_particle_name is None) or (
                particles.num_particles[cg_particle_name] == 0
            ):
                continue

            with h5py.File(path, "a") as fio:
                fio.create_group(f"{cls._group_prefix}{particle_id}")
                devlog.debug(
                    "\tCreated group %s (%s)",
                    f"{cls._group_prefix}{particle_id}",
                    cg_particle_name,
                )

            # Now we iterate through the fields, identify the expected ones and write them.
            for field_name, cg_field_name in field_registry.HDF_CG.items():
                if field_registry.get_pid(
                    field_name
                ) is not None and particle_id not in field_registry.get_pid(field_name):
                    # This isn't a valid field for this particle type.
                    continue
                if (cg_particle_name, cg_field_name) not in particles.fields:
                    # Particles don't have the data.
                    continue

                # Set up the data type and units
                value = particles[(cg_particle_name, cg_field_name)]
                units, dtype = field_registry.get_units(
                    field_name
                ), field_registry.get_dtype(field_name)

                if units is None:
                    if isinstance(value, unyt.unyt_array):
                        arr = np.asarray(value.d, dtype=dtype)
                    else:
                        arr = np.asarray(value, dtype=dtype)

                    with h5py.File(path, "a") as fio:
                        fio[f"{cls._group_prefix}{particle_id}"].create_dataset(
                            field_name, data=arr, dtype=dtype
                        )

                else:
                    arr = value.to(units)
                    arr.write_hdf5(
                        path,
                        dataset_name=field_name,
                        group_name=f"{cls._group_prefix}{particle_id}",
                    )

                devlog.debug(
                    "\t\tWrote data for particle type %s in block %s.",
                    cg_particle_name,
                    field_name,
                )
