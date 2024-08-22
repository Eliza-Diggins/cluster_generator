"""Module for handling Gadget filetypes for initial conditions."""
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Self, Type

import h5py
import numpy as np
import unyt
from numpy.typing import NDArray
from scipy.io import FortranFile

from cluster_generator.particles import ClusterParticles
from cluster_generator.utilities.logging import mylog
from cluster_generator.utilities.types import Attribute, Instance, Value

if TYPE_CHECKING:
    from yt.utilities.cosmology import Cosmology


# Mapping between the header fields in binary and in hdf5.
_GADGET_HEADER_MAP = {
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

_GADGET_HEADER_MAP_R = {v: k for k, v in _GADGET_HEADER_MAP.items()}
# Create a map between the Gadget standard and the cluster generator standard for particle names.
# We don't read the disk particles and let bulge particles be tracer particles, following IllustrisTNG.
_GADGET_PTYPE_MAP = {
    0: "gas",
    1: "dm",
    2: None,
    3: "tracer",
    4: "star",
    6: "black_hole",
}
_GADGET_PTYPE_MAP_R = {v: k for k, v in _GADGET_PTYPE_MAP.items()}
# Create a map between the expected gadget field name and the conventional names in cluster generator.
# Fields not in the list are still initialized, but this ensures that we load the correct fields for things
# like position, velocity, etc.
_GADGET_FIELD_MAP = {
    "Coordinates": "particle_position",
    "Velocities": "particle_velocity",
    "Masses": "particle_mass",
    "Density": "density",
    "Potential": "potential_energy",
    "InternalEnergy": "thermal_energy",
    "MagneticField": "magnetic_field",
    "ParticleIDs": "particle_index",
}
_GADGET_FIELD_MAP_R = {v: k for k, v in _GADGET_FIELD_MAP.items()}
# Mapping from gadget field names to the typical unit. These are read as default.
_GADGET_UNITS_MAP = {
    "Coordinates": "kpc",
    "Velocities": "km/s",
    "Masses": "1e10*Msun",
    "Density": "1e10*Msun/kpc**3",
    "InternalEnergy": "km**2/s**2",
    "Potential": "km**2/s**2",
    "PassiveScalars": "",
    "MagneticField": "1e5*sqrt(Msun)*km/s/(kpc**1.5)",
}


class GadgetHeaderDescriptor(Generic[Instance, Attribute, Value]):
    """Header attribute descriptor for Gadget files.

    These descriptors hide some of the logic used to fetch defaults, set header values,
    and allow users to alter some header values by hand while restricting others.
    """

    def __init__(
        self,
        dtype: np.dtype | str,
        size: int = None,
        settable: bool = False,
        default: Any = None,
    ):
        """Initializes the gadget header descriptor.

        Parameters
        ----------
        size: int
            The expected size of this attribute. If ``None``, we assume it is scalar.
        dtype: str or np.dtype
            The data type of this attribute.
        settable: bool
            If ``True``, then this header attribute may be set.
        default: Any
            The default value of this header descriptor.
        """
        self.attribute_name: str | None = None
        """ str: This is the name of the attribute.

        Set by ``__set_name__``.
        """
        self.binary_name: str | None = None
        """ str: This is the header's name in the binary Gadget file formats.
        """
        self.size: int = size if size is not None else 1
        """ int: The size of the header variable."""
        self.dtype: np.dtype = np.dtype(dtype)
        """ dtype: The data type of this attribute."""
        self.settable: bool = settable
        """ bool: If ``True``, then we allow the user to alter this descriptor."""
        self.default: Any = (
            np.asarray(default, dtype=self.dtype) if default is not None else None
        )
        """The default value of the attribute."""

    def __set_name__(self, owner: Type[Instance], name: str):
        self.attribute_name: str = name
        self.binary_name: str = _GADGET_HEADER_MAP_R.get(self.attribute_name, None)

    def __get__(self, instance: Instance, owner: Type[Instance]) -> Value:
        # Fetch the attribute depending on the type of the instance and owner.
        # If instance isn't none, then we return the actual value of the header variable.
        # if instance is none, then we return the data for generating the header dtypes.
        if instance is None:
            # The instance is none, we return the metadata, not the actual parameter.
            return (
                self.attribute_name,
                self.binary_name,
                self.dtype,
                self.size,
                self.settable,
                self.default,
            )
        else:
            # determine the file type from the class variable so we know what name to use.
            if instance.__class__._ftype == "binary":
                key = self.binary_name
            else:
                key = self.attribute_name

            # return the instance.
            if instance is None:
                raise AttributeError(
                    f"GadgetICFile attribute {self.attribute_name} isn't a class attribute."
                )

            return instance._header.get(key, self.default)

    def __set__(self, instance: Instance, value: Value) -> None:
        # We don't allow users to set the class version of the attribute.
        if instance is None:
            raise AttributeError(
                f"Attribute {self.attribute_name} isn't a set-able class attribute."
            )
        if not self.settable:
            raise AttributeError(f"Attribute {self.attribute_name} isn't settable.")

        # We now check the ftype to see what the right name is and return the value.
        if instance.__class__._ftype == "binary":
            key = self.binary_name
        else:
            key = self.attribute_name

        instance._header[key] = np.asarray(value, dtype=self.dtype)

    def get_from_dict(self, kwargs: dict, duplicated: str = "error"):
        """Give a dictionary of keys and values, identify matching keys and grab
        corresponding value.

        Parameters
        ----------
        kwargs: dict
            The dictionary to search.
        duplicated: str
            What to do if both the binary and hdf5 flags are present in the dictionary. If ``'hdf5'`` or ``'binary'``, then
            whichever was specified will take preference. If ``'error'``, then an error is raised.

        Returns
        -------
        key: str
            The matched key.
        value: Any
            The matched value

        Notes
        -----

        The dict ``kwargs`` gets searched for either the binary or hdf5 flag for the header variable and then
        pulled.
        """
        flag_hdf, flag_bin = self.attribute_name, self.binary_name

        key, value = None, None
        if (flag_hdf in kwargs) and (flag_bin in kwargs):
            # We have duplicate values.
            if duplicated == "error":
                raise ValueError(f"Found both {flag_hdf} and {flag_bin} in dict.")
            elif duplicated == "hdf5":
                key, value = flag_hdf, kwargs[flag_hdf]
            elif duplicated == "binary":
                key, value = flag_bin, kwargs[flag_bin]
            else:
                raise ValueError(
                    f"Parameter duplicated has value {duplicated} which is not permitted."
                )

        elif flag_hdf in kwargs:
            key, value = flag_hdf, kwargs[flag_hdf]
        elif flag_bin in kwargs:
            key, value = flag_bin, kwargs[flag_bin]
        else:
            pass

        return key, value


class GadgetICFile(ABC):
    """Generic wrapper for a Gadget-style initial conditions file.

    The :py:class:`GadgetICFile` class is a generic wrapper for the 2 Gadget-2 standards for particle IC file formatting:

    1. **Gadget-2 Style**: An unformatted fortran binary filetype with additional metadata.
    2. **Gadget-2 HDF5**: An HDF5 implementation of the Gadget-2 style.

    Because these file formats are ubiquitous in various simulation softwares, we provide a fairly complete
    implementation for the file type.

    .. note::

        We don't support the antiquated Gadget-1 binary file format because IO support is entirely
        dependent on the Makefile parameters of the simulation code and it is not widely used anymore.
    """

    # Each of the sublcasses sets ``_ftype`` to indicate to the descriptors if its binary or HDF5 and therefore which
    # header keys are correct.
    _ftype = None

    # Each header flag gets its own descriptor with all the information necessary to allow users to access the
    # attribute, to determine the data struct, to set and parse the header, etc.
    NumPart_ThisFile: NDArray[np.uint32] = GadgetHeaderDescriptor(
        "u4", size=6, settable=False
    )
    """array of int: The number of particles of each particle type."""
    MassTable: NDArray[np.float64] = GadgetHeaderDescriptor(
        "f8", size=6, settable=False, default=[0, 0, 0, 0, 0, 0]
    )
    """array of float: The fixed mass of each particle type.
    .. hint::

        If the ``Massarr`` has a zero entry for a given particle, it may still be specified explicitly by
        the inclusion of a ``particle_mass`` field for each particle individually.

    """
    Time: float = GadgetHeaderDescriptor("f8", settable=True, default=0.0)
    """float: The time (in code units) of this initial condition.

    This is generally unused.
    """
    Redshift: float = GadgetHeaderDescriptor("f8", settable=True, default=0.0)
    """float: The redshift of the simulation."""
    Flag_Sfr: int = GadgetHeaderDescriptor("i4", settable=True, default=0)
    """int: A flag indicating whether star formation is included (1 for yes, 0 for no)."""
    Flag_Feedback: int = GadgetHeaderDescriptor("i4", settable=True, default=0)
    """int: A flag indicating whether feedback processes are included (1 for yes, 0 for no)."""
    NumPart_Total: NDArray[np.uint32] = GadgetHeaderDescriptor(
        "i4", settable=False, size=6
    )
    """array of int: The total number of particles of each type across the entire simulation.
    This includes particles from all files if the snapshot is split across multiple files.
    """
    Flag_Cooling: int = GadgetHeaderDescriptor("i4", settable=True, default=0)
    """int: A flag indicating whether cooling processes are included (1 for yes, 0 for no)."""
    NumFilesPerSnapshot: int = GadgetHeaderDescriptor("i4", settable=False, default=1)
    """int: The number of files used for this snapshot.
    This is useful when the snapshot is split across multiple files.
    """
    BoxSize: float = GadgetHeaderDescriptor("f8", settable=True, default=0.0)
    """float: The size of the simulation box in comoving units."""
    Omega0: float = GadgetHeaderDescriptor("f8", settable=True, default=0.0)
    """float: The matter density parameter at redshift 0."""
    OmegaLambda: float = GadgetHeaderDescriptor("f8", settable=True, default=0.0)
    """float: The cosmological constant density parameter at redshift 0."""
    HubbleParam: float = GadgetHeaderDescriptor("f8", settable=True, default=0.0)
    """float: The Hubble constant in units of 100 km/s/Mpc."""
    Flag_StellarAge: int = GadgetHeaderDescriptor("i4", settable=True, default=0)
    """int: A flag indicating whether the age of stars is tracked (1 for yes, 0 for no)."""
    Flag_Metals: int = GadgetHeaderDescriptor("i4", settable=True, default=0)
    """int: A flag indicating whether metallicity information is included (1 for yes, 0 for no)."""
    NumPart_Total_HW: NDArray[np.int32] = GadgetHeaderDescriptor(
        "i4", settable=False, size=6, default=[0, 0, 0, 0, 0, 0]
    )
    """array of int: The high word of the total number of particles of each type.

    This is used for simulations with more than 2^32 particles.
    """
    Flag_Entropy_ICs: int = GadgetHeaderDescriptor("i4", settable=True, default=0)
    """int: A flag indicating whether entropy is used as an initial condition (1 for yes, 0 for no)."""

    def __init__(self, path: str | Path):
        """Initialize a generic :py:class:`GadgetICFile` instance.

        Parameters
        ----------
        path: str or Path
            The path to the underlying file.
        """
        self.path = Path(path)
        """ str: The path to the underlying Gadget-2 style file."""

        self._particles: ClusterParticles | None = None
        # This houses the underlying particle dataset.
        self._header: dict[str, Any] | None = None
        # This houses the underlying header.

    @property
    def header(self) -> dict[str, Any]:
        """dict: The header of this file.

        The header is a dictionary of ``key: value`` pairs containing the cosmological information
        and a variety of other metadata for the initial condition file. In many cases, these are
        almost all 0 because cosmology is not relevant.
        """
        if self._header is None:
            _ = self.load_header()

        return self._header

    @property
    def particles(self) -> ClusterParticles:
        """ClusterParticles: The particle dataset represented by this file."""
        if self._particles is None:
            _ = self.load_data()

        return self._particles

    @abstractmethod
    def load_header(self, inplace: bool = True) -> dict[str, Any]:
        """Load the header information from the underlying gadget file.

        Parameters
        ----------
        inplace: bool
            If ``True``, then the header attribute of this instance will be automatically updated.
            Default is ``True``.

        Returns
        -------
        dict
            The resulting header of the gadget file.

        Notes
        -----

        In the ``hdf5`` file, this is the ``/Header`` group. In the
        binary format, this is placed in the ``HEAD`` block at the beginning of the file and
        is more complicated to read.
        """
        pass

    @abstractmethod
    def load_data(self, inplace: bool = True) -> ClusterParticles:
        """Load the particles that are stored in the underlying file.

        Parameters
        ----------
        inplace: bool
            If ``True``, then the data of this instance will be automatically updated.
            Default is ``True``.

        Returns
        -------
        ClusterParticles
            The resulting data of the gadget file.
        """
        pass

    @classmethod
    def from_particles(
        cls, particles: ClusterParticles, path: str | Path, *args, **kwargs
    ) -> Self:
        """Convert a :py:class:`particles.ClusterParticles` instance into a Gadget file
        of a particular type.

        Parameters
        ----------
        particles: :py:class:`particles.ClusterParticles`
            The particles that should be converted to this format.
        path: str
            The path at which the gadget file should be generated.
        args
            Additional arguments.
        kwargs
            Additional kwargs.

        Returns
        -------
        :py:class:`GadgetICFile`
            The resulting file reference.
        """
        pass

    @classmethod
    def _get_header_descriptors(cls) -> dict[str, GadgetHeaderDescriptor]:
        """Finds all instances of :py:class:`GadgetHeaderDescriptor` attached to the
        class, including inherited ones.

        Returns
        -------
        dict[str, _GadgetHeaderDescriptor]
            A dictionary where the keys are the attribute names and the values are the GadgetHeaderDescriptor instances.

        Notes
        -----

        This allows the class to be self-aware of the meta data for each of the descriptors. We can then access the
        instances to construct structs and other important data from the included meta-data.
        """
        import inspect

        descriptors = {}
        # Inspect all members of the class (including inherited ones)
        for name, attribute in inspect.getmembers(cls):
            if isinstance(attribute, GadgetHeaderDescriptor):
                descriptors[name] = attribute
        return descriptors

    @classmethod
    def cosmology_to_header_params(cls, cosmology: "Cosmology") -> dict[str, Any]:
        """Parse a yt Cosmology object to extract the necessary cosmology values for the
        Gadget-2 header.

        Parameters
        ----------
        cosmology : yt.utilities.cosmology.Cosmology
            The yt Cosmology object containing the cosmological parameters.

        Returns
        -------
        dict
            A dictionary containing the cosmological values for the Gadget-2 header:
            - 'Omega0': The matter density parameter at redshift 0.
            - 'OmegaLambda': The cosmological constant density parameter at redshift 0.
            - 'HubbleParam': The Hubble constant in units of 100 km/s/Mpc.
        """
        omega0 = cosmology.omega_matter
        omega_lambda = cosmology.omega_lambda

        # Extract HubbleParam (Hubble constant in units of 100 km/s/Mpc)
        # yt's Hubble parameter is typically in units of km/s/Mpc, so divide by 100 to get HubbleParam
        hubble_param = cosmology.hubble_constant.in_units("km/s/Mpc").value / 100.0

        return {
            "Omega0": omega0,
            "OmegaLambda": omega_lambda,
            "HubbleParam": hubble_param,
        }

    @classmethod
    def generate_header(cls, mapping: dict) -> tuple[dict[str, Any], dict[str, Any]]:
        """Generate a header from a set of dictionary ``key, value`` pairs.

        Parameters
        ----------
        mapping: dict
            A mapping to parse as the parameters.

        Returns
        -------
        dict:
            The header parameters.

        Notes
        -----

        In effect, this method will search your ``mapping`` for either the ``binary`` or ``hdf5`` key for
        each of the header values. If it finds one, it will grab it and its value to build the header.

        You can also specify ``cosmology`` to auto-fill the cosmology information.
        """
        # Managing a specified cosmology if it has been provided.
        if "cosmology" in mapping:
            # we need to parse the needed values from this and add them to the mapping
            cosmo = mapping.pop("cosmology")

            mapping = {**mapping, **cls.cosmology_to_header_params(cosmo)}

        # We now iterate over all of the headers available. If they are settable we may allow them to be set.
        # otherwise, we will rely on the default.
        # This then sets us up to only have to fill out the data specific header flags (NumParts, etc.) manually.
        header = {}
        _expected_header_mapping = cls._get_header_descriptors()  # All of the flags.

        for _, descriptor in _expected_header_mapping.items():
            # grab the actual key from the mapping and the actual value.
            _true_key, _ = descriptor.get_from_dict(mapping, duplicated="error")
            header = mapping.pop(_true_key)  # we do this so we can reduce the mapping.

        # We've removed kwargs from the map as we go, return the header dict and the map seperately.
        return header, mapping


class Gadget2HDF5(GadgetICFile):
    """Class representation of a Gadget-2 style HDF5 initial conditions file for
    particles.

    The Gadget-2 HDF5 format is a widely-used format for storing initial conditions of cosmological
    simulations. It is an alternative to the unformatted binary format (Gadget-2 binary format) and
    offers more flexibility and portability due to its self-describing nature.

    .. hint::

        We generally suggest using HDF5 over unformatted binary types whenever possible.

    The HDF5 format organizes data into groups, datasets, and attributes:

    .. rubric:: Structure Overview

    - **Header Group** (``/Header``):
        The ``Header`` group contains attributes that store metadata about the simulation. This includes
        information such as the number of particles of each type, the simulation box size, the cosmological
        parameters, and other relevant flags. Each attribute is stored as a named key-value pair.

    - **Particle Groups** (``/PartTypeN``):
        Each particle type is stored in a separate group named `PartTypeN`, where `N` is an integer from
        0 to 5 corresponding to the following particle types:
        - `PartType0`: Gas particles
        - `PartType1`: Dark matter particles
        - `PartType2`: (Unused, typically)
        - `PartType3`: Tracer particles or "bulge" particles in some implementations
        - `PartType4`: Star particles
        - `PartType5`: Black hole particles

        Additionally, the prefix for each particle type may be changed.

        Each of these groups contains datasets representing different physical quantities for that
        particle type. Common datasets include:
        - `Coordinates`: The positions of the particles (in comoving kpc by default).
        - `Velocities`: The velocities of the particles (in km/s by default).
        - `ParticleIDs`: Unique identifiers for the particles.
        - `Masses`: The mass of each particle (in units of 1e10 Msun/h by default).

    .. rubric:: The Header

    The following attributes are typically found in the `/Header` group:

    - `NumPart_ThisFile` (Npart): An array of integers indicating the number of particles of each type
      in this file.
    - `NumPart_Total` (Nall): An array of integers indicating the total number of particles of each
      type across all files in the simulation.
    - `MassTable` (Massarr): An array of floats representing the fixed mass of each particle type.
      If a particle type has a non-zero mass in this array, it means that all particles of that type
      have the same mass. Otherwise, the `Masses` dataset is used.
    - `Time`: The time (in code units) at which the snapshot was taken. For cosmological simulations,
      this is usually set to 1.0 / (1.0 + Redshift).
    - `Redshift`: The redshift at which the snapshot was taken.
    - `BoxSize`: The size of the simulation box (in comoving kpc/h).
    - `Omega0`: The matter density parameter at the present time.
    - `OmegaLambda`: The cosmological constant density parameter at the present time.
    - `HubbleParam`: The Hubble constant in units of 100 km/s/Mpc.
    - `Flag_Sfr`: A flag indicating whether star formation is enabled.
    - `Flag_Cooling`: A flag indicating whether cooling processes are enabled.
    - `Flag_Feedback`: A flag indicating whether feedback processes are enabled.
    - `Flag_Age`: A flag indicating whether stellar ages are tracked.
    - `Flag_Metals`: A flag indicating whether metallicity information is included.
    - `NumFilesPerSnapshot`: The number of files in which this snapshot is split.


    Examples:
    ---------

    Loading a Gadget-2 HDF5 file:

    ```python
    from cluster_generator import Gadget2HDF5

    # Load the particles from a Gadget-2 HDF5 file
    g2_hdf5 = Gadget2HDF5('path/to/snapshot.hdf5')
    particles = g2_hdf5.particles

    # Accessing header information
    header = g2_hdf5.header
    print("Box Size:", header['BoxSize'])
    print("Number of gas particles:", header['NumPart_ThisFile'][0])
    ```

    Writing particles to a new Gadget-2 HDF5 file:

    ```python
    # Assuming 'particles' is a ClusterParticles object with the desired data
    Gadget2HDF5.from_particles(particles, 'path/to/new_snapshot.hdf5')
    ```

    Notes
    -----

    - The Gadget-2 HDF5 format is often used in large cosmological simulations such as those run with
      the Illustris or Eagle codes.
    - When writing a Gadget-2 HDF5 file from particle data, it is important to ensure that the data
      is correctly organized and that all necessary header information is provided.
    """

    _ftype = "hdf5"

    def __init__(self, path: str | Path, group_prefix: str | None = None):
        """Initialize the :py:class:`Gadget2HDF5` class.

        Parameters
        ----------
        path: str
            The path to the gadget file.
        group_prefix: str, optional
            The prefix in the HDF5 file before each particle group. By default, we will detect which is used.
            GADGET-2 convention of ``"TypeN"``, but ``"PartTypeN"`` is common in other simulation software
            like AREPO.
        """
        super(Gadget2HDF5, self).__init__(path)

        self.group_prefix: str | None = group_prefix
        """ str: The prefix in front of each particle group."""

    def load_header(self, inplace: bool = True) -> dict[str, Any]:
        _header_dict = {}

        # Determine the expected header information using the
        # descriptors. This is more necessary for the binary files.
        _header_flags = {}
        for _, descriptor in self._get_header_descriptors().items():
            _header_flags[descriptor.attribute_name] = (
                descriptor.dtype,
                descriptor.size,
            )

        # Proceed to read the header group and parse values.
        with h5py.File(self.path, "r") as fio:
            for flag, flag_params in _header_flags.items():
                if flag not in fio["Header"].attrs.keys():
                    # We couldn't find this header. Maybe missing on purpose, but we raise a warning anyways.
                    _header_dict[flag] = None
                    mylog.warning(
                        "GADGET-2 HDF5 File failed to locate header flag %s in HEADER attributes.",
                        flag,
                    )
                    continue
                else:
                    # We do have this flag.
                    _header_dict[flag] = (
                        fio["Header"].attrs[flag][...]
                        if flag_params[1] > 1
                        else fio["Header"].attrs[flag]
                    )

        if inplace:
            self._header = _header_dict

        return _header_dict

    def load_data(self, inplace: bool = True) -> ClusterParticles:
        """Load the particle data from HDF5.

        Parameters
        ----------
        inplace: bool
            If ``True``, then the data of this instance will be automatically updated.
            Default is ``True``.

        Returns
        -------
        ClusterParticles
            The resulting data of the gadget file.
        """
        import re

        # Determine which particle types we have and get the correct particle type name.
        # We allow for TypeN or PartTypeN (Gadget vs. AREPO).
        if self.group_prefix is not None:
            group_pattern = rf"^({self.group_prefix})(\d+)$"
        else:
            group_pattern = r"^(PartType|Part)(\d+)$"

        field_names, cg_field_names = {}, {}
        with h5py.File(self.path, "r") as fio:
            _available_hdf5_keys = list(fio.keys())

            # Select only those group names which meet our match criteria.
            particle_types = [
                u
                for u in [re.match(group_pattern, k) for k in _available_hdf5_keys]
                if u
            ]
            if not len(particle_types):
                raise ValueError(
                    f"Failed to load particles from {self.path}. No particles groups found."
                )

            # grab fields from each particle type.
            for particle_type in particle_types:
                # use the RE match to determine particle id and group name. Look up the corresponding
                # cluster generator name for the particle.
                ptype, pid = particle_type.group(0), int(particle_type.group(2))
                field_names[ptype] = list(fio[ptype].keys())

                # For each identified field, seek a cluster_generator alias.
                for field_name in field_names[ptype]:
                    cg_field_names[(ptype, field_name)] = (
                        _GADGET_PTYPE_MAP[pid],
                        _GADGET_FIELD_MAP[field_name],
                    )

            # Check for the prefix.
            if not self.group_prefix:
                self.group_prefix = list(
                    set(
                        list(
                            [particle_type.group(1) for particle_type in particle_types]
                        )
                    )
                )[0]

        # Proceed with actual loading procedure.
        fields = OrderedDict()
        for ptype in field_names:
            for field in field_names[ptype]:
                # look up the cluster_generator alias for the field.
                cg_field_name = cg_field_names[(ptype, field)]

                if cg_field_name[1] == "particle_index":  # We cannot load with units.
                    with h5py.File(self.path, "r") as f:
                        fields[cg_field_name] = f[ptype][field][:]
                else:
                    # This has units.
                    a = unyt.unyt_array.from_hdf5(
                        str(self.path), dataset_name=field, group_name=ptype
                    )

                    # Check the units and apply defaults. If the field is dimensionless, we try to look it up.
                    # if we cannot find it, then we let it stay.
                    if a.units == "dimensionless" and field in _GADGET_UNITS_MAP:
                        fields[cg_field_name] = unyt.unyt_array(
                            a.d.astype("float64"), str(_GADGET_UNITS_MAP[field])
                        ).in_base("galactic")
                    else:
                        fields[cg_field_name] = unyt.unyt_array(
                            a.d.astype("float64"), str(a.units)
                        ).in_base("galactic")

        cg_particle_types = [str(k) for k in fields]
        particles = ClusterParticles(cg_particle_types, fields)

        if inplace:
            self._particles = particles

        return particles

    @classmethod
    def from_particles(
        cls, particles: ClusterParticles, path: str | Path, **kwargs
    ) -> Self:
        """Create a :py:class:`Gadget2HDF5` instance and file from a
        :py:class:`particles.ClusterParticles` object.

        Parameters
        ----------
        particles : ClusterParticles
            The particle data to write to the HDF5 file.
        path : str or Path
            The path where the HDF5 file will be created.
        kwargs:
            Additional kwargs that are used to fill in the header and details about the
            included particles.

        Returns
        -------
        Gadget2HDF5
            An instance of Gadget2HDF5 initialized from the particle data.
        """
        path = Path(path)
        group_prefix = kwargs.pop(
            "group_prefix", "PartType"
        )  # The prefix on groups in HDF5.

        # Constructing the header information for the file
        # The kwargs are parsed and used to fill out the header. If defaults exist, they are used.
        header, kwargs = cls.generate_header(kwargs)

        # We are still missing some of the parameters that depend explicitly on the particle dataset.
        # The number of particles, total particles, number of files, etc. are all set intuitively.
        header["NumPart_ThisFile"] = np.array(
            [particles.num_particles.get(_GADGET_PTYPE_MAP[i], 0) for i in range(6)],
            dtype="u4",
        )
        header["NumPart_Total"] = np.array(
            [particles.num_particles.get(_GADGET_PTYPE_MAP[i], 0) for i in range(6)],
            dtype="u4",
        )

        # We don't set the MassTable because (by default) cluster_generator specifies masses for each block.
        # We may be able to speed up IO by adding some logic around self-consistent particle masses later.

        # Figure out all the fields to write.
        field_names, cg_field_names = {}, {}

        for particle_type in particles.particle_types:
            # Get the name of the group as it should appear in GADGET.
            pid = f"{group_prefix}{_GADGET_PTYPE_MAP_R[particle_type]}"
            field_names[pid] = list(
                [
                    _GADGET_FIELD_MAP_R[field[1]]
                    for field in particles.fields
                    if field[0] == particle_type
                ]
            )

            # For each identified field, seek a cluster_generator alias.
            # Effectively inverse what was done above.
            for field_name in field_names[pid]:
                cg_field_names[(pid, field_name)] = (
                    particle_type,
                    _GADGET_FIELD_MAP[field_name],
                )

        for particle_group, fields in field_names.items():
            # Create the group for each of the particle groups in the fields.
            with h5py.File(path, "w") as fio:
                fio.create_group(particle_group)

            for field in fields:
                # look up the cluster_generator alias for the field.
                cg_field_name = cg_field_names[(particle_group, field)]

                if cg_field_name[1] == "particle_index":  # We don't need units.
                    with h5py.File(path, "w") as fio:
                        fio[particle_group].create_dataset(
                            field, data=particles.fields[cg_field_name]
                        )
                else:
                    # This has units.
                    particles.fields[cg_field_name].write_hdf5(
                        str(path), dataset_name=field, group_name=particle_group
                    )

        # Return the instance of Gadget2HDF5
        instance = cls(path)

        return instance


class Gadget2Binary(GadgetICFile):
    """Class representation of a Gadget-2 style binary initial conditions file for
    particles.

    .. rubric:: Structure

    - **Header Block**:
        The file begins with a header block that contains essential metadata about the simulation, such as
        the number of particles, the simulation box size, cosmological parameters, and various flags. The header
        is structured as a fixed-size block in the binary file, with fields corresponding to specific attributes.

    - **Data Blocks**:
        Following the header, the file contains several data blocks, each identified by a 4-character block specifier.
        These blocks store particle data such as positions, velocities, masses, and internal energy. Each block is
        structured as a contiguous array of data, corresponding to the different particle types.

        Common block specifiers include:
        - `POS`: Positions of particles, stored as a 3D vector.
        - `VEL`: Velocities of particles, stored as a 3D vector.
        - `ID`: Unique identifiers for particles.
        - `MASS`: Masses of particles, stored as a scalar value.
        - `U`: Internal energy of gas particles.
        - `RHO`: Density of gas particles.
        - `HSML`: Smoothing lengths of gas particles.
        - `POT`: Gravitational potential of particles.
        - `ACCE`: Accelerations of particles.
        - `ENDT`: Rate of change of entropy for gas particles.
        - `TSTP`: Timesteps of particles.

    .. rubric:: Header Flags

    The following attributes are typically found in the header block:

    - `Npart`: An array of integers indicating the number of particles of each type in this file.
    - `Massarr`: An array of floats representing the fixed mass of each particle type. If a particle type
      has a non-zero mass in this array, it means that all particles of that type have the same mass.
    - `Time`: The time (in code units) at which the snapshot was taken. For cosmological simulations,
      this is usually set to 1.0 / (1.0 + Redshift).
    - `Redshift`: The redshift at which the snapshot was taken.
    - `BoxSize`: The size of the simulation box (in comoving kpc/h).
    - `Omega0`: The matter density parameter at the present time.
    - `OmegaLambda`: The cosmological constant density parameter at the present time.
    - `HubbleParam`: The Hubble constant in units of 100 km/s/Mpc.
    - `Flag_Sfr`: A flag indicating whether star formation is enabled.
    - `Flag_Cooling`: A flag indicating whether cooling processes are enabled.
    - `Flag_Feedback`: A flag indicating whether feedback processes are enabled.
    - `Flag_Age`: A flag indicating whether stellar ages are tracked.
    - `Flag_Metals`: A flag indicating whether metallicity information is included.
    - `NumFilesPerSnapshot`: The number of files in which this snapshot is split.

    Examples
    --------
    Loading a Gadget-2 binary file:

    ```python
    from cluster_generator import Gadget2Binary

    # Load the particles from a Gadget-2 binary file
    g2_binary = Gadget2Binary('path/to/snapshot.bin')
    particles = g2_binary.particles

    # Accessing header information
    header = g2_binary.header
    print("Box Size:", header['BoxSize'])
    print("Number of gas particles:", header['Npart'][0])
    ```

    Writing particles to a new Gadget-2 binary file:

    ```python
    Gadget2Binary.from_particles(particles, 'path/to/new_snapshot.bin')
    ```

    Notes
    -----
    - The format is less self-describing than HDF5, so it is crucial to maintain accurate documentation of
      the data structures and units used.
    """

    _ftype = "binary"

    # The default data types for the binary blocks are specified here. They may vary (i.e. AREPO) based on
    # user configuration and therefore need to mutable. Generally, we will mess with these in specific
    # frontends.
    _block_dtypes = {
        "POS": "f4",
        "VEL": "f4",
        "ID": "uint4",
        "MASS": "f4",
        "U": "f4",
        "RHO": "f4",
        "HSML": "f4",
        "POT": "f4",
        "ACCE": "f4",
        "ENDT": "f4",
        "TSTP": "f4",
    }

    # Names of each block to convert back to HDF5 and then to cluster generator field names.
    _block_names = {
        "POS": "Coordinates",
        "VEL": "Velocities",
        "ID": "ParticleIDs",
        "MASS": "Masses",
        "U": "InternalEnergy",
        "RHO": "Density",
        "HSML": "SmoothingLength",
        "POT": "Potential",
        "ACCE": "Acceleration",
        "ENDT": "RateOfChangeOfEntropy",
        "TSTP": "TimeStep",
    }

    def __init__(
        self,
        path: str | Path,
        units: dict[str, unyt.unyt_quantity | unyt.Unit] = None,
        dtypes: dict[str, str | np.dtype] = None,
    ):
        """Initialize the :py:class:`Gadget2Binary` class with a path to the binary file
        and optional units.

        Parameters
        ----------
        path : str or Path
            The path to the Gadget-2 binary file.
        units : dict[str, unyt.unyt_quantity | unyt.Unit], optional
            A dictionary of units to override AREPO style defaults.

            .. hint::

                Generally, the user **should** set these when reading a ``.g2`` file. Depending on the
                software you used to generate the file, the units may vary wildly.

        dtypes: dict[str, str | np.dtype], optional
            The dtypes for the different blocks. Specified at single precision by default.

            .. hint::

                Unlike ``units``, this shouldn't be altered except as specified by a particular simulation
                code. Some codes, like ``AREPO`` allow for double precision initial conditions, in which case this
                will matter quite a lot.
        """
        super().__init__(path)

        # Initialize units with defaults and override with any user-specified units.
        self.units = {k: unyt.Unit(v) for k, v in _GADGET_UNITS_MAP.items()}

        if units is not None:
            for k, v in units.items():
                self.units[k] = unyt.Unit(v)

        # Initialize dtypes with defaults and override with any user-specified dtypes.
        self.dtypes = {k: np.dtype(v) for k, v in self.__class__._block_dtypes.items()}

        if dtypes is not None:
            for k, v in dtypes.items():
                self.dtypes[k] = np.dtype(v)

    @classmethod
    def _get_header_struct(cls) -> np.dtype:
        # Construct the numpy struct to read the binary header data.
        # We use the descriptors to access this information.
        dtypes = {}
        _descriptors = cls._get_header_descriptors()

        for _, v in _descriptors.items():
            # proceed by grabbing the dtypes from each descriptor. We cannot immediately
            # construct because the order of the struct matters and we need to force that
            # ordering.
            binary_flag, size, dtype = v.binary_name, v.size, v.dtype
            dtypes[binary_flag] = (size, dtype)

        # Gadget-2 binary requires 256 bytes in the header. Thus, we need to fill unused with
        # 60 unsigned ascii char.
        return np.dtype(
            [
                (flag, dtypes[flag][1], dtypes[flag][0])
                if dtypes[flag][0] > 1
                else (flag, dtypes[flag][1])
                for flag in _GADGET_HEADER_MAP.keys()
            ]
            + [("unused", "a60")]
        )

    @staticmethod
    def _read_block_specifier(f: FortranFile) -> str:
        """Reads a 4-character block specifier from a Gadget-2 binary file.

        Parameters
        ----------
        f : FortranFile
            The FortranFile object representing the opened binary file.

        Returns
        -------
        str
            The block specifier as a string.

        Notes
        -----

        As the file is read, this is called at the top of each block to ensure
        that we know where we are in the file and that everything is being read correctly.
        """
        # Define a dtype for a 4-character string (4 bytes)
        block_specifier_dtype = np.dtype("S4")

        # Read the block specifier from the file
        block_specifier = f.read_record(block_specifier_dtype)

        # Convert the byte string to a regular string and return it
        return block_specifier[0].decode("utf-8")

    @classmethod
    def _get_block_struct_from_header(
        cls, block: str, header: dict[str, Any], dtypes: dict[str, np.dtype] = None
    ) -> np.dtype:
        """Create the structured dtype for reading a specific block of data from a
        Gadget-2 binary file.

        This method constructs the appropriate NumPy dtype based on the information provided
        in the file header and the block name. The dtype will account for the number of particles
        of each type and the specific data structure associated with the block (e.g., positions, velocities).

        Parameters
        ----------
        block : str
            The block specifier indicating the type of data to be read (e.g., 'POS', 'VEL').
        header : dict[str, Any]
            The header dictionary containing metadata from the Gadget-2 file, including the
            total number of particles of each type ('Nall') and the mass array ('Massarr').
        dtypes : dict[str, np.dtype], optional
            A dictionary mapping block specifiers to their corresponding NumPy data types. In most
            implementations, this will be the class's default dtypes.

        Returns
        -------
        np.dtype
            The structured NumPy dtype that can be used to read the specified block from the
            binary file.

        Notes
        -----
        - The dtype accounts for the shape of the data (e.g., 3D vectors for positions and velocities).
        - For blocks like 'MASS', only particles with inhomogeneous masses are included in the dtype.
        - Gas-specific blocks (e.g., 'U', 'RHO') are restricted to gas particles (Type 0).
        """
        if dtypes is None:
            dtypes = cls._block_dtypes

        # Parse the important parts of the header.
        num_part_total, mass_table = header["Nall"], header["Massarr"]

        # Ensure the block specifier is properly formatted.
        block = str(block).lstrip()

        # Determine the shape of the data based on the block specifier.
        if block in ["POS", "VEL", "ACC"]:
            shape = 3
        else:
            shape = 1

        # Create the dtype for each particle in the block.
        dtype = np.dtype(
            [(block, dtypes[block], (shape,)) if shape > 1 else (block, dtypes[block])]
        )
        # Construct the struct for different particle types.
        # Identify the particle types that need to be included in this block.
        if block in ["POS", "VEL", "ACCE", "TSTP"]:
            # Include all particle types.
            block_lengths = {f"Type{i}": num_part_total[i] for i in range(6)}
        elif block == "MASS":
            # Include only particles with inhomogeneous masses.
            block_lengths = {
                f"Type{i}": num_part_total[i]
                for i in range(6)
                if (num_part_total[i] > 0) and (mass_table[i] == 0.0)
            }
        else:
            # Include only gas particles (Type 0) for gas-specific blocks.
            block_lengths = {"Type0": num_part_total[0]}

        # Filter out any particle types with zero length.
        block_lengths = {k: v for k, v in block_lengths.items() if v != 0}

        # Create the final structured dtype for the block.
        struct_dtype = np.dtype([(k, dtype, (v,)) for k, v in block_lengths.items()])
        return struct_dtype

    def _get_block_struct(self, block: str):
        # Alias into the _get_block_struct_from_header method so that we can call as both
        # class and instance.
        return self.__class__._get_block_struct_from_header(
            block, self._header, self.dtypes
        )

    def load_header(self, inplace: bool = True) -> dict[str, Any]:
        """Load the Gadget2 header from the binary file.

        Parameters
        ----------
        inplace: bool
            If ``True``, then the header attribute of this instance will be automatically updated.
            Default is ``True``.

        Returns
        -------
        dict
            The resulting header of the gadget file.
        """
        # fetch the header dtype and use it to read the binary struct
        header_dtype = self.__class__._get_header_struct()
        _header_dict = {}
        with FortranFile(self.path, "r") as f:
            try:
                header = f.read_record(header_dtype)
            except (EOFError, IOError) as e:
                raise IOError(f"Failed to read header of {self.path}: {e.__str__()}.")

        for hkey in header_dtype.names:
            _header_dict[hkey] = header[hkey][0]

        if inplace:
            self._header = _header_dict

        return _header_dict

    def load_data(self, inplace: bool = True) -> ClusterParticles:
        """Load the particle data from the Gadget-2 binary file.

        Parameters
        ----------
        inplace : bool
            If ``True``, the data of this instance will be automatically updated.

        Returns
        -------
        ClusterParticles
            The resulting particle dataset of the gadget file.
        """
        fields = OrderedDict()

        with FortranFile(self.path, "r") as f:
            # Skip the header
            head = self.__class__._read_block_specifier(f)

            assert head == "HEAD", f"Binary file {self.path} doesn't start with 'HEAD'."

            _ = f.read_record(self.__class__._get_header_struct())

            while True:
                try:
                    # Each following entry should have a 4-char leader to tell us what we're reading.
                    block_name = self.__class__._read_block_specifier(f)

                    # We now use the block name to get the correct struct for the data that's coming
                    # down the pipe.
                    struct_dtype = self._get_block_struct(block_name)
                    # This should manage things like custom dtypes, and specifications for the
                    # number of particles of each type.

                    struct_data = f.read_record(struct_dtype)

                    # Figure out what our block is actually carrying. We look up the HDF5 key, but if a
                    # block doesn't have a match, we just use the block name.
                    hdf5_key_name = self.__class__._block_names.get(
                        block_name, block_name
                    )
                    cg_field_name = _GADGET_FIELD_MAP.get(hdf5_key_name, hdf5_key_name)

                    for ptype in struct_dtype.names:
                        cg_particle_name = _GADGET_PTYPE_MAP[ptype]

                        if cg_particle_name is None:
                            # We skip this because we don't recognize the particle type.
                            continue
                        else:
                            # We write
                            fields[(cg_particle_name, cg_field_name)] = unyt.unyt_array(
                                struct_data[ptype],
                                units=self.units.get(hdf5_key_name, ""),
                            )
                except (EOFError, IOError):
                    # We hit the end of the file.
                    break

        particle_types = [k[0] for k in fields.keys()]
        particles = ClusterParticles(particle_types, fields)

        if inplace:
            self._particles = particles

        return particles

    @classmethod
    def from_particles(
        cls, particles: ClusterParticles, path: str | Path, *args, **kwargs
    ) -> Self:
        """Create a :py:class:`Gadget2Binary` instance and file from a
        :py:class:`ClusterParticles` object.

        Parameters
        ----------
        particles : ClusterParticles
            The particle data to write to the binary file.
        path : str or Path
            The path where the binary file will be created.
        kwargs:
            Additional kwargs that are used to fill in the header and details about the
            included particles.

        Returns
        -------
        Gadget2Binary
            An instance of Gadget2Binary initialized from the particle data.
        """
        path = Path(path)

        # Setup the header from the provided kwargs.
        header, kwargs = cls.generate_header(kwargs)

        # Initialize NumPart_ThisFile and MassTable
        header["NPart"] = np.array(
            [particles.num_particles.get(_GADGET_PTYPE_MAP[i], 0) for i in range(6)],
            dtype="u4",
        )
        header["Nall"] = np.array(
            [particles.num_particles.get(_GADGET_PTYPE_MAP[i], 0) for i in range(6)],
            dtype="u4",
        )

        # MassTable is default 0 for all, but if the particles have homogeneous mass, it should be set.
        header["Massarr"] = np.zeros(6, dtype="f8")

        # Write the header to the fortran format.
        with FortranFile(path, "w") as f:
            # Write the header using the structured dtype
            header_struct = np.zeros(1, dtype=cls._get_header_struct())
            for hkey in header_struct.dtype.names:
                header_struct[hkey] = header[hkey]

            # Write the header to the binary file
            f.write_record(header_struct)

            # Write each of the blocks.
            for block, hdf5_key in cls._block_names.items():
                if block == "MASS" and np.all(header["MassTable"] > 0):
                    # Skip MASS block if all masses are set in MassTable
                    continue

                # Generate the structs for the block and for the block string.
                struct_dtype = cls._get_block_struct_from_header(
                    block, header, cls._block_dtypes
                )

                # Skip writing this block if there are no particles for this block
                if not any(
                    struct_dtype[ptype].shape[0] > 0 for ptype in struct_dtype.names
                ):
                    continue

                # Prepare the data to write by combining particle types into one struct
                struct_data = np.zeros(1, dtype=struct_dtype)

                for ptype in struct_dtype.names:
                    cg_particle_name = _GADGET_PTYPE_MAP[ptype]
                    cg_field_name = _GADGET_FIELD_MAP.get(hdf5_key, hdf5_key)

                    if (
                        cg_particle_name
                        and (cg_particle_name, cg_field_name) in particles.fields
                    ):
                        struct_data[ptype] = particles.fields[
                            (cg_particle_name, cg_field_name)
                        ].d

                # Write the block specifier
                f.write_record(np.array(block, dtype="S4"))

                # Write the structured data
                f.write_record(struct_data)

        # Return the instance
        instance = cls(path)
        return instance
