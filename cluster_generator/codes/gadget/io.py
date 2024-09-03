"""Module for handling Gadget file types for initial conditions.

.. rubric:: Developer Notes

- This module provides a comprehensive implementation of standard Gadget-2 file formats for initial conditions.
  The module allows for configurable behaviors and recognized fields, enabling easy extension for descendant file classes.
"""

from abc import ABC
from collections import OrderedDict
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from pydantic import model_validator

from cluster_generator.codes.io.gadget import (
    GenericGadget2_Binary,
    GenericGadget2_FieldRegistry,
    GenericGadget2_HDF5,
    GenericGadget2_Header,
    GenericGadget2_ICFile,
    GenericGadget2_ParticleRegistry,
)
from cluster_generator.utils import devLogger

devlog = devLogger

if TYPE_CHECKING:
    pass

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


class Gadget2_Header(GenericGadget2_Header):
    HEADER_FLOAT_TYPE: ClassVar[str | np.dtype] = np.float64
    HEADER_UINT_TYPE: ClassVar[str | np.dtype] = np.uint32
    HEADER_INT_TYPE: ClassVar[str | np.dtype] = np.int32
    HEADER_ORDER: ClassVar[list[str]] = list(_GADGET_HEADERS.keys())
    HEADER_SIZE: ClassVar[int] = 256
    N_PARTICLE_TYPES: ClassVar[int] = 6

    # Generate HDF5 to binary and binary to HDF5 name mappings from dataclass fields
    HDF5_TO_BINARY: ClassVar[dict[str, str]] = {}
    """Dict[str, str]: A mapping from HDF5 field names to their corresponding binary
    field names."""

    BINARY_TO_HDF5: ClassVar[dict[str, str]] = {}
    """Dict[str, str]: A mapping from binary field names to their corresponding HDF5
    field names."""

    @model_validator(mode="after")
    def unit_validation(cls, value):
        return GenericGadget2_Header.unit_validation(cls, value)


Gadget2_Header.BINARY_TO_HDF5 = Gadget2_Header.generate_field_mapping()
Gadget2_Header.HDF5_TO_BINARY = {v: k for k, v in Gadget2_Header.BINARY_TO_HDF5.items()}


class Gadget2_ParticleRegistry(GenericGadget2_ParticleRegistry):
    DEFAULT_PARTICLE_MAP = _GADGET_PARTICLES.copy()


class Gadget2_FieldRegistry(GenericGadget2_FieldRegistry):
    DEFAULT_FIELD_MAP = OrderedDict(
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


class Gadget2_ICFile(GenericGadget2_ICFile, ABC):
    pass


class Gadget2_HDF5(GenericGadget2_HDF5):
    _format = "hdf5"
    _group_prefix = _GADGET_GROUP_PREFIX


class Gadget2_Binary(GenericGadget2_Binary):
    _format = "binary"
