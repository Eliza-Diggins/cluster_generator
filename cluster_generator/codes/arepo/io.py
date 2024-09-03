"""IO utilities for AREPO.

Notes
-----

Because AREPO is a descendant of GADGET-2, the IO needs are very similar and can largely be ported over directly.
"""
from abc import ABC
from collections import OrderedDict
from typing import ClassVar

import numpy as np
from pydantic import model_validator

from cluster_generator.codes.gadget.io import _GADGET_FIELDS, _GADGET_PARTICLES
from cluster_generator.codes.io.gadget import (
    GenericGadget2_Binary,
    GenericGadget2_FieldRegistry,
    GenericGadget2_HDF5,
    GenericGadget2_Header,
    GenericGadget2_ICFile,
    GenericGadget2_ParticleRegistry,
)

# Arepo uses almost identical conventions for particle type, headers, etc.
# We create the PARTICLES, FIELDS, and HEADERS so that they have conventional names in this
# namespace, but they are effectively identical.
_AREPO_PARTICLES = _GADGET_PARTICLES.copy()
_AREPO_FIELDS = _GADGET_FIELDS.copy()
_AREPO_HEADERS = {
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
    "FlagDoublePrecision": "Flag_DoublePrecision",
}

# Default float type for Gadget files
_AREPO_DEFAULT_FLOAT_TYPE: str = "f4"
# Standard group prefix in HDF5.
_AREPO_GROUP_PREFIX: str = "PartType"


class Arepo_Header(GenericGadget2_Header):
    HEADER_FLOAT_TYPE: ClassVar[str | np.dtype] = np.float64
    HEADER_UINT_TYPE: ClassVar[str | np.dtype] = np.uint32
    HEADER_INT_TYPE: ClassVar[str | np.dtype] = np.int32
    HEADER_ORDER: ClassVar[list[str]] = list(_AREPO_HEADERS.keys())
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


Arepo_Header.BINARY_TO_HDF5 = Arepo_Header.generate_field_mapping()
Arepo_Header.HDF5_TO_BINARY = {v: k for k, v in Arepo_Header.BINARY_TO_HDF5.items()}


class Arepo_ParticleRegistry(GenericGadget2_ParticleRegistry):
    DEFAULT_PARTICLE_MAP = _AREPO_PARTICLES.copy()


class Arepo_FieldRegistry(GenericGadget2_FieldRegistry):
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
            for key, value in _AREPO_FIELDS.items()
        }
    )


class Arepo_ICFile(GenericGadget2_ICFile, ABC):
    pass


class Arepo_HDF5(GenericGadget2_HDF5):
    _format = "hdf5"
    _group_prefix = _AREPO_GROUP_PREFIX


class Arepo_Binary(GenericGadget2_Binary):
    _format = "binary"
