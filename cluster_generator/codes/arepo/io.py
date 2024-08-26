"""IO utilities for AREPO.

Notes
-----

Because AREPO is a descendant of GADGET-2, the IO needs are very similar and can largely be ported over directly.
"""
from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from cluster_generator.codes.gadget.io import (
    _GADGET_FIELDS,
    _GADGET_PARTICLES,
    Gadget2_FieldRegistry,
    Gadget2_HDF5,
    Gadget2_Header,
    Gadget2_ICFile,
    Gadget2_ParticleRegistry,
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


@dataclass
class Arepo_Header(Gadget2_Header):
    HEADER_ORDER: ClassVar[list[str]] = list(_AREPO_HEADERS.keys())
    Flag_DoublePrecision: Gadget2_Header.HEADER_INT_TYPE = Gadget2_Header.HEADER_FIELD(
        "i", "FlagDoublePrecision", default=0
    )

    # Generate HDF5 to binary and binary to HDF5 name mappings from dataclass fields
    HDF5_TO_BINARY: ClassVar[dict[str, str]] = {}
    """Dict[str, str]: A mapping from HDF5 field names to their corresponding binary
    field names."""

    BINARY_TO_HDF5: ClassVar[dict[str, str]] = {}
    """Dict[str, str]: A mapping from binary field names to their corresponding HDF5
    field names."""


Arepo_Header.BINARY_TO_HDF5 = Arepo_Header.generate_field_mapping()
Arepo_Header.HDF5_TO_BINARY = {v: k for k, v in Arepo_Header.BINARY_TO_HDF5.items()}


class Arepo_ParticleRegistry(Gadget2_ParticleRegistry):
    DEFAULT_PARTICLE_MAP: dict[int, str | None] = _AREPO_PARTICLES.copy()


class Arepo_FieldRegistry(Gadget2_FieldRegistry):
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
            for key, value in _AREPO_FIELDS.items()
        }
    )


class Arepo_ICFile(Gadget2_ICFile, ABC):
    _format: Literal[
        "hdf5", "binary"
    ] = None  # Placeholder for the format to be defined by subclasses


class Arepo_HDF5(Gadget2_HDF5, Arepo_ICFile):
    _group_prefix = "PartType"

    def __init__(self, *args, **kwargs):
        super(Arepo_HDF5, self).__init__(*args, **kwargs)

        self._group_prefix = "PartType"


def read_ctps(install_directory: str | Path):
    """Read the compile time flags from a given installation directory.

    Parameters
    ----------
    install_directory: str
        The path to the directory where AREPO is installed.

    Returns
    -------
    dict
        The compile-time flags from the installation.
    """
    import os

    path = Path(install_directory)
    assert path.exists(), f"The installation directory {path} doesn't exist!"

    config_path = Path(os.path.join(path, "Config.sh"))
    assert (
        config_path.exists()
    ), f"The compile-time flag file ({config_path}) doesn't exist."

    # Read the CTP file from disk
    with open(config_path, "r") as flag_io:
        _raw = flag_io.read()

    # Remove all of the commented lines and blank lines. This should remove comments and formatting
    _raw = [j for j in _raw.split("\n") if len(j) and j[0] != "#"]
    # Breakdown into just the provided values
    _raw = [j.split(" ")[0] for j in _raw]

    CTPs = {}

    for _ctp in _raw:
        if "=" not in _ctp:
            CTPs[_ctp] = True
        else:
            CTPs[_ctp.split("=")[0]] = int(_ctp.split("=")[1])

    return CTPs
