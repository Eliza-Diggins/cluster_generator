import os
from logging import Logger
from pathlib import Path
from typing import ClassVar

from cluster_generator.codes.abc import RuntimeParameters, SimulationCode
from cluster_generator.codes.utils import cfield
from cluster_generator.utilities.config import config_directory
from cluster_generator.utilities.logging import LogDescriptor


class AthenaRuntimeParameters(RuntimeParameters):
    pass


class Athena(SimulationCode):
    rtp: ClassVar[AthenaRuntimeParameters] = AthenaRuntimeParameters(
        os.path.join(
            Path(config_directory).parents[0],
            "codes",
            "athena",
            "runtime_parameters.yaml",
        )
    )  # Path to the correct code bin in cluster_generator.

    logger: ClassVar[Logger] = LogDescriptor()
    """ Logger: The class logger for this code."""

    # == USER PARAMETERS == #

    # == COMPILE-TIME PARAMETERS == #
    prob: str = cfield(default=None)
    """ str: The problem initializer.

    """
    eos: str = cfield(default="adiabatic", av=["isothermal", "adiabatic"])
    """ str: The equation of state for the simulation.

    Value may be ``"adiabatic"``, ``"isothermal"`` or a relative path from ``src/eos/`` pointing to a
    custom EOS implementation.

    .. warning::

        We currently only support ``"adiabatic"`` and ``"isothermal"`` EOS.

    """
    coord: str = cfield(default="cartesian", av=["cartesian"])
    """ str: The coordinate system to use at compile-time.

    This must be cartesian to be compatible with cluster generator ICs.
    """
    nscalars: int = cfield(default=0)
    """ int: The number of passive scalars to allow."""
    b: bool = cfield(default=False, av=[False])
    """ bool: If ``True``, then magnetic fields are enabled.

    .. warning::

        We currently don't support this option.

    """
    s: bool = cfield(default=False, av=[False])
    """ bool: Enable special relativity.

    .. warning::

        We currently don't support this option.
    """
    g: bool = cfield(default=False, av=[False])
    """ bool: Enable general relativity.

    .. warning::

        We currently don't support this option.
    """
    t: bool = cfield(default=False, av=[False])
    """ bool: Enable interface frame transformation in GR.

    .. warning::

        We currently don't support this option.

    """
    hdf5: bool = cfield(default=False)
    """ bool: If true, HDF5 is allowed output type."""
    sts: bool = cfield(default=False)
    """ bool: If ``True``, then super-time-stepping is enabled."""
