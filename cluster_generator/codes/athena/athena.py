from logging import Logger
from typing import ClassVar

from cluster_generator.codes.abc import RuntimeParameters, SimulationCode
from cluster_generator.utilities.logging import LogDescriptor


class AthenaRuntimeParameters(RuntimeParameters):
    pass


class Athena(SimulationCode):
    logger: ClassVar[Logger] = LogDescriptor()
    """ Logger: The class logger for this code."""
    pass
