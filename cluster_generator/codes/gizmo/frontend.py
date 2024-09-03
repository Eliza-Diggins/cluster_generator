from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import unyt

from cluster_generator import ClusterICs
from cluster_generator.codes._abc import CodeFrontend, MissingRTP
from cluster_generator.utils import LogDescriptor

if TYPE_CHECKING:
    from logging import Logger


class Gizmo(CodeFrontend):
    rtp: ClassVar[MissingRTP] = MissingRTP()
    logger: ClassVar["Logger"] = LogDescriptor

    def generate_ics(
        self, initial_conditions: ClusterICs, overwrite: bool = False, **kwargs
    ) -> Path:
        pass

    def unit_system(self) -> unyt.UnitSystem:
        pass
