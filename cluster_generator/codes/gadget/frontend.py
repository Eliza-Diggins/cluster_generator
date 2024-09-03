import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import unyt

from cluster_generator import ClusterICs
from cluster_generator.codes._abc import CodeFrontend, _MissingRTP
from cluster_generator.codes._types import UField, UnytQuantity
from cluster_generator.utils import LogDescriptor, prepare_path

if TYPE_CHECKING:
    from logging import Logger

from cluster_generator.codes.gadget.io import (
    Gadget2_Binary,
    Gadget2_FieldRegistry,
    Gadget2_HDF5,
    Gadget2_Header,
    Gadget2_ICFile,
    Gadget2_ParticleRegistry,
)


class Gadget(CodeFrontend):
    rtp: ClassVar[_MissingRTP] = _MissingRTP()
    logger: ClassVar["Logger"] = LogDescriptor

    # USER PARAMETERS
    IC_PATH: str = UField()
    FORMAT: Literal["binary", "hdf5"] = UField(default="hdf5")
    VELOCITY_UNIT: UnytQuantity[float, unyt.Unit("cm/s")] = UField(
        default="1.0 km/s",
        flag="UnitVelocity_in_cm_per_s",
        setter=lambda _inst, _, __: _inst.VELOCITY_UNIT.to_value("cm/s"),
        validate_default=True,
    )
    LENGTH_UNIT: UnytQuantity[float, unyt.Unit("cm")] = UField(
        default="1.0 kpc",
        flag="UnitLength_in_cm",
        setter=lambda _inst, _, __: _inst.LENGTH_UNIT.to_value("cm"),
        validate_default=True,
    )
    MASS_UNIT: UnytQuantity[float, unyt.Unit("cm")] = UField(
        default="1.0 Msun",
        flag="UnitMass_in_g",
        setter=lambda _inst, _, __: _inst.MASS_UNIT.to_value("g"),
        validate_default=True,
    )

    def generate_ics(
        self, initial_conditions: ClusterICs, overwrite: bool = False, **kwargs
    ) -> Gadget2_ICFile:
        self.logger.info("Generating Gadget2 ICs from %s.", initial_conditions)

        # Generate the particle dataset for the initial conditions.
        # This should be stored in a temporary directory so that we don't clutter the file system
        # unless the user specifies otherwise.
        products_directory: str | None | Path = kwargs.pop("products_directory", None)
        if products_directory in ["tmp", "temp", "temporary"]:
            _temp_dir_flag = True
            _temp_dir = tempfile.TemporaryDirectory()
            products_directory = Path(_temp_dir.name)

            initial_conditions.directory = products_directory
        elif products_directory is not None:
            _temp_dir_flag = False
            _temp_dir = None
            products_directory = Path(products_directory)
            products_directory.mkdir(exist_ok=True, parents=True)

            initial_conditions.directory = products_directory
        else:
            _temp_dir_flag = (False,)
            _temp_dir = None
            pass

        try:
            self.logger.debug(
                "\tGenerating a particle dataset from %s.", initial_conditions
            )
            particles = initial_conditions.setup_particle_ics()

            # The particles need to have a particle index field added to them.
            particles.add_index()

            # Setup the output path, ensure the directory exists, etc.
            _ = prepare_path(self.IC_PATH, overwrite=overwrite)

            # Now we generate the initial conditions.
            if self.FORMAT == "binary":
                field_class = Gadget2_Binary
            else:
                field_class = Gadget2_HDF5

            # Either grab or generate default particle and field registries for
            # constructing the HDF5 / G2 file underlying everything.
            particle_registry, field_registry = kwargs.pop(
                "particle_registry", Gadget2_ParticleRegistry({})
            ), kwargs.pop("field_registry", Gadget2_FieldRegistry({}))

            # setup the header
            header = Gadget2_Header.from_particles(
                particles, particle_registry, **kwargs
            )

            gadget_file = field_class.from_particles(
                particles,
                self.IC_PATH,
                particle_registry=particle_registry,
                field_registry=field_registry,
                header=header,
                overwrite=overwrite,
            )

        finally:
            if _temp_dir_flag:
                _temp_dir.cleanup()

        return gadget_file

    @property
    def unit_system(self) -> unyt.UnitSystem:
        """Returns the unit system for the simulation.

        Returns
        -------
        unyt.UnitSystem
            The unit system used for the simulation.

        Examples
        --------
        .. code-block:: python

            arepo = Arepo(IC_PATH="path/to/ic")
            unit_system = arepo.unit_system
        """
        if self._unit_system is None:
            # construct the unit system from scratch
            self._unit_system = unyt.UnitSystem(
                "AREPO",
                unyt.Unit(self.LENGTH_UNIT),
                unyt.Unit(self.MASS_UNIT),
                unyt.Unit(self.LENGTH_UNIT / self.VELOCITY_UNIT),
                "K",
                "rad",
            )
        return self._unit_system
