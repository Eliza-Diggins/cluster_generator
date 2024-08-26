"""Frontend tools for interacting with the Gadget-2 hydrodynamics code.

This module provides the `Gadget2` class and associated runtime parameters needed to
run simulations with the Gadget-2 code. It includes functionalities for setting up
initial conditions, managing runtime parameters, and interacting with the Gadget-2
simulation software.

- **Runtime Parameters Management**: This module provides a structured way to define and manage
  runtime parameters for Gadget-2 simulations. The `Gadget2RuntimeParameters` class allows for the
  specification of various simulation parameters, including output settings, box size,
  softening lengths, and more.

- **Initial Conditions Setup**: The `Gadget2` class includes methods for generating initial
  conditions compatible with Gadget-2, including support for different initial conditions formats.

- **File Formats**: Gadget-2 supports several file formats for input and output, including:
  - **Binary Format**: A simple binary format used to store data, compatible with earlier versions of the Gadget code.
  - **HDF5 Format**: A more modern and versatile format that uses the HDF5 file format standard for storing large amounts of data. This format allows for efficient storage and retrieval of large simulation datasets and supports metadata inclusion. It is the preferred format for most modern simulations due to its flexibility and performance.

- **Initial Conditions (IC) Files**: The initial state of the simulation, which includes
  positions, velocities, masses, internal energies, and other relevant physical quantities
  of the simulation particles.
- **Parameter Files**: Text files that specify the runtime parameters needed by Gadget-2 to
  execute the simulation, such as the number of particles, time steps, and physical constants.

For more details on Gadget-2 file formats and runtime configurations, refer to the official
Gadget-2 documentation:
- Gadget-2 Documentation: http://www.mpa-garching.mpg.de/gadget/
- Gadget-2 GitHub Repository: https://github.com/springel/gadget

See Also
--------
:py:class:`cluster_generator.codes.abc.RuntimeParameters`
:py:class:`cluster_generator.codes.abc.SimulationCode`
"""

import os
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, ClassVar, Literal, Type

import numpy as np
import unyt
from typing_extensions import Self
from unyt import unyt_array, unyt_quantity

from cluster_generator.codes.abc import RuntimeParameters, SimulationCode
from cluster_generator.codes.utils import _const_factory, cfield, ufield
from cluster_generator.ics import ClusterICs
from cluster_generator.utilities.config import config_directory
from cluster_generator.utilities.logging import LogDescriptor
from cluster_generator.utilities.types import Instance


class Gadget2RuntimeParameters(RuntimeParameters):
    """Class for handling runtime parameters specific to Gadget-2."""

    @staticmethod
    def set_InitCondFile(instance: Instance, _: Type[Instance], __: ClusterICs) -> str:
        """Set the initial condition file path for Gadget-2."""
        return str(Path(instance.IC_PATH).absolute())

    @staticmethod
    def set_OutputListOn(instance: Instance, _: Type[Instance], __: ClusterICs) -> int:
        """Set the output list flag for Gadget-2."""
        return 1 if isinstance(instance.OUTPUT_STYLE, unyt_array) else 0

    @staticmethod
    def set_OutputListFilename(
        instance: Instance, _: Type[Instance], __: ClusterICs
    ) -> str:
        """Set the output list filename for Gadget-2."""
        if isinstance(instance.OUTPUT_STYLE, unyt_array):
            return os.path.join(instance.OUTPUT_DIR, "output_list.txt")
        return ""

    @staticmethod
    def set_TimeBetSnapshot(
        instance: Instance, _: Type[Instance], __: ClusterICs
    ) -> float | str:
        if isinstance(instance.OUTPUT_STYLE, unyt_array):
            # We aren't using this, but AREPO still wants the flag to be present.
            return 0
        else:
            # we have a tuple, we want to use it.
            return instance.OUTPUT_STYLE[1].in_base(instance.unit_system).value

    @staticmethod
    def set_TimeOfFirstSnapshot(
        instance: Instance, _: Type[Instance], __: ClusterICs
    ) -> float | int:
        if isinstance(instance.OUTPUT_STYLE, unyt_array):
            return 0  # This flag is ignored because the output list is on.
        else:
            # we have a tuple, we want to use it.
            return instance.OUTPUT_STYLE[0].in_base(instance.unit_system).value

    @staticmethod
    def set_BoxSize(instance: Instance, _: Type[Instance], ic: ClusterICs) -> float:
        if instance.BOXSIZE is not None:
            return instance.BOXSIZE.to_value(instance.LENGTH_UNIT)
        else:
            # Deduce correct boxsize from the provided ICs.
            return (
                unyt_quantity(2 * ic.r_max, "kpc").in_base(instance.unit_system).value
            )

    @staticmethod
    def set_TimeBetStatistics(
        instance: Instance, _: Type[Instance], __: ClusterICs
    ) -> float:
        # We want to match the output frequency.
        if isinstance(instance.OUTPUT_STYLE, unyt_array):
            # This is an unyt array, we take the minimum separation
            _tdiff = np.amax(
                (instance.OUTPUT_STYLE[1:] - instance.OUTPUT_STYLE[:-1])
                .in_base(instance.unit_system)
                .d
            )
        else:
            _tdiff = instance.OUTPUT_STYLE[1].in_base(instance.unit_system).d

        return _tdiff

    def deduce_instance_values(
        self, instance: Instance, owner: Type[Instance], ics: ClusterICs
    ) -> tuple[bool, Any]:
        """Deduce runtime parameter values for a Gadget-2 simulation instance."""
        _, errors = super().deduce_instance_values(instance, owner, ics)

        if errors is None:
            errors = []

        # Additional handling for Gadget-2 specific parameters can be added here

        if errors:
            from cluster_generator.utilities.logging import ErrorGroup

            return False, ErrorGroup(
                f"Failed to set RTPs for {instance}!", error_list=errors
            )
        return True, None

    def write_rtp_template(
        self,
        instance: Instance,
        owner: Type[Instance],
        path: str | Path,
        overwrite: bool = False,
    ) -> Path:
        """Write the Gadget-2 runtime parameters file.

        Parameters
        ----------
        instance : Gadget2
            The Gadget2 instance to write the parameter file for.
        owner : Type[Gadget2]
            The owner class.
        path : str or Path
            The output location.
        overwrite : bool, optional
            Allow method to overwrite an existing file at this path. Default is ``False``.

        Returns
        -------
        Path
            The path to the written parameter file.
        """
        instance.logger.info(f"Generating Gadget-2 RTP template for {instance}.")

        path = Path(path)
        if path.exists() and not overwrite:
            raise ValueError(
                f"Gadget-2 RTP template already exists at {path}. Overwrite=False."
            )
        elif path.exists() and overwrite:
            path.unlink()

        with open(path, "w") as file:
            # Write parameters to file
            for param, value in instance.__dict__.items():
                file.write(f"{param} = {value}\n")

        return path


@dataclass
class Gadget(SimulationCode):
    """:py:class:`SimulationCode` class for interacting with the Gadget-2 simulation
    software."""

    rtp: ClassVar[Gadget2RuntimeParameters] = Gadget2RuntimeParameters(
        os.path.join(
            Path(config_directory).parents[0],
            "codes",
            "gadget",
            "runtime_parameters.yaml",
        )
    )  # Path to the correct code bin in cluster_generator.

    logger: ClassVar[Logger] = LogDescriptor()
    """Logger: The class logger for this code."""

    # USER PARAMETERS
    IC_PATH: str = ufield()
    """str: The path to the initial conditions file."""

    OUTPUT_STYLE: tuple[unyt_quantity, unyt_quantity] | unyt_array = ufield()
    """ tuple of unyt_quantity or unyt_array: The frequency of snapshot outputs.

    If a tuple is provided, the first value is the time of first output and the second is the time between remaining snapshots.
    If an array of times is provided, then they will be used for the output times.
    """

    TIME_MAX: unyt_quantity = ufield(flag="TimeMax")
    """unyt_quantity: The maximum (simulation) time at which to end the simulation."""

    OUTPUT_DIR: str = ufield(default="./output", flag="OutputDir")
    """str: The directory in which to generate the simulation output. Default is ``./output``."""

    IC_FORMAT: Literal[1, 2] = ufield(default=2, flag="ICFormat", av=[2])
    """int: The format of the initial conditions file. Default is 2 (binary format)."""

    SNAP_FORMAT: Literal[1, 2] = ufield(default=2, flag="SnapFormat")
    """int: The format of the snapshot outputs. Default is 2 (binary format)."""

    VELOCITY_UNIT: unyt_quantity = ufield(
        default_factory=_const_factory(unyt_quantity(1.0, "km/s")),
        flag="UnitVelocity_in_cm_per_s",
        setter=lambda _inst, _, __: _inst.VELOCITY_UNIT.to_value("cm/s"),
    )
    """unyt_quantity: The code-velocity unit. Default is 1 km / s."""

    LENGTH_UNIT: unyt_quantity = ufield(
        default_factory=_const_factory(unyt_quantity(1.0, "kpc")),
        flag="UnitLength_in_cm",
        setter=lambda _inst, _, __: _inst.LENGTH_UNIT.to_value("cm"),
    )
    """unyt_quantity: The code-length unit. Default is 1 kpc."""

    MASS_UNIT: unyt_quantity = ufield(
        default_factory=_const_factory(unyt_quantity(1.0, "Msun")),
        flag="UnitMass_in_g",
        setter=lambda _inst, _, __: _inst.MASS_UNIT.to_value("g"),
    )
    """unyt_quantity: The code-mass unit. Default is 1 Msun."""
    BOXSIZE: unyt_quantity = ufield(default=None)
    """ unyt_quantity: The size of the simulation's bounding box.

    If this is left unspecified, then it will be taken from the IC's :py:attr:`cluster_generator.ics.ClusterICs.r_max` parameter
    (multiplied by a factor of 2). In most cases, this is a considerable overestimate.
    """
    START_TIME: unyt_quantity = ufield(
        default_factory=_const_factory(unyt_quantity(0.0, "Gyr")),
        flag="TimeBegin",
        setter=lambda _inst, _, __: _inst.START_TIME.to_value(
            _inst.LENGTH_UNIT / _inst.VELOCITY_UNIT
        ),
    )
    """ unyt_quantity: The start time of the simulation.

    Default is 0.0 Gyr.
    """
    END_TIME: unyt_quantity = ufield(
        default_factory=_const_factory(unyt_quantity(10.0, "Gyr")),
        flag="TimeBegin",
        setter=lambda _inst, _, __: _inst.START_TIME.to_value(
            _inst.LENGTH_UNIT / _inst.VELOCITY_UNIT
        ),
    )
    """ unyt_quantity: The end time of the simulation.

    Default is 10 Gyrs.
    """
    HAVE_HDF5: bool = cfield(default=False, av=[True])
    """ bool: If ``True``, then input and output can be in HDF5."""
    ISOTHERM_EQS: bool = cfield(default=False, av=[False])
    """ bool: Gas is treated as isothermal.

    .. warning::

        Cluster generator does not support this flag.

    """
    ADAPTIVE_GRAVSOFT_FORGAS: bool = cfield(default=False)
    """ bool: Allow the softening length for fluid cells to vary with cell size."""

    def __post_init__(self):
        super().__post_init__()

    def generate_ics(
        self, initial_conditions: ClusterICs, overwrite: bool = False, **kwargs
    ) -> Path:
        """Generate initial conditions file for Gadget-2.

        Parameters
        ----------
        initial_conditions : ClusterICs
            The initial conditions for the simulation.
        overwrite : bool, optional
            Whether to overwrite the existing initial conditions file. Default is ``False``.

        Returns
        -------
        Path
            The path to the generated initial conditions file.
        """
        from cluster_generator.codes.gadget.io import (
            Gadget2_FieldRegistry,
            Gadget2_HDF5,
            Gadget2_Header,
            Gadget2_ParticleRegistry,
        )

        # Create a shared particle dataset from the initial conditions
        # we do this in ./arepo_tmp to contain any particle files generated.
        self.logger.info(f"Constructing AREPO ICs from {initial_conditions}.")

        initial_conditions.directory = "./arepo_tmp"
        self.logger.debug(f"\tIC dump directory: {Path(initial_conditions.directory)}.")

        self.logger.info("Combining constituent models and generating particles...")
        combined_particles = initial_conditions.setup_particle_ics(**kwargs)
        self.logger.info("Combining constituent models and generating particles [DONE]")

        # Produce compliant HDF5 file
        if self.IC_FORMAT == 3:
            # Create the field registry. For initial conditions, we only need blocks up to the temperature.
            # The dtypes will depend on the INPUT_IN_DOUBLEPRECISION flag.
            # The units will also depend on the unit system of the Arepo instance.
            fields = {
                "Coordinates": (
                    "particle_position",
                    "POS",
                    self.unit_system["length"],
                    "f4",
                    None,
                    3,
                ),
                "Velocities": (
                    "particle_velocity",
                    "VEL",
                    self.unit_system["length"] / self.unit_system["time"],
                    "f4",
                    None,
                    3,
                ),
                "ParticleIDs": ("particle_index", "ID", None, "i4", None, 1),
                "Masses": (
                    "particle_mass",
                    "MASS",
                    self.unit_system["mass"],
                    "f4",
                    None,
                    1,
                ),
                "InternalEnergy": (
                    "thermal_energy",
                    "U",
                    (self.unit_system["length"] / self.unit_system["time"]) ** 2,
                    "f4",
                    [0],
                    1,
                ),
            }
            fields = {
                key: {
                    "cluster_name": value[0],
                    "binary_name": value[1],
                    "units": value[2],
                    "dtype": value[3],
                    "pid": value[4],
                    "size": value[5],
                }
                for key, value in fields.items()
            }

            field_registry = Gadget2_FieldRegistry(fields)

            # Now we deal with the particle registry. These should all be by convention, so we just generate
            # the default.
            particle_registry = Gadget2_ParticleRegistry({})

            # The header now needs to be produced. We generate it from the particle dataset with all of the header
            # attributes as kwargs.
            _header_dict = {}

            _header_dict["Time"] = self.START_TIME.in_base(self.unit_system).d

            # We don't allow cosmological evolution, so these parameters can simply be set to their defaults.
            # Setting header flags. These are all pulled from RTPs.
            _header_dict["Flag_Sfr"] = self.rtp["StarformationOn"]
            _header_dict["Flag_Cooling"] = self.rtp["CoolingOn"]
            _header_dict["Flag_Feedback"] = self.rtp["StarformationOn"]
            _header_dict["Flag_Metals"] = 0.0
            _header_dict["Flag_StellarAge"] = 0.0

            header = Gadget2_Header.from_particles(
                combined_particles, particle_registry, **_header_dict
            )

            self.logger.info("Writing particles to GADGET-2 HDF5...")
            ICS = Gadget2_HDF5.from_particles(
                combined_particles,
                self.IC_PATH,
                field_registry=field_registry,
                particle_registry=particle_registry,
                header=header,
                overwrite=overwrite,
            )
            self.logger.info("Writing particles to AREPO HDF5. [DONE]")
            return ICS
        else:
            raise ValueError(f"IC_FORMAT={self.IC_FORMAT} is not currently supported.")

    @property
    def unit_system(self) -> unyt.UnitSystem:
        if self._unit_system is None:
            self._unit_system = unyt.UnitSystem(
                "GADGET2",
                unyt.Unit(self.LENGTH_UNIT),
                unyt.Unit(self.MASS_UNIT),
                unyt.Unit(self.LENGTH_UNIT / self.VELOCITY_UNIT),
                "K",
                "rad",
            )
        return self._unit_system

    @classmethod
    def from_install_directory(
        cls, installation_directory: str | Path, **parameters
    ) -> Self:
        """Determine the relevant class parameters for Gadget-2 from its installation
        directory.

        Parameters
        ----------
        installation_directory : str or Path
            The directory in which Gadget-2 is installed.
        parameters :
            Other parameters to pass to the initializer for :py:class:`Gadget2`.

        Returns
        -------
        Gadget2
            An instance of the Gadget-2 code class.

        Notes
        -----
        This method seeks out the ``Config.sh`` file in the installation directory and reads it to determine the
        enabled and disabled flags.
        """
        raise NotImplementedError()
