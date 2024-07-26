"""Frontend tools for interacting with the Arepo hydrodynamics code."""
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


class ArepoRuntimeParameters(RuntimeParameters):
    @staticmethod
    def set_InitCondFile(instance: Instance, _: Type[Instance], __: ClusterICs) -> str:
        # The initial condition file needs to be set to the absolute path.
        # The suffix needs to be removed as Arepo adds this back on by default.
        return str(Path(instance.IC_PATH).absolute().with_suffix(""))

    @staticmethod
    def set_OutputListOn(instance: Instance, _: Type[Instance], __: ClusterICs) -> int:
        if isinstance(instance.OUTPUT_STYLE, unyt_array):
            return 1
        else:
            return 0

    @staticmethod
    def set_OutputListFilename(
        instance: Instance, _: Type[Instance], __: ClusterICs
    ) -> str:
        if isinstance(instance.OUTPUT_STYLE, unyt_array):
            return os.path.join(instance.OUTPUT_DIR, "tout_list.txt")
        else:
            # We aren't using this, but AREPO still wants the flag to be present.
            return "0"

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
        # Run the standard method -> get everything that's specified normally.
        _, errors = super().deduce_instance_values(instance, owner, ics)

        if errors is None:
            errors = []

        # MANAGING SOFTENING
        # ------------------
        # Because softening parameters are adaptive to NTYPES and NTYPESSOFT, we have to add them dynamically.
        #
        # This section of code determines the fill-in values for the softening lengths and applies them.
        for particle_type in range(instance.NTYPES):
            # iterate through each type of particle and assign the softening type to the particle.
            instance.rtp[
                f"SofteningTypeOfPartType{particle_type}"
            ] = instance.SOFTENING_TYPES.get(particle_type, 1)
            instance.logger.debug(
                f"\tSofteningTypeOfPartType{particle_type} -> {instance.rtp[f'SofteningTypeOfPartType{particle_type}']}"
            )

        # Assign the softening lengths. Starting with the comoving softening length.
        for softening_type in range(instance.NSOFTTYPES):
            try:
                instance.rtp[f"SofteningComovingType{softening_type}"] = (
                    instance.SOFTENING_COMOVING[softening_type]
                    .in_base(instance.unit_system)
                    .d
                )
                instance.logger.debug(
                    f"\tSofteningComovingType{softening_type} -> {instance.rtp[f'SofteningComovingType{softening_type}']}"
                )
            except KeyError:
                errors.append(
                    ValueError(
                        f"(SofteningComovingType{softening_type}): No softening length provided!"
                    )
                )

        # Assign the physical softening lengths. If not provided, set as the comoving value.
        for softening_type in range(instance.NSOFTTYPES):
            try:
                instance.rtp[f"SofteningMaxPhysType{softening_type}"] = (
                    instance.SOFTENING_PHYSICAL[softening_type]
                    .in_base(instance.unit_system)
                    .d
                )
            except (KeyError, TypeError):
                instance.rtp[f"SofteningMaxPhysType{softening_type}"] = instance.rtp[
                    f"SofteningComovingType{softening_type}"
                ]

            instance.logger.debug(
                f"\tSofteningMaxPhysType{softening_type} -> {instance.rtp[f'SofteningMaxPhysType{softening_type}']}"
            )

        # Return in the same fashion as the standard implementation.
        if len(errors):
            from cluster_generator.utilities.logging import ErrorGroup

            return False, ErrorGroup(
                f"Failed to set RTPs for {instance}!", error_list=errors
            )
        else:
            return True, None

    def write_rtp_template(
        self,
        instance: Instance,
        owner: Type[Instance],
        path: str | Path,
        overwrite: bool = False,
    ) -> Path:
        """Write the Arepo runtime parameters file.

        Parameters
        ----------
        instance: Arepo
            The Arepo instance to write the parameter file for.
        owner: Type[Arepo]
            The owner class.
        path: str
            The output location.
        overwrite: bool, optional
            Allow method to overwrite an existing file at this path. Default is ``False``.
        """
        instance.logger.info(f"Generating Arepo RTP template for {instance}.")

        # Managing overwrite issues and other potential IO/file system pitfalls.
        path = Path(path)

        if path.exists() and overwrite is False:
            # return an error, overwrite = False.
            raise ValueError(
                f"Arepo RTP template already exists at {path}. Overwrite=False."
            )
        elif path.exists() and overwrite is True:
            # Delete the path.
            path.unlink()
        else:
            # ensure parent directory exists.
            assert path.parents[
                0
            ].exists(), f"Parent directory of {path} doesn't exist."

        # Adding RTPs for the softening groups.
        # ------------------------------------
        # Dynamic softening length parameters need to be added to the owner rtps otherwise we cannot find them
        # when iterating through defaults!
        for particle_type_id in range(instance.NTYPES):
            owner.rtp[f"SofteningTypeOfPartType{particle_type_id}"] = dict(
                required=True, group="Softening", default_value=1
            )

        for softening_type in range(instance.NSOFTTYPES):
            owner.rtp[f"SofteningComovingType{softening_type}"] = dict(
                required=True, group="Softening", default_value=2.0
            )
            owner.rtp[f"SofteningMaxPhysType{softening_type}"] = dict(
                required=True, group="Softening", default_value=2.0
            )

        # Sorting RTPs
        # -------------
        # For Arepo, we want to have commented groups in the paramfile.txt. Here, we group our RTPs and get them
        # ready to write to disk.
        _rtp_write_dict = {}
        _available_groups = list(
            set([_rtp.get("group", "misc") for _, _rtp in owner.rtp.items()])
        )

        # Convert RTP types
        # -----------------
        # Any non-writable types need to be converted down.
        instance_rtps = self._convert_rtp_to_output_types(instance)

        for _group in _available_groups:
            # determine the right group keys
            _rtp_write_dict[_group] = {}
            _group_keys = [
                k for k, v in owner.rtp.items() if v.get("group", "misc") == _group
            ]

            # evaluate the need to write each key.
            # ------------------------------------
            # For Arepo, we check compile-time flags, then check for nulls, then proceed.
            for _gk in _group_keys:
                _instance_value = instance_rtps.get(
                    _gk, None
                )  # Get the value from the instance.
                _yflags, _nflags = owner.rtp[_gk].get("compile_flags", ([], []))
                _req = owner.rtp[_gk].get("required", False)

                if not all(
                    (getattr(owner, _yf) not in [False, None]) for _yf in _yflags
                ) or any(getattr(owner, _nf) for _nf in _nflags):
                    # Flag is missing
                    continue

                if not _req and _instance_value is None:
                    # We don't have to have it and it's null
                    continue

                if _instance_value is None:
                    raise ValueError(
                        f"RTP {_gk} is required but has null value in {instance}."
                    )

                _rtp_write_dict[_group][_gk] = _instance_value

        # Writing the template to disk
        with open(path, "w") as file:
            for k, v in _rtp_write_dict.items():
                file.write(f"%----- {k}\n")
                for _kg, _vg in v.items():
                    file.write("%(key)-40s%(value)s\n" % dict(key=_kg, value=_vg))
                file.write("\n")
            file.write("\n")

        return path


@dataclass
class Arepo(SimulationCode):
    """:py:class:`SimulationCode` class for interacting with the Arepo simulation
    software."""

    rtp: ClassVar[ArepoRuntimeParameters] = ArepoRuntimeParameters(
        os.path.join(
            Path(config_directory).parents[0],
            "codes",
            "arepo",
            "runtime_parameters.yaml",
        )
    )  # Path to the correct code bin in cluster_generator.

    logger: ClassVar[Logger] = LogDescriptor()
    """ Logger: The class logger for this code."""

    # USER PARAMETERS
    #
    # These are the parameters the user need to provide at the time of instantiation.
    IC_PATH: str = ufield()
    """str: The path to the initial conditions file."""
    SOFTENING_COMOVING: dict[int, unyt_quantity] = ufield()
    """
    dict: The comoving softening lengths.

    These should be specified in units of physical length. During runtime, it is interpreted as a comoving equivalent. If co-moving
    integration is not enabled, then these are simply physical lengths.
    """
    OUTPUT_STYLE: tuple[unyt_quantity, unyt_quantity] | unyt_array = ufield()
    """ tuple of unyt_quantity or unyt_array: The frequency of snapshot outputs.

    If a tuple is provided, the first value is the time of first output and the second is the time between remaining snapshots.
    If an array of times is provided, then they will be used for the output times.
    """
    TIME_MAX: unyt_quantity = ufield(flag="TimeMax")
    """ unyt_quantity: The maximum (simulation) time at which to end the simulation."""
    OUTPUT_DIR: str = ufield(default="./output", flag="OutputDir")
    """str: The directory in which to generate the simulation output.

    Default is ``./output``.
    """
    IC_FORMAT: Literal[1, 2, 3] = ufield(default=3, flag="ICFormat", av=[3])
    """int: The format of the initial conditions file.

    Format 1 corresponds to a Gadget-style IC, Format 2 is similar with 4 char block identifier, and Format 3 (default)
    corresponds to HDF5.
    """
    SNAP_FORMAT: Literal[1, 2, 3] = ufield(default=3, flag="SnapFormat")
    """int: The format of the snapshot outputs.

    Format conventions are the same as those in :py:attr:`Arepo.IC_FORMAT`. We suggest using 3 (default) to get HDF5 outputs.
    """
    VELOCITY_UNIT: unyt_quantity = ufield(
        default_factory=_const_factory(unyt_quantity(1.0, "km/s")),
        flag="UnitVelocity_in_cm_per_s",
        setter=lambda _inst, _, __: _inst.VELOCITY_UNIT.to_value("cm/s"),
    )
    """ unyt_quantity: The code-velocity unit.

    Default is 1 km / s
    """
    LENGTH_UNIT: unyt_quantity = ufield(
        default_factory=_const_factory(unyt_quantity(1.0, "kpc")),
        flag="UnitLength_in_cm",
        setter=lambda _inst, _, __: _inst.LENGTH_UNIT.to_value("cm"),
    )
    """ unyt_quantity: The code-length unit.

    Default is 1 kpc.
    """
    MASS_UNIT: unyt_quantity = ufield(
        default_factory=_const_factory(unyt_quantity(1.0, "Msun")),
        flag="UnitMass_in_g",
        setter=lambda _inst, _, __: _inst.MASS_UNIT.to_value("g"),
    )
    """ unyt_quantity: The code-mass unit

    Default is 1 Msun.
    """
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
    SOFTENING_TYPES: dict[int, unyt_quantity] = ufield(
        default_factory=_const_factory({0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}),
    )
    """ dict: The assigned softening types for each particle type.

    Should be a dictionary of keys from 0 to ``NTYPES-1`` with values corresponding to the preferred softening type. Generally,
    its advisable to have a type for gas particles and a type for non-gas particles. This is the default configuration.
    """
    SOFTENING_PHYSICAL: dict[int, unyt_quantity] = ufield(default=None)
    """
    dict: The maximum physical softening lengths.

    These are non-comoving softening lengths that come into play as soon as the comoving value begins to exceed this one.
    If left unspecified, they are set to the same value as the comoving so that they play no role.
    """

    # COMPILE-TIME PARAMETERS
    #
    # All CTPs should be attributes of the class and fill-able during initialization. They should have defaults
    # provided.
    NTYPES: int = cfield(default=6, av=[6])
    """ int: The number of particle types to recognize in the simulation.

    By default, this is ``6`` (the minimum), but it can be increased to accommodate additional particle species. Generally,
    these are as follows: ``{0:'gas',1:'dm',2:None,3:'stars / wind',4:'tracer'}``.

    .. warning::

        Cluster generator only supports these 5 particle types; as such, additional particle types will need to be
        implemented independently.

    """
    TWODIMS: bool = cfield(default=False, av=[False])
    """ bool: If ``True``, then AREPO is configured for 2D simulation only.

    .. note::

        Cluster generator does not allow for 2D simulations. Setting this flag to ``True`` will lead to an error.

    """
    ONEDIMS: bool = cfield(default=False, av=[False])
    """ bool: If ``True``, then AREPO is configured for 1D simulation only.

    .. note::

        Cluster generator does not allow for 1D simulations (non-spherical). Setting this flag to ``True`` will lead to an error.

    """
    ONEDIMS_SPHERICAL: bool = cfield(default=False, av=[False])
    """ bool: If ``True``, then AREPO is configured for 1D spherical simulation only.

    .. warning::

        :py:class:`Arepo` will eventually support 1D spherical simulations of single models; however, this is not
        yet available.

    """
    ISOTHERM_EQS: bool = cfield(default=False, av=[False])
    """ bool: Gas is treated as isothermal.

    .. warning::

        Cluster generator does not support this flag.

    """
    MHD: bool = cfield(default=False, av=[False])
    """ bool: If ``True``, then MHD is enabled.

    This flag will determine whether or not :py:class:`ics.ClusterICs` have their magnetic fields written to the IC files.
    """
    MHD_POWELL: bool = cfield(default=False)
    """ bool: If ``True``, then the Powell divergence cleaning scheme is used."""
    MHD_POWELL_LIMIT_TIMESTEP: bool = cfield(default=False)
    """ bool: If ``True``, then :py:attr:`Arepo.MHD_POWELL` will cause an additional time constraint restriction."""
    MHD_SEEDFIELD: bool = cfield(default=False)
    """ bool: Not understood."""
    TILE_ICS: bool = cfield(default=False)
    """ bool: If ``True``, then the RTP ``TimeICsFactor`` determines the number of times ICs are tiled along each dimension."""
    ADDBACKGROUNDGRID: int = cfield(default=None)
    """ int or None: Re-grid hydro quantities on an AMR grid, converts SPH into moving mesh."""
    REFINEMENT_VOLUME_LIMIT: bool = cfield(default=False)
    """ bool: Limit the max difference in volume between neighboring cells."""
    NODEREFINE_BACKGROUND_GRID: bool = cfield(default=False)
    """ bool: Prevent the background grid from de-refining.

    In the paramfile, the RTP ``MeanVolume`` must be set, and cells with ``V > 0.1*MeanVolume`` will not be allowed to derefine.
    """
    REGULARIZE_MESH_FACE_ANGLE: bool = cfield(default=False)
    """ bool: Use maximum face angle as roundness criterion in mesh regularization.
    """
    VORONOI_STATIC_MESH: bool = cfield(default=False)
    """ bool: If ``True``, then the Voronoi mesh doesn't change with time."""
    REFINEMENT_SPLIT_CELLS: bool = cfield(default=False)
    """ bool: Allow refinement."""
    REFINEMENT_MERGE_CELLS: bool = cfield(default=False)
    """ bool: Allow de-refinement."""
    ADAPTIVE_HYDRO_SOFTENING: bool = cfield(default=False)
    """ bool: Allow the softening length for fluid cells to vary with cell size."""
    SUBFIND: bool = cfield(default=False)
    """ bool: Enable subfind."""
    COOLING: bool = cfield(default=False)
    """ bool: Use a simple primordial cooling routine."""
    USE_SFR: bool = cfield(default=False)
    """ bool: Allow for star formation."""
    TOLERATE_WRITE_ERROR: bool = cfield(default=False)
    PASSIVE_SCALARS: int = cfield(default=0, av=[0])
    """ int: The number of passive scalars being advected with the fluid."""
    NSOFTTYPES: int = cfield(default=2, av=[2])
    """ int: The number of softening types.

    There must be an equal number of ``SofteningComovingTypeX`` attributes in the paramfile.txt.
    """
    INPUT_IN_DOUBLEPRECISION: bool = cfield(default=False)
    """ bool: Are initial conditions in double or single precision."""
    READ_COORDINATES_IN_DOUBLE: bool = cfield(default=False)
    """ bool: Read coordinates in double precision from ICs."""
    SHIFT_BY_HALF_BOX: bool = cfield(default=False)
    """ bool: If ``True``, then the simulation box is shifted by half a box after reading in."""
    NTYPES_ICS: int = cfield(default=6, av=[6])
    """ int: The number of particle types in the IC files."""
    READ_MASS_AS_DENSITY_IN_INPUT: bool = cfield(default=False, av=[False])
    """ bool: If ``True``, the mass field is interpreted as cell density in ICs."""
    HAVE_HDF5: bool = cfield(default=False, av=[True])
    """ bool: If ``True``, then input and output can be in HDF5."""

    def __post_init__(self):
        super().__post_init__()

    def generate_ics(
        self, initial_conditions: ClusterICs, overwrite: bool = False, **kwargs
    ) -> Path:
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
            from cluster_generator.codes.arepo.io import write_particles_to_arepo_hdf5

            self.logger.info("Writing particles to AREPO HDF5...")
            write_particles_to_arepo_hdf5(
                self, combined_particles, path=self.IC_PATH, overwrite=overwrite
            )
            self.logger.info("Writing particles to AREPO HDF5. [DONE]")
        else:
            raise ValueError(f"IC_FORMAT={self.IC_FORMAT} is not currently supported.")

    @property
    def unit_system(self) -> unyt.UnitSystem:
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

    @classmethod
    def from_install_directory(
        cls, installation_directory: str | Path, **parameters
    ) -> Self:
        """Determine the relevant class parameters for AREPO from its installation
        directory.

        Parameters
        ----------
        installation_directory: str
            The directory in which AREPO is installed.
        parameters:
            Other parameters to pass to the initializer for :py:class:`Arepo`.

        Returns
        -------
        Arepo
            An instance of the AREPO code class.

        Notes
        -----

        This method seeks out the ``Config.sh`` file in the installation directory and reads it to determine the
        enabled and disabled flags.
        """
        from dataclasses import fields

        from cluster_generator.codes.arepo.io import read_ctps

        ctps = read_ctps(installation_directory)
        required_ctps = {
            k: v for k, v in ctps.items() if k in [f.name for f in fields(cls)]
        }

        return cls(**required_ctps, **parameters)


if __name__ == "__main__":
    q = Arepo(
        IC_PATH="/test.hdf5",
        SOFTENING_COMOVING={k: unyt.unyt_quantity(2.0, "kpc") for k in range(6)},
        OUTPUT_STYLE=(unyt.unyt_quantity(0.0, "Gyr"), unyt.unyt_quantity(0.01, "Gyr")),
        TIME_MAX=unyt_quantity(10, "Gyr"),
        BOXSIZE=unyt_quantity(14, "kpc"),
    )

    q.determine_runtime_params(None)
    q.generate_rtp_template("test.txt")
