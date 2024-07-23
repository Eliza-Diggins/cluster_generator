"""Frontend tools for interacting with the Arepo hydrodynamics code."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, Type

import unyt
from typing_extensions import Self
from unyt import unyt_array, unyt_quantity

from cluster_generator.codes.abc import RuntimeParameters, SimulationCode
from cluster_generator.codes.utils import _const_factory, cfield, ufield
from cluster_generator.ics import ClusterICs
from cluster_generator.utilities.config import config_directory
from cluster_generator.utilities.logging import mylog
from cluster_generator.utilities.types import Instance


class ArepoRuntimeParameters(RuntimeParameters):
    @staticmethod
    def set_InitCondFile(instance: Instance, _: Type[Instance], __: ClusterICs) -> str:
        """Sets the InitCondFile flag for Arepo's RTPs."""
        return str(instance.IC_PATH)

    @staticmethod
    def set_OutputListOn(instance: Instance, _: Type[Instance], __: ClusterICs) -> int:
        if isinstance(instance.OUTPUT_STYLE, unyt_array):
            return 1
        else:
            return 0

    @staticmethod
    def set_OutputListFilename(
        instance: Instance, _: Type[Instance], __: ClusterICs
    ) -> str | None:
        if isinstance(instance.OUTPUT_STYLE, unyt_array):
            return os.path.join(instance.OUTPUT_DIR, "tout_list.txt")
        else:
            return None

    @staticmethod
    def set_TimeBetSnapshot(
        instance: Instance, _: Type[Instance], __: ClusterICs
    ) -> float | None:
        if isinstance(instance.OUTPUT_STYLE, unyt_array):
            return None
        else:
            # we have a tuple, we want to use it.
            return instance.OUTPUT_STYLE[1].to_value(
                instance.LENGTH_UNIT / instance.VELOCITY_UNIT
            )

    @staticmethod
    def set_TimeOfFirstSnapshot(
        instance: Instance, _: Type[Instance], __: ClusterICs
    ) -> float | None:
        if isinstance(instance.OUTPUT_STYLE, unyt_array):
            return None
        else:
            # we have a tuple, we want to use it.
            return instance.OUTPUT_STYLE[0].to_value(
                instance.LENGTH_UNIT / instance.VELOCITY_UNIT
            )

    @staticmethod
    def set_BoxSize(instance: Instance, _: Type[Instance], ic: ClusterICs) -> float:
        if instance.BOXSIZE is not None:
            return instance.BOXSIZE.to_value(instance.LENGTH_UNIT)
        else:
            # Deduce correct boxsize from the provided ICs.
            return unyt_quantity(2 * ic.r_max, "kpc").to_value(instance.LENGTH_UNIT)

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
        mylog.info(f"Generating Arepo RTP template for {instance}.")

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

        # Sorting RTPs
        # -------------
        # For Arepo, we want to have commented groups in the paramfile.txt. Here, we group our RTPs and get them
        # ready to write to disk.
        _rtp_write_dict = {}
        _available_groups = list(
            set([_rtp.get("group", "misc") for _, _rtp in owner.rtp.items()])
        )

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
                _instance_value = instance.rtp[_gk]
                _yflags, _nflags = owner.rtp[_gk].get("compile_flags", ([], []))
                _req = owner.rtp[_gk].get("required", False)

                if not all(getattr(owner, _yf) for _yf in _yflags) or any(
                    getattr(owner, _nf) for _nf in _nflags
                ):
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

    # USER PARAMETERS
    #
    # These are the parameters the user need to provide at the time of instantiation.
    IC_PATH: str = ufield()
    """str: The path to the initial conditions file."""
    OUTPUT_STYLE: tuple[unyt_quantity, unyt_quantity] | unyt_array = ufield()
    """ tuple of unyt_quantity or unyt_array: The frequency of snapshot outputs.

    If a tuple is provided, the first value is the time of first output and the second is the time between remaining snapshots.
    If an array of times is provided, then they will be used for the output times.
    """
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
    NSOFTTYPES: int = cfield(default=4)
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
        initial_conditions.directory = "./arepo_tmp"
        combined_particles = initial_conditions.setup_particle_ics(**kwargs)

        # Produce compliant HDF5 file
        if self.IC_FORMAT == 3:
            from cluster_generator.codes.arepo.io import write_particles_to_arepo_hdf5

            write_particles_to_arepo_hdf5(
                self, combined_particles, path=self.IC_PATH, overwrite=overwrite
            )
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
        from cluster_generator.codes.arepo.io import read_ctps

        ctps = read_ctps(installation_directory)

        return cls(**ctps, **parameters)
