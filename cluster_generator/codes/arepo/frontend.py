import os
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Type, TypeVar

import unyt
from pydantic import BaseModel, Field
from unyt import unyt_quantity

from cluster_generator import ClusterICs
from cluster_generator.codes._abc import (
    CodeFrontend,
    RuntimeParameters,
    runtime_parameters_directory,
)
from cluster_generator.codes._types import CField, UField, UnytQuantity
from cluster_generator.utils import LogDescriptor

if TYPE_CHECKING:
    from cluster_generator.codes.arepo.io import Arepo_HDF5

Instance = TypeVar("Instance", bound="Arepo")
Value = TypeVar("Value")
Attribute = TypeVar("Attribute")


class ArepoRuntimeParameters(RuntimeParameters):
    """Manages runtime parameters for Arepo simulations.

    Provides methods to deduce instance values for various parameters and write
    templates for runtime parameter files.

    Examples
    --------

    .. code-block:: python

        instance = Arepo()
        runtime_params = ArepoRuntimeParameters()
        runtime_params.write_rtp_template(instance, Arepo, "/path/to/rtp_file", overwrite=True)
    """

    def initialize_instance_model(self, instance: Instance) -> Type[BaseModel]:
        # We need to include all of the relevant softening types.
        from cluster_generator.codes._models import create_model

        rtp_schema = self._read_yaml_schema()
        base_fields = self._create_schema_fields(rtp_schema)

        # We can now proceed by adding the correct number of softening length parameters
        # to the fields and the generating the relevant model.
        softening_metadata = dict(
            default_units="kpc",
            default_value=unyt_quantity(1, "kpc"),
            group="Softening",
            compile_flags=[[], []],
            dtype="float",
            required=True,
        )

        for softening_type in range(instance.NSOFTTYPES):
            _comoving_field = (
                f"SofteningComovingType{softening_type}",
                UnytQuantity[float, unyt.Unit("kpc")],
                Field(
                    default=unyt_quantity(1, "kpc"),
                    json_schema_extra=softening_metadata,
                ),
            )
            _physical_field = (
                f"SofteningMaxPhysType{softening_type}",
                UnytQuantity[float, unyt.Unit("kpc")],
                Field(
                    default=unyt_quantity(1, "kpc"),
                    json_schema_extra=softening_metadata,
                ),
            )
            base_fields[_comoving_field[0]] = (_comoving_field[1], _comoving_field[2])
            base_fields[_physical_field[0]] = (_physical_field[1], _physical_field[2])

        return create_model("Dynamical Model", **base_fields)

    @staticmethod
    def _write_validator_required(instance, field, field_info, value):
        _errors = []
        _skip = False
        try:
            _default_value = field_info.json_schema_extra.get("default_value", None)
            _required = field_info.json_schema_extra.get("required", False)
            _compile_flags = field_info.json_schema_extra.get("compile_flags", [[], []])

            _yes_flag_status = all(getattr(instance, k) for k in _compile_flags[0])
            _no_flags_status = any(~getattr(instance, k) for k in _compile_flags[1])

            _status_flag = _no_flags_status | _yes_flag_status

            if not _status_flag:
                _skip = True
            else:
                # validate.
                if _required and (value is None):
                    _errors.append(
                        ValueError(f"Field {field} is required but value is None.")
                    )
                    _skip = True
                elif (not _required) and (value is None):
                    _skip = True
                elif (not _required) and (value is not None):
                    if value == _default_value:
                        _skip = True
                else:
                    pass

        except Exception as e:
            _errors.append(
                ValueError(f"Failed to validate required status for {field}: {e}")
            )
            _skip = True

        return _errors, _skip, value

    _WRITE_VALIDATORS = [
        _write_validator_required
    ] + RuntimeParameters._WRITE_VALIDATORS[1:]

    @staticmethod
    def _setter_output_parameters(
        instance: Instance, _: Type[Instance], __: ClusterICs
    ):
        # Set the InitCodeFile
        params = {
            "InitCondFile": str(Path(instance.IC_PATH).absolute().with_suffix("")),
            "OutputListOn": int(instance.OUTPUTS.OUTPUT_TIMES is not None),
        }

        if params["OutputListOn"] == 1:
            params["OutputListFilename"] = os.path.join(
                instance.OUTPUT_DIR, "tout_list.txt"
            )
        else:
            params["OutputListFilename"] = "0"

        if params["OutputListOn"] == 0:
            params["TimeOfFirstSnapshot"] = instance.OUTPUTS.START
            params["TimeBetSnapshot"] = instance.OUTPUTS.INTERVAL
        else:
            params["TimeOfFirstSnapshot"], params["TimeBetSnapshot"] = 0, 0

        return params

    @staticmethod
    def _setter_box_size(instance: Instance, _: Type[Instance], ic: ClusterICs):
        if instance.BOXSIZE is not None:
            return {"BoxSize": instance.BOXSIZE}
        else:
            # Deduce correct boxsize from the provided ICs.
            return {"BoxSize": unyt_quantity(2 * ic.r_max, "kpc")}

    @staticmethod
    def _setter_time_bet_stats(instance: Instance, _: Type[Instance], __: ClusterICs):
        if instance.OUTPUTS.INTERVAL is None:
            interval = {
                "TimeBetStatistics": (instance.OUTPUTS.END - instance.OUTPUTS.START)
                / 100
            }
        else:
            interval = {"TimeBetStatistics": instance.OUTPUTS.INTERVAL}

        return interval

    @staticmethod
    def _setter_softening(instance: Instance, _: Type[Instance], __: ClusterICs):
        params = {}
        for k, v in instance.SOFTENING_TYPES.items():
            params[f"SofteningTypeOfPartType{k}"] = v

        for k, v in instance.SOFTENING_COMOVING.items():
            params[f"SofteningComovingType{k}"] = v

        assert set(instance.SOFTENING_TYPES.values()) == set(
            instance.SOFTENING_COMOVING.keys()
        ), (
            f"Expected softenings for {set(instance.SOFTENING_TYPES.values())}, "
            f"got them for {set(instance.SOFTENING_COMOVING.keys())}."
        )

        if instance.SOFTENING_PHYSICAL is not None:
            for k, _ in instance.SOFTENING_COMOVING.items():
                params[f"SofteningMaxPhysType{k}"] = instance.SOFTENING_PHYSICAL.get(
                    k, instance.SOFTENING_COMOVING[k]
                )
        else:
            for k, _ in instance.SOFTENING_COMOVING.items():
                params[f"SofteningMaxPhysType{k}"] = instance.SOFTENING_COMOVING[k]

        return params

    def write_rtp_template(
        self,
        instance: Instance,
        path: str | Path,
        overwrite: bool = False,
    ) -> Path:
        instance.logger.info("Generating Arepo RTP template for %s", instance)

        # setup the path to ensure that we are not inadvertantly
        # overwriting important data.
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

        # Configure the RTP groups that AREPO expects.
        field_groups = {
            field_name: field.json_schema_extra.get("group", "misc")
            if field.json_schema_extra is not None
            else "misc"
            for field_name, field in instance.rtp.__fields__.items()
        }
        _available_groups = list(set(field_groups.values()))
        _raw_rtp_values = self._setup_write(instance)
        _write_dict = {}

        for group in _available_groups:
            _group_fields = [
                field_name
                for field_name, field in _raw_rtp_values.items()
                if field_groups[field_name] == group
            ]

            if not len(_group_fields):
                pass
            else:
                _write_dict[group] = {
                    k: v for k, v in _raw_rtp_values.items() if k in _group_fields
                }

        # Writing the template to disk
        with open(path, "w") as file:
            for group, group_fields in _write_dict.items():
                file.write(f"%----- {group}\n")
                for flag, value in group_fields.items():
                    file.write("%(key)-40s%(value)s\n" % dict(key=flag, value=value))
                file.write("\n")
            file.write("\n")

        return path


class Arepo(CodeFrontend):
    """SimulationCode class for interacting with the Arepo simulation software.

    Attributes
    ----------
    rtp : ClassVar[ArepoRuntimeParameters]
        The runtime parameters specific to Arepo, loaded from a YAML file.
    logger : ClassVar[Logger]
        The class logger for this code.

    User Parameters
    ---------------
    IC_PATH : str
        The path to the initial conditions file.
    SOFTENING_COMOVING : dict[int, unyt_quantity]
        The comoving softening lengths specified in units of physical length.
        Interpreted as comoving equivalents during runtime.
    OUTPUT_STYLE : tuple[unyt_quantity, unyt_quantity] or unyt_array
        The frequency of snapshot outputs. Can be a tuple (start time, interval) or an array of times.
    TIME_MAX : unyt_quantity
        The maximum simulation time at which to end the simulation.
    OUTPUT_DIR : str
        The directory in which to generate the simulation output. Default is `./output`.
    IC_FORMAT : Literal[1, 2, 3]
        The format of the initial conditions file. Default is 3 (HDF5).

        .. warning::

            We currently only support ``3``.

    SNAP_FORMAT : Literal[1, 2, 3]
        The format of the snapshot outputs. Default is 3 (HDF5).
    VELOCITY_UNIT : unyt_quantity
        The code-velocity unit. Default is 1 km/s.
    LENGTH_UNIT : unyt_quantity
        The code-length unit. Default is 1 kpc.
    MASS_UNIT : unyt_quantity
        The code-mass unit. Default is 1 Msun.
    BOXSIZE : unyt_quantity
        The size of the simulation's bounding box.
    START_TIME : unyt_quantity
        The start time of the simulation. Default is 0.0 Gyr.
    END_TIME : unyt_quantity
        The end time of the simulation. Default is 10 Gyr.
    SOFTENING_TYPES : dict[int, unyt_quantity]
        The assigned softening types for each particle type.
    SOFTENING_PHYSICAL : dict[int, unyt_quantity]
        The maximum physical softening lengths.

    Compile-Time Parameters
    -----------------------
    NTYPES : int
        The number of particle types to recognize in the simulation. Default is 6.
    TWODIMS : bool
        If `True`, AREPO is configured for 2D simulation only. Not supported by Cluster generator.
    ONEDIMS : bool
        If `True`, AREPO is configured for 1D simulation only. Not supported by Cluster generator.
    ONEDIMS_SPHERICAL : bool
        If `True`, AREPO is configured for 1D spherical simulation only. Not fully supported yet.
    ISOTHERM_EQS : bool
        If `True`, gas is treated as isothermal. Not supported by Cluster generator.
    MHD : bool
        If `True`, MHD is enabled. Not yet supported.
    MHD_POWELL : bool
        If `True`, the Powell divergence cleaning scheme is used. Not yet supported.
    MHD_POWELL_LIMIT_TIMESTEP : bool
        If `True`, MHD_POWELL causes an additional time constraint restriction. Not yet supported.
    MHD_SEEDFIELD : bool
        Not well understood. Likely not used.
    TILE_ICS : bool
        If `True`, the RTP `TimeICsFactor` determines the number of times ICs are tiled along each dimension.
    ADDBACKGROUNDGRID : int or None
        If set, re-grids hydro quantities on an AMR grid and converts SPH into moving mesh.
    REFINEMENT_VOLUME_LIMIT : bool
        Limits the max difference in volume between neighboring cells.
    NODEREFINE_BACKGROUND_GRID : bool
        Prevents the background grid from de-refining.
    REGULARIZE_MESH_FACE_ANGLE : bool
        Uses maximum face angle as roundness criterion in mesh regularization.
    VORONOI_STATIC_MESH : bool
        If `True`, the Voronoi mesh does not change with time.
    REFINEMENT_SPLIT_CELLS : bool
        Allows refinement.
    REFINEMENT_MERGE_CELLS : bool
        Allows de-refinement.
    ADAPTIVE_HYDRO_SOFTENING : bool
        Allows the softening length for fluid cells to vary with cell size.
    SUBFIND : bool
        Enables subfind.
    COOLING : bool
        Uses a simple primordial cooling routine.
    USE_SFR : bool
        Allows for star formation.
    TOLERATE_WRITE_ERROR : bool
        If `True`, tolerates write errors.
    PASSIVE_SCALARS : int
        The number of passive scalars being advected with the fluid. Default is 0.

        .. warning::

            Not currently supported.

    NSOFTTYPES : int
        The number of softening types. Default is 2.
    INPUT_IN_DOUBLEPRECISION : bool
        Determines if initial conditions are in double or single precision.
    READ_COORDINATES_IN_DOUBLE : bool
        Reads coordinates in double precision from ICs.
    SHIFT_BY_HALF_BOX : bool
        If `True`, the simulation box is shifted by half a box after reading in.
    NTYPES_ICS : int
        The number of particle types in the IC files. Default is 6.
    READ_MASS_AS_DENSITY_IN_INPUT : bool
        If `True`, the mass field is interpreted as cell density in ICs. Not supported by cluster generator.
    HAVE_HDF5 : bool
        If `True`, input and output can be in HDF5. Required by cluster generator.

    Examples
    --------
    .. code-block:: python

        arepo = Arepo(IC_PATH="path/to/ic", OUTPUT_DIR="path/to/output")
        arepo.generate_ics(initial_conditions=my_ics, overwrite=True)

    See Also
    --------
    ArepoRuntimeParameters
        For setting runtime parameters specific to Arepo.
    """

    rtp: ClassVar[ArepoRuntimeParameters] = ArepoRuntimeParameters(
        os.path.join(
            Path(runtime_parameters_directory),
            "arepo_rtp.yaml",
        )
    )

    logger: ClassVar[Logger] = LogDescriptor()
    """ Logger: The class logger for this code."""
    NTYPES: ClassVar[int] = 6

    IC_PATH: str = UField()
    SOFTENING_COMOVING: dict[int, UnytQuantity[float, unyt.Unit("kpc")]] = UField()
    OUTPUT_DIR: str = UField(default="./output", flag="OutputDir")
    IC_FORMAT: Literal[3] = UField(default=3, flag="ICFormat")
    SNAP_FORMAT: Literal[1, 2, 3] = UField(default=3, flag="SnapFormat")
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
    BOXSIZE: UnytQuantity[float, unyt.Unit("kpc")] = UField(default=None)
    SOFTENING_TYPES: dict[int, int] = UField(
        default_factory=lambda: {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
    )
    SOFTENING_PHYSICAL: dict[int, UnytQuantity[float, unyt.Unit("kpc")]] = UField(
        default=None
    )
    TWODIMS: Literal[False] = CField(default=False)
    ONEDIMS: Literal[False] = CField(default=False)
    ONEDIMS_SPHERICAL: Literal[False] = CField(default=False)
    ISOTHERM_EQS: Literal[False] = CField(default=False)
    MHD: Literal[False] = CField(default=False)
    MHD_POWELL: bool = CField(default=False)
    MHD_POWELL_LIMIT_TIMESTEP: bool = CField(default=False)
    MHD_SEEDFIELD: bool = CField(default=False)
    TILE_ICS: bool = CField(default=False)
    ADDBACKGROUNDGRID: int = CField(default=None)
    REFINEMENT_VOLUME_LIMIT: bool = CField(default=False)
    NODEREFINE_BACKGROUND_GRID: bool = CField(default=False)
    REGULARIZE_MESH_FACE_ANGLE: bool = CField(default=False)
    VORONOI_STATIC_MESH: bool = CField(default=False)
    REFINEMENT_SPLIT_CELLS: bool = CField(default=False)
    REFINEMENT_MERGE_CELLS: bool = CField(default=False)
    ADAPTIVE_HYDRO_SOFTENING: bool = CField(default=False)
    SUBFIND: bool = CField(default=False)
    COOLING: bool = CField(default=False)
    USE_SFR: bool = CField(default=False)
    TOLERATE_WRITE_ERROR: bool = CField(default=False)
    PASSIVE_SCALARS: Literal[0] = UField(default=0)
    NSOFTTYPES: int = UField(default=2, gt=0, lt=7)
    INPUT_IN_DOUBLEPRECISION: Literal[False] = CField(default=False)
    READ_COORDINATES_IN_DOUBLE: bool = CField(default=False)
    SHIFT_BY_HALF_BOX: bool = CField(default=False)
    NTYPES_ICS: Literal[6] = CField(default=6)
    READ_MASS_AS_DENSITY_IN_INPUT: Literal[False] = CField(default=False)
    HAVE_HDF5: Literal[True] = CField(default=True)

    def model_post_init(self, __context: dict):
        super(Arepo, self).model_post_init(__context)

    def _setup_hdf5_fields(self):
        from cluster_generator.codes.arepo.io import Arepo_FieldRegistry

        if self.INPUT_IN_DOUBLEPRECISION:
            _ftype = "f8"
        else:
            _ftype = "f4"

        fields = {
            "Coordinates": (
                "particle_position",
                "POS",
                self.unit_system["length"],
                _ftype,
                None,
                3,
            ),
            "Velocities": (
                "particle_velocity",
                "VEL",
                self.unit_system["length"] / self.unit_system["time"],
                _ftype,
                None,
                3,
            ),
            "ParticleIDs": ("particle_index", "ID", None, "i4", None, 1),
            "Masses": (
                "particle_mass",
                "MASS",
                self.unit_system["mass"],
                _ftype,
                None,
                1,
            ),
            "InternalEnergy": (
                "thermal_energy",
                "U",
                (self.unit_system["length"] / self.unit_system["time"]) ** 2,
                _ftype,
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

        return Arepo_FieldRegistry(fields)

    def _setup_hdf5_header(self):
        # The header now needs to be produced. We generate it from the particle dataset with all of the header
        # attributes as kwargs.
        _header_dict = {
            "Time": self.START_TIME.in_base(self.unit_system).d,
            "Flag_Sfr": self.rtp["StarformationOn"],
            "Flag_Cooling": self.rtp["CoolingOn"],
            "Flag_Feedback": self.rtp["StarformationOn"],
            "Flag_Metals": 0.0,
            "Flag_StellarAge": 0.0,
            "Flag_DoublePrecision": self.INPUT_IN_DOUBLEPRECISION,
        }

        # We don't allow cosmological evolution, so these parameters can simply be set to their defaults.
        # Setting header flags. These are all pulled from RTPs.

        return _header_dict

    def generate_ics(
        self, initial_conditions: ClusterICs, overwrite: bool = False, **kwargs
    ) -> "Arepo_HDF5":
        """Generates initial conditions for the simulation.

        Parameters
        ----------
        initial_conditions : ClusterICs
            The initial conditions to use for generating the simulation.
        overwrite : bool, optional
            If True, overwrite existing files. Default is False.
        **kwargs : dict
            Additional keyword arguments to pass during IC generation.

        Returns
        -------
        Path
            The path to the generated initial conditions file.

        Raises
        ------
        ValueError
            If an unsupported IC_FORMAT is specified.

        Examples
        --------
        .. code-block:: python

            arepo = Arepo(IC_PATH="path/to/ic", OUTPUT_DIR="path/to/output")
            my_ics = ClusterICs()
            arepo.generate_ics(initial_conditions=my_ics, overwrite=True)
        """
        from cluster_generator.codes.arepo.io import (
            Arepo_HDF5,
            Arepo_Header,
            Arepo_ParticleRegistry,
        )

        # Create a shared particle dataset from the initial conditions
        # we do this in ./arepo_tmp to contain any particle files generated.
        self.logger.info(f"Constructing AREPO ICs from {initial_conditions}.")

        initial_conditions.directory = "./arepo_tmp"
        self.logger.debug(f"\tIC dump directory: {Path(initial_conditions.directory)}.")

        self.logger.info("Combining constituent models and generating particles...")
        combined_particles = initial_conditions.setup_particle_ics(**kwargs)
        self.logger.info("Combining constituent models and generating particles [DONE]")

        # The particles need to have a particle_index generated.
        combined_particles.add_index()

        # Produce compliant HDF5 file
        if self.IC_FORMAT == 3:
            # Create the field registry. For initial conditions, we only need blocks up to the temperature.
            field_registry = self._setup_hdf5_fields()
            # remove fields we don't want to write.
            field_registry.remove_field("Density")
            field_registry.remove_field("Potential")

            # Now we deal with the particle registry. These should all be by convention, so we just generate
            # the default.
            particle_registry = Arepo_ParticleRegistry({})
            header = Arepo_Header.from_particles(combined_particles, particle_registry)

            self.logger.info("Writing particles to AREPO HDF5...")
            ICS = Arepo_HDF5.from_particles(
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
