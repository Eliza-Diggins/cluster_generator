"""
Frontend tools for interacting with the Arepo hydrodynamics code.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, Type

from unyt import unyt_array, unyt_quantity

from cluster_generator.codes._abc import RuntimeParameters, SimulationCode
from cluster_generator.codes.utils import _const_factory, cfield, ufield
from cluster_generator.ics import ClusterICs
from cluster_generator.utils import Instance, config_directory, mylog


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
        """
        Write the Arepo runtime parameters file.

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
    """
    :py:class:`SimulationCode` class for interacting with the Arepo simulation software.
    """

    rtp: ClassVar[ArepoRuntimeParameters] = ArepoRuntimeParameters(
        os.path.join(
            Path(config_directory).parents[0],
            "codes",
            "arepo",
            "runtime_parameters.yaml",
        )
    )  # Path to the correct code bin in cluster_generator.

    # User Specified Fields
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
    IC_FORMAT: Literal[1, 2, 3] = ufield(default=3, flag="ICFormat")
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

    # Compile time flags
    NTYPES: int = cfield(default=6)
    TWODIMS: bool = cfield(default=False)
    ONEDIMS: bool = cfield(default=False)
    ONEDIMS_SPHERICAL: bool = cfield(default=False)
    ISOTHERM_EQS: bool = cfield(default=False)
    MHD: bool = cfield(default=False)
    MHD_POWELL: bool = cfield(default=False)
    MHD_POWELL_LIMIT_TIMESTEP: bool = cfield(default=False)
    MHD_SEEDFIELD: bool = cfield(default=False)
    TILE_ICS: bool = cfield(default=False)
    ADDBACKGROUNDGRID: bool = cfield(default=False)
    REFINEMENT_VOLUME_LIMIT: bool = cfield(default=False)
    NODEREFINE_BACKGROUND_GRID: bool = cfield(default=False)
    REGULARIZE_MESH_FACE_ANGLE: bool = cfield(default=False)
    VORONOI_STATIC_MESH: bool = cfield(default=False)
    REFINEMENT_SPLIT_CELLS: bool = cfield(default=False)
    REFINEMENT_MERGE_CELLS: bool = cfield(default=False)
    ADAPTIVE_HYDRO_SOFTENING: bool = cfield(default=False)
    SUBFIND: bool = cfield(default=False)
    COOLING: bool = cfield(default=False)
    USE_SFR: bool = cfield(default=False)
    TOLERATE_WRITE_ERROR: bool = cfield(default=False)

    def generate_ics(self, initial_conditions: ClusterICs) -> Path:
        pass


if __name__ == "__main__":
    q = Arepo(
        IC_PATH="test.h5",
        OUTPUT_STYLE=(unyt_quantity(0.0, "Gyr"), unyt_array(10.0, "Gyr")),
    )
    q.determine_runtime_params(None)
    print(q.get_rtp_class().write_rtp_template(q, Arepo, "test.txt", overwrite=True))
