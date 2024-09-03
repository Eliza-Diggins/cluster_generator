"""Frontend tools for interacting with the RAMSES hydrodynamics code."""
import os
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import ClassVar, Literal, Type

import numpy as np
import unyt
from numpy.typing import NDArray
from unyt import unyt_array, unyt_quantity

from cluster_generator import ClusterICs
from cluster_generator.codes._abc import RuntimeParameters, SimulationCode
from cluster_generator.codes.utils import _const_factory, cfield, ufield
from cluster_generator.utilities.config import config_directory
from cluster_generator.utilities.logging import LogDescriptor
from cluster_generator.utilities.types import Instance


class RamsesRuntimeParameters(RuntimeParameters):
    # == GRID RTP setters == #
    @staticmethod
    def set_levelmin(instance: Instance, _: Type[Instance], __: ClusterICs) -> int:
        # This sets the minimum level by taking LEVEL_BOUNDS[0].
        return instance.LEVEL_BOUNDS[0]

    @staticmethod
    def set_levelmax(instance: Instance, _: Type[Instance], __: ClusterICs) -> int:
        # This sets the maximum level by taking LEVEL_BOUNDS[1]
        return instance.LEVEL_BOUNDS[1]

    @staticmethod
    def set_nparttot(_: Instance, __: Type[Instance], ics: ClusterICs) -> int:
        # This sets the total number of particles to allocate space for.
        # we just take the sum of all of the non-gas particles in the IC.
        npart = 0

        for k, v in ics.num_particles.items():
            if k != "gas":
                npart += v

        return npart

    @staticmethod
    def set_ngridtot(instance: Instance, _: Type[Instance], __: ClusterICs) -> int:
        # The total number of grids to expect to allocate.
        # This is just going to be a guess based on FRAC_MAX_REFINED.
        ngrid_max_refined = 2 ** (
            3 * instance.LEVEL_BOUNDS[1]
        )  # The maximum possible number of grids.

        return int(np.ceil(instance.FRAC_MAX_REFINED * ngrid_max_refined))

    @staticmethod
    def set_nsubcycle(
        instance: Instance, _: Type[Instance], __: ClusterICs
    ) -> NDArray[int]:
        _nlevels = instance.LEVEL_BOUNDS[1] - instance.LEVEL_BOUNDS[0]
        # Set the sub-cycles
        if isinstance(instance.SUBCYCLE_TIMING, int):
            # The sub-cycles was a single int, we propogate it as an array
            nsubcycles = instance.SUBCYCLE_TIMING * np.ones(_nlevels)
        else:
            # The sub-cycles are specified explicitly
            nsubcycles = instance.SUBCYCLE_TIMING

        assert (
            len(nsubcycles) == _nlevels
        ), f"SUBCYCLE_TIMING has length {len(nsubcycles)}, expected {_nlevels}."

        return np.array(nsubcycles, dtype="uint32")

    # == OUTPUT RTP setters == #
    @staticmethod
    def set_noutput(instance: Instance, _: Type[Instance], __: ClusterICs) -> int:
        # Sets the number of outputs.
        outputs = instance.get_output_times()

        return len(outputs)

    @staticmethod
    def set_tout(
        instance: Instance, _: Type[Instance], __: ClusterICs
    ) -> NDArray[float]:
        # Set the output times.
        return instance.get_output_times().to_value(unyt.Unit(instance.TIME_UNIT))

    @staticmethod
    def set_units_density(
        instance: Instance, _: Type[Instance], __: ClusterICs
    ) -> float:
        # TIME and LENGTH are automatic, DENSITY needs to be converted to CGS and then written.
        _DENSITY_UNIT = (instance.TIME_UNIT**-2) / unyt.physical_constants.G

        return _DENSITY_UNIT.to_value("g/cm**3")

    def write_rtp_template(
        self,
        instance: Instance,
        owner: Type[Instance],
        path: str | Path,
        overwrite: bool = False,
    ) -> Path:
        """Write the RAMSES runtime parameters file.

        Parameters
        ----------
        instance: Ramses
            The Arepo instance to write the parameter file for.
        owner: Type[Ramses]
            The owner class.
        path: str
            The output location.
        overwrite: bool, optional
            Allow method to overwrite an existing file at this path. Default is ``False``.
        """
        from cluster_generator.io.nml import F90Namelist, F90NamelistGroup

        instance.logger.info(f"Generating RAMSES RTP template for {instance}.")

        # Managing overwrite issues and other potential IO/file system pitfalls.
        path = Path(path)

        if path.exists() and overwrite is False:
            # return an error, overwrite = False.
            raise ValueError(
                f"RAMSES RTP template already exists at {path}. Overwrite=False."
            )
        elif path.exists() and overwrite is True:
            # Delete the path.
            path.unlink()
        else:
            # ensure parent directory exists.
            assert path.parents[
                0
            ].exists(), f"Parent directory of {path} doesn't exist."

        # Convert RTP types
        # -----------------
        # Any non-writable types need to be converted down.
        instance_rtps = self._convert_rtp_to_output_types(instance)

        # Write RTPS
        # ---------------
        # RTPs for RAMSES are F90 namelist files. We evaluate the need to write each entry in a group,
        # then evaluate the need to write each group before writing.
        with F90Namelist(path) as namelist:
            _available_namelist_groups = list(
                set([_rtp.get("group", "misc") for _, _rtp in owner.rtp.items()])
            )

            for _group in _available_namelist_groups:
                # pull the group data, determine if it needs to actually be added.
                group = F90NamelistGroup({}, name=_group, owner=namelist)

                _group_keys = [
                    k for k, v in owner.rtp.items() if v.get("group", "misc") == _group
                ]
                for _gk in _group_keys:
                    _instance_value = instance_rtps.get(
                        _gk, None
                    )  # fetch from the user's RTPs.
                    _required = owner.rtp[_gk].get("required", False)

                    if not _required and _instance_value is None:
                        # We don't need this key, proceed as if nothing happens.
                        continue

                    if _instance_value is None:
                        raise ValueError(
                            f"RTP {_gk} is required but has null value in {instance}."
                        )

                    group[_gk] = _instance_value

                # Check if the group is empty. If so, we can just drop it.
                if not len(group):
                    namelist.drop_group(group)

        return path


@dataclass
class Ramses(SimulationCode):
    """:py:class:`SimulationCode` class for interacting with the RAMSES simulation
    software."""

    rtp: ClassVar[RamsesRuntimeParameters] = RamsesRuntimeParameters(
        os.path.join(
            Path(config_directory).parents[0],
            "codes",
            "ramses",
            "runtime_parameters.yaml",
        )
    )

    logger: ClassVar[Logger] = LogDescriptor()
    """ Logger: The class logger for this code."""

    # == USER PARAMETERS == #
    OUTPUT_STYLE: tuple[unyt_quantity, unyt_quantity] | unyt_array = ufield()
    """ tuple of unyt_quantity or unyt_array: The frequency of snapshot outputs.

    If a tuple is provided, the first value is the time of first output and the second is the time between remaining snapshots.
    If an array of times is provided, then they will be used for the output times.
    """

    TIME_MAX: unyt_quantity = ufield()
    """ unyt_quantity: The maximum time (in the simulation) to run.

    This should be the "end-point" of the simulation.
    """

    BOXSIZE: unyt_quantity = ufield(
        flag="boxlen",
        setter=lambda _inst, _, __: _inst.BOXSIZE.to_value(
            unyt.Unit(_inst.LENGTH_UNIT)
        ),
    )
    """ unyt_quantity: The size of the simulation domain.

    This should be sufficiently large to contain the entire simulation with some padding to ensure that
    boundary influences are minimized.
    """
    # Grid user parameters
    LEVEL_BOUNDS: tuple[int, int] = ufield(
        default=(7, 15),
    )
    """ tuple of int: The minimum and maximum AMR level for the simulation grid.

    Should be a length 2 tuple with the first element being the minimum grid refinement level and the
    second element being the maximum grid refinement level.
    This will force the base-grid to have a minimum of :math:`2^{3N}` grids.
    """
    SUBCYCLE_TIMING: NDArray[int] | int = ufield(default=2)

    # Unit System user parameters
    TIME_UNIT: unyt_quantity = ufield(
        default_factory=_const_factory(unyt_quantity(1.0, "Myr")),
        flag="units_time",
        setter=lambda _inst, _, __: _inst.TIME_UNIT.to_value("s"),
    )
    """ unyt_quantity: The code-time unit.

    Default is 1 Myr. Converted to CGS in the namelist file.

    .. hint::

        In RAMSES, units require :math:`G=1`. Thus, the ``units_density`` RTP is specified using the
        :py:attr:`LENGTH_UNIT` and :py:attr:`TIME_UNIT` attributes and this condition.

    """
    LENGTH_UNIT: unyt_quantity = ufield(
        default_factory=_const_factory(unyt_quantity(1.0, "kpc")),
        flag="units_length",
        setter=lambda _inst, _, __: _inst.LENGTH_UNIT.to_value("cm"),
    )
    """ unyt_quantity: The code-length unit.

    Default is 1 kpc. Converted to CGS in the namelist file.

    .. hint::

        In RAMSES, units require :math:`G=1`. Thus, the ``units_density`` RTP is specified using the
        :py:attr:`LENGTH_UNIT` and :py:attr:`TIME_UNIT` attributes and this condition.

    """

    # Computation User parameters
    FRAC_MAX_REFINED: float = ufield(default=0.0005)
    """ float: An estimate of the fraction of the domain that will reach maximal refinement.

    This has *no implications* for generating the ICs, but is used when creating the RTP template.
    Because RAMSES requires ``npartmax`` and ``ngridmax`` to be specified, this parameter is used
    to estimate the values of these RTPs.

    .. warning::

        The estimation of ``npartmax`` and ``ngridmax`` is somewhat naive. For grids, we have
        ``ngridtot = FRAC_MAX_REF*(2**(3*LEVELMAX))``. In each case, we assume equal loading on the
        CPUs.

        This is done only to provide the user with a reasonable estimate of the RTPs. Users should still
        evaluate the values in the context of their science goals and runtime environment.

    """

    # == COMPILE-TIME PARAMETERS == #
    NDIM: int = cfield(default=3, av=[3])
    """ int: The number of dimensions RAMSES was compiled to run on."""

    NPREC: int = cfield(default=8, av=[8])
    """ int: The number of byes in a float.
    The default is 8, corresponding to a standard double in C.
    """

    SOLVER: Literal["hydro", "mhd", "rhd"] = cfield(default="hydro", av=["hydro"])
    """ str: The solver RAMSES uses."""

    PATCH: str = cfield(default="")
    """ str: The patch (if any) being used in this RAMSES installation."""

    NENER: int = cfield(default=0, av=[0])
    """ int: Number of additional energies."""

    GRACKLE: int = cfield(default=0, av=[0])
    """ int: Use Grackle cooling? 1=Yes, 0=No."""

    NMETALS: int = cfield(default=0, av=[0])
    """ int: Number of metal species."""

    NPSCAL: int = cfield(default=0, av=[0])
    """ int: The number of passive scalar fields."""

    def get_grid_size(self, refinement_level: int) -> unyt_quantity:
        """Determine the size of a single grid element at a specific refinement level.

        Parameters
        ----------
        refinement_level: int
            The refinement level for which to determine the grid size.

        Returns
        -------
        unyt_quantity
            The resulting size of a single grid chunk.
        """
        return self.BOXSIZE / (2**refinement_level)

    def get_output_times(self) -> unyt_array:
        """Determine the array of times at which to place the outputs.

        Returns
        -------
        unyt_array
            The output times.
        """

        if isinstance(self.OUTPUT_STYLE, tuple):
            # We have (start, frequency) specified.
            _start, _frequency, _stop = (
                self.OUTPUT_STYLE[0].to_value("Myr"),
                self.OUTPUT_STYLE[1].to_value("Myr"),
                self.TIME_MAX.to_value("Myr"),
            )

            outputs = unyt_array(
                np.concatenate(
                    [np.arange(_start, _stop, _frequency), np.array([_stop])]
                ),
                "Myr",
            )

        else:
            # We have an array outright.
            outputs = self.OUTPUT_STYLE[
                self.OUTPUT_STYLE < self.TIME_MAX
            ]  # strict < prevents duplicate.
            outputs = np.concatenate(
                [outputs, unyt_array([self.TIME_MAX.value], self.TIME_MAX.units)]
            )

        return outputs

    def generate_ics(
        self, initial_conditions: ClusterICs, overwrite: bool = False, **kwargs
    ) -> Path:
        pass

    @property
    def unit_system(self) -> unyt.UnitSystem:
        if self._unit_system is None:
            # The units need to be constructed using G=1.
            _DENSITY_UNIT = (self.TIME_UNIT**-2) / unyt.physical_constants.G
            _MASS_UNIT = _DENSITY_UNIT * self.LENGTH_UNIT**3

            self._unit_system = unyt.UnitSystem(
                "RAMSES",
                unyt.Unit(self.LENGTH_UNIT),
                unyt.Unit(_MASS_UNIT),
                unyt.Unit(self.TIME_UNIT),
                "K",
                "rad",
            )

        return self._unit_system


if __name__ == "__main__":
    r = Ramses(
        OUTPUT_STYLE=unyt_array([0, 1, 1.1, 1.3, 4], "Gyr"),
        TIME_MAX=unyt_quantity(2, "Gyr"),
        BOXSIZE=unyt_quantity(10, "kpc"),
    )
    print(r.get_output_times())

    r.determine_runtime_params(None)
