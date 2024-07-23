"""IO tools for AREPO."""
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
from unyt import unyt_array, unyt_quantity

from cluster_generator.particles import ClusterParticles

if TYPE_CHECKING:
    from cluster_generator.codes.arepo.arepo import Arepo

from cluster_generator.utilities.logging import mylog

species_to_arepo_ptype = {
    "gas": 0,
    "dm": 1,
    "bulge": 2,
    "star": 3,
    "black_hole": 4,
    "tracer": 5,
}

cg_fields_to_arepo_fields = {
    "particle_mass": "Masses",
    "thermal_energy": "InternalEnergy",
    "particle_velocity": "Velocities",
    "particle_position": "Coordinates",
    "density": "Density",
}

arepo_ptype_to_species = {v: k for k, v in species_to_arepo_ptype.items()}
arepo_fields_to_cg_fields = {v: k for k, v in cg_fields_to_arepo_fields.items()}


def write_particles_to_arepo_hdf5(
    instance: "Arepo",
    particles: ClusterParticles,
    path: str | Path,
    overwrite: bool = False,
):
    """Write a particle dataset to an AREPO compliant HDF5 initial conditions file."""
    path = Path(path)

    if path.suffix != "hdf5":
        instance.logger.warning(
            f"Converting {path.suffix} to HDF5 so AREPO recognizes."
        )
        path = path.with_suffix(".hdf5")

    # Setup HDF5 path, create the host directory, etc.
    if not path.parents[0].exists():
        # we need to generate the directory as well.
        path.parents[0].mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        raise ValueError("{path} exists and overwrite = False.")
    elif path.exists() and overwrite:
        path.unlink()  # remove the file.
    else:
        pass

    # Generate the HDF5 stencil -> create the necessary groups and include metadata.
    particle_groups = [
        species_to_arepo_ptype[species] for species in particles.particle_types
    ]

    # Determine the particle types and generate the HDF5 structure for the stencil.
    with h5py.File(path, "w") as fio:
        # Create the header group
        fio.create_group("Header")

        num_particles = np.array(
            [
                particles.num_particles.get(arepo_ptype_to_species[i], 0)
                for i in range(6)
            ],
            dtype=np.intc,
        )
        fio["Header"].attrs["NumPart_ThisFile"] = num_particles
        # number of particles of each expected type, dtype = H5 NATIVE INT
        fio["Header"].attrs["NumPart_Total"] = np.array(
            [k % (2**32) for k in num_particles], dtype=np.uintc
        )
        # number of particles total -> same as per file because we use only 1 file. dtype = H5 NATIVE UINT
        # NOTE: we take % 2^32 because NumPart_Total_HighWord carries the integer multipliers. This allows for UINT instead of LONG
        # See https://www.tng-project.org/data/docs/specifications/ for an example of this in snapshot files.
        fio["Header"].attrs["NumPart_Total_HighWord"] = np.array(
            [k // (2**32) for k in num_particles], dtype=np.uintc
        )
        # Set the NumPart_Total_HighWord to N/2^32 rounded downward. DTYPE = H5 NATIVE UINT
        fio["Header"].attrs["NumFilesPerSnapshot"] = np.intc(1)
        # Set the NumFilesPerSnapshot header attribute to 1 -> indicate that this is a stand-alone IC. For sim, paramfile provides.

        # Writing the mass table to the header.
        # Because our particle class carries a native (potentially non-constant) mass field, we write a trivial
        # mass table and differ to mass fields in the ICs.
        fio["Header"].attrs["MassTable"] = np.zeros((6,), dtype=np.float64)
        # Set MassTable to zeros to indicate use of MASS field. DTYPE is H5 NATIVE DOUBLE
        fio["Header"].attrs["Time"] = np.float64(
            instance.START_TIME.in_base(instance.unit_system).d
        )
        # Set Time to start time in NATIVE DOUBLE; this is actually ignored by Arepo.
        fio["Header"].attrs["Redshift"] = np.float64(0.0)
        # NOTE: we don't allow comoving integration (doing so will break or start / end time and cause a load of problems).
        # as such, we don't let the user set these RTPs (as enforced by the RTP class), and we can hardcode these values.
        fio["Header"].attrs["Omega0"] = np.float64(0.0)
        fio["Header"].attrs["OmegaLambda"] = np.float64(0.0)
        fio["Header"].attrs["HubbleParam"] = np.float64(1.0)

        # Setting header flags. These are all pulled from RTPs.
        fio["Header"].attrs["Flag_Sfr"] = np.intc(instance.rtp["StarformationOn"])
        fio["Header"].attrs["Flag_Cooling"] = np.intc(instance.rtp["CoolingOn"])
        fio["Header"].attrs["Flag_Feedback"] = np.intc(instance.rtp["StarformationOn"])
        fio["Header"].attrs["Flag_Metals"] = np.intc(0.0)
        fio["Header"].attrs["Flag_StellarAge"] = np.intc(0.0)
        fio["Header"].attrs["Flag_DoublePrecision"] = np.intc(
            instance.INPUT_IN_DOUBLEPRECISION
        )

        # Setting the BoxSize attribute.
        if instance.BOXSIZE is None:
            mylog.warning(
                "[AREPO] BOXSIZE was not set by user. Computing a default from particle dataset."
            )

            min_max_diff = 0
            for species in particles.particle_types:
                coord_max = np.amax(
                    particles[species, "particle_position"].to_value("kpc"), axis=0
                )
                coord_min = np.amin(
                    particles[species, "particle_position"].to_value("kpc"), axis=0
                )

                spec_diff = np.ravel(coord_max - coord_min)

                min_max_diff = np.amax(np.append(spec_diff, min_max_diff))

            instance.BOXSIZE = unyt_quantity(2 * min_max_diff, "kpc")

            mylog.info(f"[AREPO] BOXSIZE is {instance.BOXSIZE}.")
        _bs = instance.BOXSIZE.in_base(
            instance.unit_system
        ).d  # BS in code-length units.
        fio["Header"].attrs["BoxSize"] = np.float64(_bs)

        """
        # Writing the group buffers.
        # For the non-gas particle types, we need only provide the following fields
        # Coordinates
        # ParticleIDs
        # Velocities
        # Masses
        # If we are working on gas particles, we also need to write the following fields
        # InternalEnergy
        """

        # CTP flag INPUT_IN_DOUBLEPRECISION
        _buffer_dtype = np.float64 if instance.INPUT_IN_DOUBLEPRECISION else np.float32
        # CTP flag SHIFT_BY_HALF_BOX will auto-center the ICs, otherwise, we need to do so manually.

        # TODO: In principle, INIT_GAS_TEMP might be specified, but why would anyone want that behavior in this context?
        if not instance.SHIFT_BY_HALF_BOX:
            # shift if we aren't doing it in AREPO.
            particles.add_offsets(
                unyt_array([instance.BOXSIZE.d / 2] * 3, instance.BOXSIZE.units),
                unyt_array([0, 0, 0], "km/s"),
            )
            # Cutoff the particles to ensure that our boxsize is a valid size
            particles.make_boxsize_cut(instance.BOXSIZE, centered=False)
        else:
            particles.make_boxsize_cut(instance.BOXSIZE, centered=False)

        # Writing groups.
        _total_particles_count = 0  # -> use this for ID offsets.
        for species, group_id in zip(particles.particle_types, particle_groups):
            # Iterate through the species / group numbers and write the correct fields.
            _np = particles.num_particles[species]

            # create the group for the particle type.
            _fields_to_write = [
                "particle_position",
                "particle_velocity",
                "particle_mass",
            ]
            if group_id == 0:  # -> gas, we need to also move the thermal energy.
                _fields_to_write += ["thermal_energy", "density"]

            # Create the group
            _group_name = f"PartType{group_id}"
            group = fio.create_group(_group_name)

            # ParticleIDs
            # -------------
            # This is not a simple set of arange calls. Each one needs to be unique across all particle types.
            # We just utilize an offset.
            group.create_dataset(
                "ParticleIDs",
                shape=(_np,),
                data=np.arange(_total_particles_count, _np + _total_particles_count),
                dtype=np.uintc,
            )
            _total_particles_count += _np
            # TODO: how to manage if > 2^32? Does anyone use > 2^32...

            # Write data fields
            for fld in _fields_to_write:
                group.create_dataset(
                    cg_fields_to_arepo_fields[fld],
                    dtype=_buffer_dtype,
                    data=np.array(
                        particles[species, fld].in_base(instance.unit_system).d,
                        dtype=_buffer_dtype,
                    ),
                )


def read_ctps(install_directory: str | Path):
    """Read the compile time flags from a given installation directory.

    Parameters
    ----------
    install_directory: str
        The path to the directory where AREPO is installed.

    Returns
    -------
    dict
        The compile-time flags from the installation.
    """
    import os

    path = Path(install_directory)
    assert path.exists(), f"The installation directory {path} doesn't exist!"

    config_path = Path(os.path.join(path, "Config.sh"))
    assert (
        config_path.exists()
    ), f"The compile-time flag file ({config_path}) doesn't exist."

    # Read the CTP file from disk
    with open(config_path, "r") as flag_io:
        _raw = flag_io.read()

    # Remove all of the commented lines and blank lines. This should remove comments and formatting
    _raw = [j for j in _raw.split("\n") if len(j) and j[0] != "#"]
    # Breakdown into just the provided values
    _raw = [j.split(" ")[0] for j in _raw]

    CTPs = {}

    for _ctp in _raw:
        if "=" not in _ctp:
            CTPs[_ctp] = True
        else:
            CTPs[_ctp.split("=")[0]] = int(_ctp.split("=")[1])

    return CTPs
