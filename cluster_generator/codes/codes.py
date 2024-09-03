"""
Code-specific utilities for the ``cluster_generator`` library.
"""
from pathlib import Path

import numpy as np
from unyt import uconcatenate

from cluster_generator.model import ClusterModel
from cluster_generator.utils import mylog


def write_amr_particles(
    particles,
    output_filename,
    ptypes,
    ptype_num,
    overwrite=True,
    in_cgs=False,
    format="hdf5",
):
    """
    Write the particles to an HDF5 file to be read in by the GAMER,
    FLASH, or RAMSES codes.

    Parameters
    ----------
    particles : :class:`cluster_generator.particles.ClusterParticles`
        The ClusterParticles instance which will be written.
    output_filename : string
        The file to write the particles to.
    overwrite : boolean, optional
        Overwrite an existing file with the same name. Default: False.
    """
    import h5py
    from scipy.io import FortranFile

    if Path(output_filename).exists() and not overwrite:
        raise IOError(
            f"Cannot create {output_filename}. It exists and overwrite=False."
        )
    nparts = [particles.num_particles[ptype] for ptype in ptypes]
    if format == "hdf5":
        write_class = h5py.File
    elif format == "fortran":
        write_class = FortranFile
    num_particles = 0
    with write_class(output_filename, "w") as f:
        pdata = []
        for field in ["particle_position", "particle_velocity", "particle_mass"]:
            fd = uconcatenate([particles[ptype, field] for ptype in ptypes], axis=0)
            if hasattr(fd, "units") and in_cgs:
                fd.convert_to_cgs()
            if format == "hdf5":
                f.create_dataset(field, data=np.asarray(fd))
            else:
                if field == "particle_mass":
                    num_particles = fd.size
                pdata.append(np.asarray(fd).astype("float64").T)
        if format == "hdf5":
            fd = np.concatenate(
                [
                    ptype_num[ptype] * np.ones(nparts[i])
                    for i, ptype in enumerate(ptypes)
                ]
            )
            f.create_dataset("particle_type", data=fd)
        else:
            f.write_record(num_particles)
            f.write_record(np.vstack(pdata).T)


def setup_gamer_ics(ics, regenerate_particles=False, use_tracers=False):
    r"""

    Generate the "Input_TestProb" lines needed for use
    with the ClusterMerger setup in GAMER. If the particles
    (dark matter and potentially star) have not been
    created yet, they will be created at this step. New profile
    files will also be created which have all fields in CGS units
    for reading into GAMER. If a magnetic field file is present
    in the ICs, a note will be given about how it should be named
    for GAMER to use it.

    Parameters
    ----------
    ics : ClusterICs object
        The ClusterICs object to generate the GAMER ICs from.
    regenerate_particles : boolean, optional
        If particle files have already been created and this
        flag is set to True, the particles will be
        re-created. Default: False
    use_tracers : boolean
        Set to True to add tracer particles. Default: False
    """
    gamer_ptypes = ["dm", "star"]
    if use_tracers:
        gamer_ptypes.insert(0, "tracer")
    gamer_ptype_num = {"tracer": 0, "dm": 2, "star": 3}
    hses = [ClusterModel.from_h5_file(hf) for hf in ics.profiles]
    parts = ics._generate_particles(regenerate_particles=regenerate_particles)
    outlines = [f"Merger_Coll_NumHalos\t\t{ics.num_halos}\t# number of halos"]
    for i in range(ics.num_halos):
        particle_file = f"{ics.basename}_gamerp_{i + 1}.h5"
        if ics.num_particles["star"][i] == 0:
            ptypes = gamer_ptypes[:-1]
        else:
            ptypes = gamer_ptypes
        write_amr_particles(
            parts[i], particle_file, ptypes, gamer_ptype_num, in_cgs=True, format="hdf5"
        )
        hse_file_gamer = ics.profiles[i].replace(".h5", "_gamer.h5")
        hses[i].write_model_to_h5(
            hse_file_gamer, overwrite=True, in_cgs=True, r_max=ics.r_max[i]
        )
        vel = ics.velocity[i].to_value("km/s")
        outlines += [
            f"Merger_File_Prof{i + 1}\t\t{hse_file_gamer}\t# profile table of cluster {i + 1}",
            f"Merger_File_Par{i + 1}\t\t{particle_file}\t# particle file of cluster {i + 1}",
            f"Merger_Coll_PosX{i + 1}\t\t{ics.center[i][0].v}\t# X-center of cluster {i + 1} in kpc",
            f"Merger_Coll_PosY{i + 1}\t\t{ics.center[i][1].v}\t# Y-center of cluster {i + 1} in kpc",
            f"Merger_Coll_VelX{i + 1}\t\t{vel[0]}\t# X-velocity of cluster {i + 1} in km/s",
            f"Merger_Coll_VelY{i + 1}\t\t{vel[1]}\t# Y-velocity of cluster {i + 1} in km/s",
        ]
    mylog.info("Write the following lines to Input__TestProblem: ")
    for line in outlines:
        print(line)
    if ics.mag_file is not None:
        mylog.info(
            f"Rename the file '{ics.mag_file}' to 'B_IC' "
            f"and place it in the same directory as the "
            f"Input__* files, and set OPT__INIT_BFIELD_BYFILE "
            f"to 1 in Input__Parameter"
        )


def setup_athena_ics(ics):
    r"""
    Parameters
    ----------
    ics : ClusterICs object
        The ClusterICs object to generate the Athena ICs from.
    """
    mylog.info("Add the following lines to athinput.cluster3d: ")


def setup_enzo_ics(ics):
    r"""
    Parameters
    ----------
    ics : ClusterICs object
        The ClusterICs object to generate the Enzo ICs from.
    """
    pass


def setup_ramses_ics(ics, regenerate_particles=False):
    r"""
    Parameters
    ----------
    ics : ClusterICs object
        The ClusterICs object to generate the Ramses ICs from.
    regenerate_particles : boolean, optional
        If particle files have already been created, particles
        are being used, and this flag is set to True, the particles
        will be re-created. Default: False
    """
    names = ["Main", "Sub", "Third"]
    config_lines = ["# Merger Dynamics Setting, do not change the general format"]
    hses = [ClusterModel.from_h5_file(hf) for hf in ics.profiles]
    parts = ics._generate_particles(regenerate_particles=regenerate_particles)
    fields_to_write = ["radius", "density", "pressure"]
    for i in range(ics.num_halos):
        if i > 0:
            config_lines.append("#")
        config_lines += [f"# {names[i]}", "#", "#", f"Halo {i + 1}"]
        hses[i].write_model_to_binary(
            f"halo{i + 1}_prof.dat",
            overwrite=True,
            in_cgs=True,
            r_max=ics.r_max,
            fields_to_write=fields_to_write,
        )
        vel = ics.velocity[i].to_value("km/s")
        pos = ics.center[i].to_value("kpc")
        config_lines += [
            f"x_cen[kpc]     ={pos[0]:16.6e}",
            f"y_cen[kpc]     ={pos[1]:16.6e}",
            f"z_cen[kpc]     ={pos[2]:16.6e}",
            f"vx_cen[kms]    ={vel[0]:16.6e}",
            f"vy_cen[kms]    ={vel[1]:16.6e}",
            f"vz_cen[kms]    ={vel[2]:16.6e}",
        ]
        write_amr_particles(
            parts[i],
            f"halo{i + 1}_part.dat",
            ["dm"],
            {"dm": 1},
            format="fortran",
            in_cgs=True,
        )
    mylog.info("Simulation setups saved to Merger_Config.txt.")
    np.savetxt("Merger_Config.txt", config_lines, fmt="%s")


def setup_gizmo_ics(ics):
    r"""
    Parameters
    ----------
    ics : ClusterICs object
        The ClusterICs object to generate the GIZMO funcs from.
    """
    pass


def setup_art_ics(ics):
    pass
