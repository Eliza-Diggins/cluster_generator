from cluster_generator.cluster_model import ClusterModel
from cluster_generator.utils import mylog


def setup_gamer_ics(ics, regenerate_particles=False, r_max=None):
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
    """
    hses = [ClusterModel.from_h5_file(hf) for hf in ics.hse_files]
    parts = ics._generate_particles(
        regenerate_particles=regenerate_particles)
    outlines = [
        f"Merger_Coll_NumHalos\t\t{ics.num_halos}\t# number of halos"
    ]
    for i in range(ics.num_halos):
        particle_file = f"{ics.basename}_gamerp_{i+1}.h5"
        parts[i].write_gamer_input(particle_file)
        if r_max is None:
            hse = hses[i]
        else:
            hse = hses[i].set_rmax(r_max)
        hse_file_gamer = ics.hse_files[i].replace(".h5", "_gamer.h5")
        hse.write_model_to_h5(hse_file_gamer, overwrite=True,
                              in_cgs=True, r_max=ics.r_max)
        vel = ics.velocity[i].to_value("km/s")
        outlines += [
            f"Merger_File_Prof{i+1}\t\t{hse_file_gamer}\t# profile table of cluster {i+1}",
            f"Merger_File_Par{i+1}\t\t{particle_file}\t# particle file of cluster {i+1}",
            f"Merger_Coll_PosX{i+1}\t\t{ics.center[i][0].v}\t# X-center of cluster {i+1} in kpc",
            f"Merger_Coll_PosY{i+1}\t\t{ics.center[i][1].v}\t# Y-center of cluster {i+1} in kpc",
            f"Merger_Coll_VelX{i+1}\t\t{vel[0]}\t# X-velocity of cluster {i+1} in km/s",
            f"Merger_Coll_VelY{i+1}\t\t{vel[1]}\t# Y-velocity of cluster {i+1} in km/s"
        ]
    mylog.info("Write the following lines to Input__TestProblem: ")
    for line in outlines:
        print(line)
    num_particles = sum([ics.tot_np[key] for key in ics.tot_np])
    mylog.info(f"In the Input__Parameter file, "
               f"set PAR__NPAR = {num_particles}.")
    if ics.mag_file is not None:
        mylog.info(f"Rename the file '{ics.mag_file}' to 'B_IC' "
                   f"and place it in the same directory as the "
                   f"Input__* files, and set OPT__INIT_BFIELD_BYFILE "
                   f"to 1 in Input__Parameter")


def setup_flash_ics(ics, use_particles=True, regenerate_particles=False):
    r"""

    Generate the "flash.par" lines needed for use
    with the GalaxyClusterMerger setup in FLASH. If the particles
    (dark matter and potentially star) have not been 
    created yet, they will be created at this step. 

    Parameters
    ----------
    ics : ClusterICs object
        The ClusterICs object to generate the GAMER ICs from.
    use_particles : boolean, optional
        If True, set up particle distributions. Default: True
    regenerate_particles : boolean, optional
        If particle files have already been created, particles
        are being used, and this flag is set to True, the particles 
        will be re-created. Default: False
    """
    if use_particles:
        ics._generate_particles(
            regenerate_particles=regenerate_particles)
    outlines = [
        f"testSingleCluster    {ics.num_halos} # number of halos"
    ]
    for i in range(ics.num_halos):
        vel = ics.velocity[i].to("km/s")
        outlines += [
            f"profile{i+1}\t=\t{ics.hse_files[i]}\t# profile table of cluster {i+1}",
            f"xInit{i+1}\t=\t{ics.center[i][0]}\t# X-center of cluster {i+1} in kpc",
            f"yInit{i+1}\t=\t{ics.center[i][1]}\t# Y-center of cluster {i+1} in kpc",
            f"vxInit{i+1}\t=\t{vel[0]}\t# X-velocity of cluster {i+1} in km/s",
            f"vyInit{i+1}\t=\t{vel[1]}\t# Y-velocity of cluster {i+1} in km/s",
        ]
        if use_particles:
            outlines.append(
                f"Merger_File_Par{i+1}\t=\t{ics.particle_files[i]}\t# particle file of cluster {i+1}",
            )
    mylog.info("Add the following lines to flash.par: ")
    for line in outlines:
        print(line)


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


def setup_ramses_ics(ics):
    r"""
    Parameters
    ----------
    ics : ClusterICs object
        The ClusterICs object to generate the Ramses ICs from.
    """
    pass


def make_gizmo_funcs(ics):
    r"""
    Parameters
    ----------
    ics : ClusterICs object
        The ClusterICs object to generate the GIZMO funcs from.
    """
    pass


