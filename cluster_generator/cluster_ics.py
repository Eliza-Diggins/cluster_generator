from cluster_generator.utils import mylog
from cluster_generator.cluster_model import ClusterModel
from cluster_generator.virial import VirialEquilibrium
from cluster_generator.cluster_particles import \
    ClusterParticles, \
    combine_two_clusters, \
    combine_three_clusters, \
    resample_one_cluster, \
    resample_two_clusters, \
    resample_three_clusters
import os
import numpy as np


def set_param(value, param_name, param):
    if param is not None:
        mylog.warning(f"'{param_name}' has already been set with value {param}. "
                      f"Overwriting with {value}!")
    return value


def check_for_list(x):
    if isinstance(x, list):
        return {i: v for i, v in enumerate(x)}
    else:
        return x


def format_array(x):
    return f"{x[0]} {x[1]} {x[2]}"


class ClusterICs:
    def __init__(self, num_halos, hse_files, num_particles,
                 center, velocity, mag_file=None, resample_file=None):
        self.num_halos = num_halos
        self.hse_files = check_for_list(hse_files)
        self.num_particles = num_particles
        self.center = check_for_list(center)
        self.velocity = check_for_list(velocity)
        self.mag_file = mag_file
        self.resample_file = resample_file

    def _generate_particles(self, r_max=None):
        parts = []
        if self.num_halos == 1:
            r_max = None
        for i, hf in enumerate(self.hse_files):
            hse = ClusterModel.from_h5_file(hf)
            vird = VirialEquilibrium.from_hse_model(hse, ptype="dm")
            p = vird.generate_particles(
                self.num_particles[i+1]["dm"], r_max=r_max)
            if self.num_particles["star"][i+1] > 0:
                virs = VirialEquilibrium.from_hse_model(hse, ptype="star")
                sp = virs.generate_particles(
                    self.num_particles[i+1]["star"], r_max=r_max)
                p = p + sp
            if self.num_particles["gas"][i+1] > 0:
                gp = hse.generate_particles(
                    self.num_particles[i+1]["star"], r_max=r_max)
                p = p + gp
            parts.append(p)
        return parts

    def to_file(self, filename, overwrite=False):
        r"""
        Write the initial conditions information to a file.

        Parameters
        ----------
        filename : string
            The file to write the initial conditions information to.
        overwrite : boolean, optional
            If True, overwrite a file with the same name. Default: False
        """
        if os.path.exists(filename) and not overwrite:
            raise RuntimeError(f"{filename} exists and overwrite=False!")
        outlines = [
            f"num_halos = {self.num_halos} # number of halos\n",
            f"profile1 = {self.hse_files[1]} # profile table of cluster 1\n",
            f"center1 = {format_array(self.center[1])} # center of cluster 1 in kpc\n",
            f"velocity1 = {format_array(self.velocity[1])} # velocity of cluster 1 in km/s\n"
        ]
        if self.num_particles[1]['dm'] > 0:
            outlines.append(
                f"num_dm_particles1 = {self.num_particles[1]['dm']} # number of dm particles for cluster 1\n"
            )
        if self.num_particles[1]['gas'] > 0:
            outlines.append(
                f"num_gas_particles1 = {self.num_particles[1]['gas']} # number of gas particles for cluster 1\n"
                )
        if self.num_particles[1]['star'] > 0:
            outlines.append(
                f"num_star_particles1 = {self.num_particles[1]['star']} # number of star particles for cluster 1\n"
                )
        if self.num_halos > 1:
            outlines += [
                f"\nprofile2 = {self.hse_files[2]} # profile table of cluster 2\n",
                f"center2 = {format_array(self.center[2])} # center of cluster 2 in kpc\n",
                f"velocity2 = {format_array(self.velocity[2])} # velocity of cluster 2 in km/s\n",
            ]
            if self.num_particles[2]['dm'] > 0:
                outlines.append(
                    f"num_dm_particles2 = {self.num_particles[2]['dm']} # number of dm particles for cluster 2\n"
                )
            if self.num_particles[2]['gas'] > 0:
                outlines.append(
                    f"num_gas_particles2 = {self.num_particles[2]['gas']} # number of gas particles for cluster 2\n"
                )
            if self.num_particles[2]['star'] > 0:
                outlines.append(
                    f"num_star_particles2 = {self.num_particles[2]['star']} # number of star particles for cluster 2\n"
                )
        if self.num_halos == 3:
            outlines += [
                f"\nprofile3 = {self.hse_files[3]} # profile table of cluster 3\n",
                f"center3 = {format_array(self.center[3])} # center of cluster 3 in kpc\n",
                f"velocity3 = {format_array(self.velocity[3])} # velocity of cluster 3 in km/s\n"
            ]
            if self.num_particles[3]['dm'] > 0:
                outlines.append(
                    f"num_dm_particles3 = {self.num_particles[3]['dm']} # number of dm particles for cluster 3\n"
                )
            if self.num_particles[3]['gas'] > 0:
                outlines.append(
                    f"num_gas_particles3 = {self.num_particles[3]['gas']} # number of gas particles for cluster 3\n"
                )
            if self.num_particles[3]['star'] > 0:
                outlines.append(
                    f"num_star_particles3 = {self.num_particles[3]['star']} # number of star particles for cluster 3\n"
                )
        if self.mag_file is not None:
            outlines.append(f"\nmag_file = {self.mag_file} # 3D magnetic field file\n")
        if self.resample_file is not None:
            outlines.append(f"\nresample_file = {self.resample_file} # Gadget file for resampling\n")
        with open(filename, "w") as f:
            f.writelines(outlines)

    @classmethod
    def from_file(cls, filename):
        r"""
        Read the initial conditions information
        from an appropriately formatted *filename*.
        """
        f = open(filename, "r")
        lines = f.readlines()
        f.close()
        num_halos = None
        hse_files = {1: None, 2: None, 3: None}
        num_particles = {i: {ptype: None for ptype in ["dm", "star", "gas"]}
                         for i in range(1, 4)}
        center = {1: None, 2: None, 3: None}
        velocity = {1: None, 2: None, 3: None}
        mag_file = None
        resample_file = None
        for line in lines:
            words = line.strip().split()
            if words[0].startswith("#"):
                continue
            elif len(words) == 3:
                if words[0] == "num_halos":
                    num_halos = set_param("num_halos", int(words[2]), num_halos)
                elif words[0] == "hse_file1":
                    hse_files[1] = set_param("hse_file1", words[2], hse_files[1])
                elif words[0] == "hse_file2" and num_halos > 1:
                    hse_files[2] = set_param("hse_file2", words[2], hse_files[2])
                elif words[0] == "hse_file3" and num_halos > 2:
                    hse_files[3] = set_param("hse_file3", words[2], hse_files[3])
                elif words[0] == "num_dm_particles1":
                    num_particles[1]["dm"] = set_param("num_dm_particles1", int(words[2]), 
                                                       num_particles[1]["dm"])
                elif words[0] == "num_dm_particles2":
                    num_particles[2]["dm"] = set_param("num_dm_particles2", int(words[2]),
                                                       num_particles[2]["dm"])
                elif words[0] == "num_dm_particles3":
                    num_particles[3]["dm"] = set_param("num_dm_particles3", int(words[2]),
                                                       num_particles[3]["dm"])
                elif words[0] == "num_gas_particles1":
                    num_particles[1]["gas"] = set_param("num_gas_particles1", int(words[2]),
                                                        num_particles[1]["gas"])
                elif words[0] == "num_gas_particles2":
                    num_particles[2]["gas"] = set_param("num_gas_particles2", int(words[2]),
                                                        num_particles[2]["gas"])
                elif words[0] == "num_gas_particles3":
                    num_particles[3]["gas"] = set_param("num_gas_particles3", int(words[2]),
                                                        num_particles[3]["gas"])
                elif words[0] == "num_star_particles1":
                    num_particles[1]["star"] = set_param("num_star_particles1", int(words[2]),
                                                         num_particles[1]["star"])
                elif words[0] == "num_star_particles2":
                    num_particles[2]["star"] = set_param("num_star_particles2", int(words[2]),
                                                         num_particles[2]["star"])
                elif words[0] == "num_star_particles3":
                    num_particles[3]["star"] = set_param("num_star_particles3", int(words[2]),
                                                         num_particles[3]["star"])
                elif words[0] == "mag_file":
                    mag_file = set_param("mag_file", words[2], mag_file)
                elif words[0] == "resample_file":
                    resample_file = set_param("resample_file", words[2], resample_file)
            elif len(words) == 5:
                if words[0] == "center1":
                    center[1] = set_param(
                        "center1", np.array(words[2:]).astype("float64"),
                        center[1])
                if words[0] == "center2":
                    center[2] = set_param(
                        "center2", np.array(words[2:]).astype("float64"),
                        center[2])
                if words[0] == "center3":
                    center[3] = set_param(
                        "center3", np.array(words[2:]).astype("float64"),
                        center[3])
                if words[0] == "velocity1":
                    velocity[1] = set_param(
                        "velocity1", np.array(words[2:]).astype("float64"),
                        velocity[1])
                if words[0] == "velocity2":
                    velocity[2] = set_param(
                        "velocity2", np.array(words[2:]).astype("float64"),
                        velocity[2])
                if words[0] == "velocity3":
                    velocity[3] = set_param(
                        "velocity3", np.array(words[2:]).astype("float64"),
                        velocity[3])

        return cls(num_halos, hse_files, num_particles,
                   center, velocity, mag_file=mag_file)

    def setup_gamer_ics(self, input_testprob, particle_file_prefix,
                        overwrite=False):
        r"""

        Write the "Input_TestProb" file for use with the 
        ClusterMerger setup in GAMER.

        Parameters
        ----------
        input_testprob : string 
            The path to the Input__TestProb file which will
            include the parameters. 
        overwrite : boolean, optional
            If True, a file of the same name will be overwritten.
            Default: False
        """
        if os.path.exists(input_testprob) and not overwrite:
            raise RuntimeError(f"{input_testprob} exists and overwrite=False!")
        parts = self._generate_particles()
        outlines = [
            "# problem-specific runtime parameters\n",
            f"Merger_Coll_NumHalos    {self.num_halos} # number of halos\n"
        ]
        for i in range(1, self.num_halos+1):
            particle_file = f"{particle_file_prefix}_parts{i}.h5"
            parts[i-1].write_gamer_input(particle_file)
            outlines += [
                f"Merger_File_Prof{i}       {self.hse_files[i]} # profile table of cluster {i}\n",
                f"Merger_File_Par{i}        {particle_file} # particle file of cluster {i}\n",
                f"Merger_Coll_PosX{i}       {self.center[i][0]} # X-center of cluster {i} in kpc\n",
                f"Merger_Coll_PosY{i}       {self.center[i][1]} # Y-center of cluster {i} in kpc\n",
                f"Merger_Coll_VelX{i}       {self.velocity[i][0]} # X-velocity of cluster {i} in km/s\n",
                f"Merger_Coll_VelY{i}       {self.velocity[i][1]} # Y-velocity of cluster {i} in km/s\n",
            ]
        with open(input_testprob, "w") as f:
            f.writelines(outlines)
        num_particles = 0
        for i in range(1, self.num_halos+1):
            for key in ["dm", "star"]:
                if self.num_particles[i][key] is not None:
                    num_particles += self.num_particles[i][key]
        mylog.info(f"In the Input__Parameter file, set PAR__NPAR = {num_particles}.")
        if self.mag_file is not None:
            mylog.info(f"Rename the file '{self.mag_file}' to 'B_IC' "
                       f"and place it in the same directory as the "
                       f"Input__* files, and set OPT__INIT_BFIELD_BYFILE "
                       f"to 1 in Input__Parameter")

    def setup_gadget_ics(self, filename, box_size, r_max=5000.0,
                         dtype='float32', overwrite=False):
        r"""
        From a set of initial conditions, set up an initial conditions
        file for use with the Gadget code or one of its derivatives
        (GIZMO, Arepo, etc.). Specifically, the output of this routine
        is a HDF5 file in "snapshot" format suitable for use as an
        initial condition.

        This routine will either write a single cluster or will combine
        two or three clusters together. If more than one cluster is 
        written, the gas particles will have their densities set by 
        adding the densities from the overlap of the two particles 
        together, and will have their thermal energies and velocities 
        set by mass-weighting them from the two profiles.

        Parameters
        ----------
        filename : string
            The file to be written to. 
        box_size : float
            The box size in which the initial conditions will be placed
            in units of kpc.
        r_max : float, optional
            The maximum radius in kpc for each cluster to which the 
            resampling from the profiles occurs. Ignored in the case
            of a single cluster. Default: 5000.0
        dtype : string, optional
            The datatype of the fields to be written. Default: float32
        overwrite : boolean, optional
            If True, a file of the same name will be overwritten. 
            Default: False
        """
        if os.path.exists(filename) and not overwrite:
            raise RuntimeError(f"{filename} exists and overwrite=False!")
        hses = [ClusterModel.from_h5_file(hf) for hf in self.hse_files]
        parts = self._generate_particles(r_max=r_max)
        if self.num_halos == 1:
            all_parts = parts[0]
        elif self.num_halos == 2:
            all_parts = combine_two_clusters(parts[0], parts[1], hses[0],
                                             hses[1], self.center[1], 
                                             self.center[2], self.velocity[1],
                                             self.velocity[2])
        else:
            all_parts = combine_three_clusters(parts[0], parts[1], parts[2],
                                               hses[0], hses[1], hses[2], 
                                               self.center[1], self.center[2], 
                                               self.center[3], self.velocity[1],
                                               self.velocity[2], self.velocity[3])
        all_parts.write_to_gadget_file(filename, box_size, dtype=dtype,
                                       overwrite=overwrite)

    def resample_gadget_ics(self, filename, dtype='float32', r_max=5000.0,
                            overwrite=False):
        r"""
        Given a Gadget-HDF5-like initial conditions file which has been
        output from some type of relaxation process (such as making a 
        glass or using MESHRELAX in the case of Arepo), resample the density,
        thermal energy, and velocity fields onto the gas particles/cells from
        the initial hydrostatic profiles.

        Parameters
        ----------
        filename : string
            The name of file to output the resampled ICs to.
        dtype : string, optional
            The datatype of the fields to be written. Default: float32
        r_max : float, optional
            The maximum radius in kpc for each cluster to which the 
            resampling from the profiles occurs. Ignored in the case
            of a single cluster. Default: 5000.0
        overwrite : boolean, optional
            If True, a file of the same name will be overwritten.
            Default: False
        """
        if os.path.exists(filename) and not overwrite:
            raise RuntimeError(f"{filename} exists and overwrite=False!")
        hses = [ClusterModel.from_h5_file(hf) for hf in self.hse_files]
        if self.resample_file is None or not os.path.exists(self.resample_file):
            raise IOError("A 'resample_file' must be specified in the "
                          "parameter file for this operation!")
        parts = ClusterParticles.from_gadget_file(self.resample_file)
        if self.num_halos == 1:
            new_parts = resample_one_cluster(parts, hses[0], self.center[1],
                                             self.velocity[1])
        elif self.num_halos == 2:
            new_parts = resample_two_clusters(parts, hses[0], hses[1],
                                              self.center[1], self.center[2],
                                              self.velocity[1], self.velocity[2],
                                              [r_max]*2)
        else:
            new_parts = resample_three_clusters(parts, hses[0], hses[1], hses[2],
                                                self.center[1], self.center[2],
                                                self.center[3], self.velocity[1], 
                                                self.velocity[2], self.velocity[3], 
                                                [r_max]*3)
        new_parts.write_to_gadget_file(filename, parts.box_size, dtype=dtype,
                                       overwrite=overwrite)
