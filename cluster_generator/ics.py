from cluster_generator.utils import ensure_ytarray, ensure_list, \
    parse_prng
from cluster_generator.model import ClusterModel
from cluster_generator.particles import \
    ClusterParticles, \
    combine_two_clusters, \
    combine_three_clusters, \
    resample_one_cluster, \
    resample_two_clusters, \
    resample_three_clusters
import os
import numpy as np
from ruamel.yaml import YAML


def compute_centers_for_binary(center, d, b, a=0.0):
    """
    Given a common center and distance parameters, calculate the
    central positions of two clusters.

    First, the separation along the x-direction is determined
    by:

    sep_x = sqrt(d**2-b**2-a**2)

    where d is the distance between the two clusters, b is the
    impact parameter in the y-direction, and a is the impact 
    parameter in the z-direction. So the resulting centers are
    calculated as:

    center1 = [center-0.5*sep_x, center-0.5*b, center-0.5*a]
    center2 = [center+0.5*sep_x, center+0.5*b, center+0.5*a]

    Parameters
    ----------
    center : array-like
        The center from which the distance parameters for
        the two clusters will be calculated.
    d : float
        The distance between the two clusters.
    b : float
        The impact parameter in the y-direction, in kpc.
    a : float, optional
        The impact parameter in the z-direction, in kpc.
        Default: 0.0
    """
    d = np.sqrt(d*d-b*b-a*a)
    diff = np.array([d, b, a])
    center1 = center - 0.5*diff
    center2 = center + 0.5*diff
    return center1, center2


class ClusterICs:
    """
    The ``ClusterICs`` object is the user-side class containing the complete initial condition for a given simulation.

    Parameters
    ----------
    basename: str
        The base name of the initial condition. This can be more or less arbitrary.
    num_halos: int
        The number of halos included in the simulation.
    profiles: list of str
        List of path strings to the ``.h5`` files containing the cluster models for the given simulation.
    center: unyt_array
        The center of the various halos. This should be an ``m x 3`` array, where ``m`` is the number of halos.
    velocity: unyt_array
        The velocities of the various halos. This should be an ``m x 3`` array.
    num_particles: dict
        The number of each type of particle to include. Keys should be ``dm,star,gas``.
    mag_file: str
        The path to the magnetic field file.
    particle_files: list of str
        The paths to the additional particle files to include.
    r_max: float
        The maximal radius up to which data is permitted.

    """
    def __init__(self, basename, num_halos, profiles, center,
                 velocity, num_particles=None, mag_file=None, 
                 particle_files=None, r_max=20000.0):
        #  Managing parameters
        # ------------------------------------------------------------------------------------------------------------ #
        #: The name of the cluster initial conditions
        self.basename = basename
        #: the number of halos included.
        self.num_halos = num_halos
        #: the list of profile files.
        self.profiles = ensure_list(profiles)
        #: the centers of the clusters.
        self.center = ensure_ytarray(center, "kpc")
        #: The velocities of the clusters.
        self.velocity = ensure_ytarray(velocity, "kpc/Myr")

        if self.num_halos == 1: #--> make sure (3,) sized arrays become (1,3)
            self.center = self.center.reshape(1, 3)
            self.velocity = self.velocity.reshape(1, 3)

        #: The magnetic field file.
        self.mag_file = mag_file
        #: The maximal allowed radius.
        self.r_max = r_max

        if num_particles is None:
            #: dictionary containing the total numbers of all the particles.
            self.tot_np = {"dm": 0, "gas": 0, "star": 0}
        else:
            self.tot_np = num_particles

        #  T
        # ------------------------------------------------------------------------------------------------------------ #
        self._determine_num_particles()
        self.particle_files = [None]*3
        if particle_files is not None:
            self.particle_files[:num_halos] = particle_files[:]

    def _determine_num_particles(self):
        """
        Computes the number of particles and their allocations.
        """
        from collections import defaultdict
        # - Pre-allocation of arrays -------------------------------#
        dm_masses = []
        gas_masses = []
        star_masses = []

        # - Analyzing models ---------------------------------------#
        for pf in self.profiles:
            # Loading the cluster model object #
            p = ClusterModel.from_h5_file(pf)
            idxs = p["radius"] < self.r_max # These are the valid radii for our use.

            # Determining allowable number.
            for field_name,lst in zip(["dark_matter_mass","gas_mass","stellar_mass"],
                                      [dm_masses,gas_masses,star_masses]):
                if field_name in p:
                    # We found a matching profile
                    lst.append(p[field_name][idxs][-1].value)
            else:
                lst.append(0.0)

        # - Finding true total masses - #
        tot_dm_mass = np.sum(dm_masses)
        tot_gas_mass = np.sum(gas_masses)
        tot_star_mass = np.sum(star_masses)

        # Determining the necessary particles / halo distribution.
        # ----------------------------------------------------------------------------------------------------------------- #
        self.num_particles = defaultdict(list)
        for i in range(self.num_halos):
            for ptype,pmasses,ptmass in zip(["dm","gas","star"],
                                            [dm_masses,gas_masses,star_masses],
                                            [tot_dm_mass,tot_gas_mass,tot_star_mass]):
                if self.tot_np.get(ptype,0) > 0:
                    _n = np.rint(self.tot_np[ptype]*pmasses[i]/ptmass).astype("int")
                else:
                    _n = 0
                self.num_particles[ptype].append(_n)


    def _generate_particles(self, regenerate_particles=False, prng=None):
        """
        Generates the particles for the ``ClusterIC`` object.
        Parameters
        ----------
        regenerate_particles: bool
            Make ``True`` to regenerate particles instead of looking for them in the particle files.

        Returns
        -------

        """
        #  Setup
        # ------------------------------------------------------------------------------------------------------------ #
        prng = parse_prng(prng)
        parts = []

        for i, pf in enumerate(self.profiles):
            # -- Cycle through each halo and generate the corresponding particles -- #
            if regenerate_particles or self.particle_files[i] is None:
                m = ClusterModel.from_h5_file(pf)
                p = m.generate_dm_particles(
                    self.num_particles["dm"][i], r_max=self.r_max, prng=prng)
                if self.num_particles["star"][i] > 0:
                    sp = m.generate_star_particles(
                        self.num_particles["star"][i], r_max=self.r_max,
                        prng=prng)
                    p = p + sp
                if self.num_particles["gas"][i] > 0:
                    gp = m.generate_gas_particles(
                        self.num_particles["gas"][i], r_max=self.r_max,
                        prng=prng)
                    p = p + gp
                parts.append(p)
                outfile = f"{self.basename}_{i}_particles.h5"
                p.write_particles(outfile, overwrite=True)
                self.particle_files[i] = outfile
            else:
                # Regenerate particles is false and we found a file, we can make the particles from file.
                p = ClusterParticles.from_file(self.particle_files[i])
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
        from ruamel.yaml.comments import CommentedMap
        out = CommentedMap()
        out["basename"] = self.basename
        out.yaml_add_eol_comment("base name for ICs", key="basename")
        out["num_halos"] = self.num_halos
        out.yaml_add_eol_comment("number of halos", key='num_halos')
        out["profile1"] = self.profiles[0]
        out.yaml_add_eol_comment("profile for cluster 1", key='profile1')
        out["center1"] = self.center[0].tolist()
        out.yaml_add_eol_comment("center for cluster 1", key='center1')
        out["velocity1"] = self.velocity[0].tolist()
        out.yaml_add_eol_comment("velocity for cluster 1", key='velocity1')
        if self.particle_files[0] is not None:
            out["particle_file1"] = self.particle_files[0]
            out.yaml_add_eol_comment("particle file for cluster 1",
                                     key='particle_file1')
        if self.num_halos > 1:
            out["profile2"] = self.profiles[1]
            out.yaml_add_eol_comment("profile for cluster 2", key='profile2')
            out["center2"] = self.center[1].tolist()
            out.yaml_add_eol_comment("center for cluster 2", key='center2')
            out["velocity2"] = self.velocity[1].tolist()
            out.yaml_add_eol_comment("velocity for cluster 2", key='velocity2')
            if self.particle_files[1] is not None:
                out["particle_file2"] = self.particle_files[1]
                out.yaml_add_eol_comment("particle file for cluster 2", 
                                         key='particle_file2')
        if self.num_halos == 3:
            out["profile3"] = self.profiles[2]
            out.yaml_add_eol_comment("profile for cluster 3", key='profile3')
            out["center3"] = self.center[2].tolist()
            out.yaml_add_eol_comment("center for cluster 3", key='center3')
            out["velocity3"] = self.velocity[2].tolist()
            out.yaml_add_eol_comment("velocity for cluster 3", key='velocity3')
            if self.particle_files[2] is not None:
                out["particle_file3"] = self.particle_files[2]
                out.yaml_add_eol_comment("particle file for cluster 3",
                                         key='particle_file3')
        if self.tot_np.get("dm", 0) > 0:
            out["num_dm_particles"] = self.tot_np["dm"]
            out.yaml_add_eol_comment("number of DM particles", 
                                     key='num_dm_particles')
        if self.tot_np.get("gas", 0) > 0:
            out["num_gas_particles"] = self.tot_np["gas"]
            out.yaml_add_eol_comment("number of gas particles", 
                                     key='num_gas_particles')
        if self.tot_np.get("star", 0) > 0:
            out["num_star_particles"] = self.tot_np["star"]
            out.yaml_add_eol_comment("number of star particles", 
                                     key='num_star_particles')
        if self.mag_file is not None:
            out["mag_file"] = self.mag_file
            out.yaml_add_eol_comment("3D magnetic field file", key='mag_file')
        out["r_max"] = self.r_max
        out.yaml_add_eol_comment("Maximum radius of particles", key='r_max')
        yaml = YAML()
        with open(filename, "w") as f:
            yaml.dump(out, f)

    @classmethod
    def from_file(cls, filename):
        r"""
        Read the initial conditions information
        from a YAML-formatted `filename`.
        """
        from ruamel.yaml import YAML
        yaml = YAML()
        with open(filename, "r") as f:
            params = yaml.load(f)
        basename = params["basename"]
        num_halos = params["num_halos"]
        profiles = [params[f"profile{i}"] for i in range(1, num_halos+1)]
        center = [np.array(params[f"center{i}"]) for i in range(1, num_halos+1)]
        velocity = [np.array(params[f"velocity{i}"]) 
                    for i in range(1, num_halos+1)]
        num_particles = {k: params.get(f"num_{k}_particles", 0)
                         for k in ["gas", "dm", "star"]}
        mag_file = params.get("mag_file", None)
        particle_files = [params.get(f"particle_file{i}", None)
                          for i in range(1, num_halos+1)]
        r_max = params.get("r_max", 20000.0)
        return cls(basename, num_halos, profiles, center, velocity,
                   num_particles=num_particles, mag_file=mag_file,
                   particle_files=particle_files, r_max=r_max)

    def setup_particle_ics(self, regenerate_particles=False, prng=None):
        r"""
        From a set of cluster models and their relative positions and
        velocities, set up initial conditions for use with SPH codes.

        This routine will either generate a single cluster or will combine
        two or three clusters together. If more than one cluster is 
        generated, the gas particles will have their densities set by 
        adding the densities from the overlap of the two particles 
        together, and will have their thermal energies and velocities 
        set by mass-weighting them from the two profiles.

        Parameters
        ----------
        """
        profiles = [ClusterModel.from_h5_file(hf) for hf in self.profiles]
        parts = self._generate_particles(
            regenerate_particles=regenerate_particles, prng=prng)
        if self.num_halos == 1:
            all_parts = parts[0]
            all_parts.add_offsets(self.center[0], self.velocity[0])
        elif self.num_halos == 2:
            all_parts = combine_two_clusters(parts[0], parts[1], profiles[0],
                                             profiles[1], self.center[0],
                                             self.center[1], self.velocity[0],
                                             self.velocity[1])
        else:
            all_parts = combine_three_clusters(parts[0], parts[1], parts[2],
                                               profiles[0], profiles[1], 
                                               profiles[2], self.center[0], 
                                               self.center[1], self.center[2], 
                                               self.velocity[0], self.velocity[1],
                                               self.velocity[2])
        return all_parts

    def resample_particle_ics(self, parts, passive_scalars=None):
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
        """
        profiles = [ClusterModel.from_h5_file(hf) for hf in self.profiles]
        if self.num_halos == 1:
            new_parts = resample_one_cluster(parts, profiles[0], self.center[0],
                                             self.velocity[0])
        elif self.num_halos == 2:
            new_parts = resample_two_clusters(parts, profiles[0], profiles[1],
                                              self.center[0], self.center[1],
                                              self.velocity[0], self.velocity[1],
                                              [self.r_max]*2,
                                              passive_scalars=passive_scalars)
        else:
            new_parts = resample_three_clusters(parts, profiles[0], profiles[1], 
                                                profiles[2], self.center[0], 
                                                self.center[1], self.center[2], 
                                                self.velocity[0], self.velocity[1], 
                                                self.velocity[2], [self.r_max]*3, 
                                                passive_scalars=passive_scalars)
        return new_parts

    def create_dataset(self, domain_dimensions, box_size, left_edge=None,
                       **kwargs):
        """
        Create an in-memory, uniformly gridded dataset in 3D using yt by
        placing the clusters into a box. When adding multiple clusters,
        per-volume quantities from each cluster such as density and
        pressure are added, whereas per-mass quantites such as temperature
        and velocity are mass-weighted.

        Parameters
        ----------
        domain_dimensions : 3-tuple of ints
            The number of cells on a side for the domain.
        box_size : float
            The size of the box in kpc.
        left_edge : array_like, optional
            The minimum coordinate of the box in all three dimensions,
            in kpc. Default: None, which means the left edge will
            be [0, 0, 0].
        """
        from yt.loaders import load_uniform_grid
        from scipy.interpolate import InterpolatedUnivariateSpline
        if left_edge is None:
            left_edge = np.zeros(3)
        left_edge = np.array(left_edge)
        bbox = [
            [left_edge[0], left_edge[0]+box_size],
            [left_edge[1], left_edge[1]+box_size],
            [left_edge[2], left_edge[2]+box_size]
        ]
        x, y, z = np.mgrid[
            bbox[0][0]:bbox[0][1]:domain_dimensions[0]*1j,
            bbox[1][0]:bbox[1][1]:domain_dimensions[1]*1j,
            bbox[2][0]:bbox[2][1]:domain_dimensions[2]*1j,
        ]
        fields1 = ["density", "pressure", "dark_matter_density"
                   "stellar_density", "gravitational_potential"]
        fields2 = ["temperature"]
        fields3 = ["velocity_x", "velocity_y", "velocity_z"]
        units = {
            "density": "Msun/kpc**3",
            "pressure": "Msun/kpc/Myr**2",
            "dark_matter_density": "Msun/kpc**3",
            "stellar_density": "Msun/kpc**3",
            "temperature": "K",
            "gravitational_potential": "kpc**2/Myr**2",
            "velocity_x": "kpc/Myr",
            "velocity_y": "kpc/Myr",
            "velocity_z": "kpc/Myr",
            "magnetic_field_strength": "G"
        }
        fields = fields1+fields2
        data = {}
        for i, profile in enumerate(self.profiles):
            p = ClusterModel.from_h5_file(profile)
            xx = x-self.center.d[i][0]
            yy = y-self.center.d[i][1]
            zz = z-self.center.d[i][2]
            rr = np.sqrt(xx*xx+yy*yy+zz*zz)
            fd = InterpolatedUnivariateSpline(p["radius"].d,
                                              p["density"].d)
            for field in fields:
                if field not in p:
                    continue
                if field not in data:
                    data[field] = (
                        np.zeros(domain_dimensions), units[field]
                    )
                f = InterpolatedUnivariateSpline(p["radius"].d,
                                                 p[field].d)
                if field in fields1:
                    data[field][0] += f(rr)
                elif field in fields2:
                    data[field][0] += f(rr)*fd(rr)
            for field in fields3:
                data[field][0] += self.velocity.d[i][0]*fd(rr)
        if "density" in data:
            for field in fields2+fields3:
                data[field][0] /= data["density"][0]
        return load_uniform_grid(data, domain_dimensions, length_unit="kpc", 
                                 bbox=bbox, mass_unit="Msun", time_unit="Myr",
                                 **kwargs)
if __name__ == '__main__':
    test = ClusterICs(
        "test",
        2,
        ["/home/ediggins/test/Newtonian_model.h5","/home/ediggins/test/AQUAL_model.h5"],
        [0,0,0],
        [0,0,0],
        num_particles={"dm":2e4}

    )

    test.setup_particle_ics()