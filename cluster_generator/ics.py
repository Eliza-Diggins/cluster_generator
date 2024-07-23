"""Initial conditions for use in simulation codes.

Notes
-----

In effect, the :py:class:`ClusterICs` class allows for the combination of individual :py:class:`model.ClusterModel` instances
into a single, interacting system for use in external codes.
"""
import os
from numbers import Number
from pathlib import Path
from typing import Collection

import numpy as np
from numpy.typing import NDArray
from ruamel.yaml import YAML
from unyt import unyt_array

from cluster_generator.model import ClusterModel
from cluster_generator.particles import ClusterParticles, recognized_particle_types
from cluster_generator.utilities.types import (
    PRNG,
    MaybeUnitVector,
    Self,
    ensure_list,
    ensure_ytarray,
)
from cluster_generator.utilities.utils import parse_prng


def compute_centers_for_binary(
    center: Collection[float], d: float, b: float, a: float = 0.0
) -> tuple[NDArray[float], NDArray[float]]:
    r"""Given a common center and distance parameters, calculate the central positions of
    two clusters.

    First, the separation along the x-direction is determined
    by:

    .. math::

        x = \sqrt{d^2 - b^2 - a^2}

    where :math:`d` is the distance between the two clusters, :math:`b` is the
    impact parameter in the y-direction, and :math:`a` is the impact
    parameter in the z-direction. So the resulting centers are
    calculated as:

    .. code-block:: python

        center1 = [center-0.5*sep_x, center-0.5*b, center-0.5*a]
        center2 = [center+0.5*sep_x, center+0.5*b, center+0.5*a]

    Parameters
    ----------
    center : array-like
        The center from which the distance parameters for
        the two clusters will be calculated. This should be a ``(3,)`` array of
        coordinates.

    d : float
        The distance between the two clusters, in kpc.
    b : float
        The impact parameter in the y-direction, in kpc.
    a : float, optional
        The impact parameter in the z-direction, in kpc.
        Default: 0.0

    Examples
    --------

    If we have 2 clusters separated by 5000 kpc and with an impact parameter of 3000 kpc, then we expect that the
    :math:`x` separation should be 4000 kpc. Thus,

    >>> compute_centers_for_binary([0,0,0],5000,3000)
    (array([-2000., -1500.,     0.]), array([2000., 1500.,    0.]))
    """
    center = np.array(center)
    d = np.sqrt(d * d - b * b - a * a)
    diff = np.array([d, b, a])
    center1 = center - 0.5 * diff
    center2 = center + 0.5 * diff
    return center1, center2


class ClusterICs:
    """Class representing a complete set of initial conditions for a simulation."""

    _massfields = [
        "dark_matter_mass",
        "stellar_mass",
        "gas_mass",
        "tracer_mass",
    ]

    _ic_particle_types = [k for k in recognized_particle_types if k != "black_hole"]
    # These are the virialized particle species which should be generated from models. BH are excluded because they
    # aren't virialized.

    def __init__(
        self,
        basename: str,
        num_halos: int,
        models: Collection[str],
        centers: list[MaybeUnitVector],
        velocities: list[MaybeUnitVector],
        num_particles: dict[str, int] = None,
        mag_file: str | Path = None,
        particle_files=None,
        r_max: NDArray[float] | float = 20000.0,
        r_max_tracer: NDArray[float] | float = None,
        storage_directory: str | Path = None,
    ):
        """Initialize a :py:class:`ClusterICs` instance.

        Parameters
        ----------
        basename: str
            The base name for this initial conditions file.
        num_halos: int
            The number of halos in this initial conditions instance.
        models: Collection[str]
            The paths to each of the constituent clusters.
        centers: array-like
            The center of each of the clusters. Should be ``(N,3)``.
        velocities: array-like
            The velocity of each of the clusters. Should be ``(N,3)``.
        num_particles: dict of str: int, optional
            The number of particles associated with each of the particle species.
        mag_file: str
            The file containing the magnetic field prescription.
        particle_files: list[str]
            The files containing the particle data for each of the IC components.
        r_max: array-like or float, optional
            The maximal radius for each of the constituent clusters.
        r_max_tracer: array-like or float, optional
            The maximal radius for the tracer particles.
        storage_directory: str or Path, optional
            If specified, then all of the generated particle files will be kept together in this directory.
            If the directory doesn't exist, we will attempt to create it.
        """
        # Setting basic attributes
        self.basename: str = basename
        """ str: The name of this IC system."""
        self.num_halos: int = num_halos
        """ int: The number of halos present in the IC system."""
        self.models: list[Path] = [Path(i) for i in ensure_list(models)]
        """ list of Path: The paths of the constituent profiles."""

        # Managing the IC directory
        self._directory: Path | None = (
            Path(storage_directory) if storage_directory is not None else None
        )

        # Adding some redundancy for odd user input of centers and velocities.
        if (
            self.num_halos == 1
        ):  # User may provide centers as single list / array of size 3.
            if len(centers) == 3:
                centers = [centers]
            if len(velocities) == 3:
                velocities = [velocities]

        self.centers: list[unyt_array] = [
            ensure_ytarray(c, "kpc") for c in ensure_list(centers)
        ]
        """ list of unyt_array: The centers of the constituent profiles.

        This should be a ``(N,3)`` list where ``N`` is the number of profiles included.
        """
        self.velocities: list[unyt_array] = [
            ensure_ytarray(v, "kpc/Myr") for v in ensure_list(velocities)
        ]
        """ list of unyt_array: The velocities of the constituent profiles.

        This should be a ``(N,3)`` list where ``N`` is the number of profiles included.
        """
        self.mag_file: Path = Path(mag_file) if mag_file is not None else None
        """ Path: The file containing the magnetic field information for the ICs."""

        # Check that the halos all match up

        if not (
            self.num_halos
            == len(self.models)
            == len(self.centers)
            == len(self.velocities)
        ):
            raise ValueError(
                f"Expected {self.num_halos} halos, but had {len(self.models)} clusters and center / velocity of shape {len(self.centers)},{len(self.velocities)}"
            )

        # Fixing attribute formats.
        if isinstance(r_max, Number):
            r_max = [r_max] * num_halos
        self.r_max: NDArray[float] = np.array(r_max)
        """:py:class:`np.ndarray`: The maximal radii for each of the constituent
        clusters."""

        if r_max_tracer is None:
            r_max_tracer = r_max
        if isinstance(r_max_tracer, Number):
            r_max_tracer = [r_max_tracer] * num_halos
        self.r_max_tracer: NDArray[float] = np.array(r_max_tracer)
        """:py:class:`np.ndarray`: The maximal (tracer) radii for each of the
        constituent clusters."""

        # Managing number of particles
        if num_particles is None:
            _np = {k: 0 for k in self.__class__._ic_particle_types}
        else:
            _np = {
                k: num_particles[k] if k in num_particles else 0
                for k in self.__class__._ic_particle_types
            }

        _res = self._determine_num_particles(
            _np
        )  # --> Determines how to partition the particle counts
        self.num_particles: dict[str, list[int]] = _res[0]
        """ dict: The number of particles of each species in each cluster.

        This is calculated at instantiation from the provided number of particles and the cluster masses.
        """
        self.total_masses: dict[str, float] = _res[1]
        """ dict: The total mass of each particle species in the entire IC.
        """

        self.particle_files: list[Path | None] = [None] * self.num_halos
        """ list: The paths of the different particle files for each cluster.
        """

        if particle_files is not None:
            self.particle_files[: self.num_halos] = [Path(i) for i in particle_files]

    def __repr__(self):
        return f"<ClusterICs: {self.num_halos} models>"

    def __str__(self):
        return self.__repr__()

    @property
    def directory(self) -> Path | None:
        """The storage directory for this IC, where new particle files are stored."""
        if self._directory is None:
            return None

        # check it exists
        if not self._directory.exists():
            self._directory.mkdir(parents=True)

        return self._directory

    @directory.setter
    def directory(self, value: str | Path):
        self._directory = Path(value)

        if not self._directory.exists():
            self._directory.mkdir(parents=True)

    def _determine_num_particles(
        self, n_parts: dict[str, int]
    ) -> tuple[dict[str, list[int]], dict[str, float]]:
        """Determines the number of particles of each type to be attributed to each
        cluster.

        Parameters
        ----------
        n_parts: dict
            The number of each type of particle to allocate in the IC.

        Returns
        -------
        num_particles: dict[str, list[int]]
            The number of particle to attribute to each of the clusters.
        total_species_masses: dict[str, float]
            The total mass of each species in the combined IC system.
        """
        from collections import defaultdict

        # Setup particle-species buffers for each cluster model
        species_masses = {species: [] for species in self.__class__._ic_particle_types}

        # determine masses for each cluster
        for i, model_file in enumerate(self.models):
            model = ClusterModel.from_h5_file(model_file)

            # Construct the truncation point indices
            idxs = model["radius"] < self.r_max[i]  # the index at r_max.
            idxst = model["radius"] < self.r_max_tracer[i]

            for species, field, idx in zip(
                self.__class__._ic_particle_types,
                self.__class__._massfields,
                [idxs, idxs, idxs, idxst],
            ):
                if field in model:
                    _mass = model[field][idx][-1].value
                else:
                    _mass = 0

                species_masses[species].append(_mass)

        total_species_masses = {k: np.sum(v) for k, v in species_masses.items()}

        # Allocating particles to each cluster based on total number of particles specified.
        num_particles = defaultdict(list)
        for i in range(self.num_halos):
            for species in self.__class__._ic_particle_types:
                if total_species_masses.get(species, 0) > 0:
                    num_particles[species].append(
                        int(
                            np.rint(
                                n_parts[species]
                                * (
                                    species_masses[species][i]
                                    / total_species_masses[species]
                                ),
                            )
                        )
                    )
                else:
                    num_particles[species].append(0)

        return num_particles, total_species_masses

    def _generate_particles(
        self,
        regenerate_particles: bool = False,
        prng: int | np.random.RandomState = None,
    ):
        """Generate particles for the initial conditions."""
        prng = parse_prng(prng)
        parts = []

        for i, model_file in enumerate(self.models):
            # Check for the existence of a particle file for this model / if we need to regenerate
            if (
                (self.particle_files[i] is not None)
                and (self.particle_files[i].exists())
                and (not regenerate_particles)
            ):
                # The particle file exists and is valid, we aren't regenerating. We can just bypass it.
                model_particles = ClusterParticles.from_file(self.particle_files[i])
                parts.append(model_particles)
                continue

            # Otherwise, we don't already have the particle information and need to proceed.
            model = ClusterModel.from_h5_file(model_file)

            model_particles = None
            for species in self.__class__._ic_particle_types:
                # Iterate through the various particle species and generate them.
                if self.num_particles[species][i] <= 0:
                    continue

                _generator = getattr(model, f"generate_{species}_particles")

                _p = _generator(
                    self.num_particles[species][i], r_max=self.r_max[i], prng=prng
                )

                if model_particles is None:
                    model_particles = _p
                else:
                    model_particles += _p

            parts.append(model_particles)
            outfile = f"{self.basename}_{str(i)}_particles.h5"

            if self.directory is not None:
                outfile = os.path.join(self.directory, outfile)

            model_particles.write_particles(outfile, overwrite=True)
            self.particle_files[i] = Path(outfile)

    def add_black_holes(
        self,
        halo_ids: list[int] | int,
        bh_mass: list[MaybeUnitVector] | MaybeUnitVector,
        pos: list[MaybeUnitVector] | MaybeUnitVector = None,
        vel: list[MaybeUnitVector] | MaybeUnitVector = None,
        use_pot_min: list[bool] | bool = False,
    ) -> None:
        """Add black holes to the specified halos.

        Parameters
        ----------
        halo_ids: list of int
            The ids (indices) of the models that should have black holes added to them.
        bh_mass: list of unyt_quantities
            The masses of the black holes for each of the halos.
        pos: list of unyt_array
            The relative position (in the model's frame of reference) of the black hole.
        vel: list of unyt_array
            The relative velocity (in the model's frame of reference) of the black hole.
        use_pot_min: list of bool
            Whether or not to use the potential minimum to set the position of the BH. This overrides the
            ``pos`` arguments.
        """
        # forcing everything to be self consistent.
        halo_ids = halo_ids if isinstance(halo_ids, list) else [int(halo_ids)]
        bh_mass = bh_mass if isinstance(bh_mass, list) else [bh_mass] * len(halo_ids)

        if len(halo_ids) == 1:
            # pos and vel may be odd because they could be single dim lists.
            if isinstance(pos, list) and len(pos) == 3:
                pos = [ensure_ytarray(pos, "kpc")]
            if isinstance(vel, list) and len(vel) == 3:
                vel = [ensure_ytarray(vel, "km/s")]

        pos = pos if isinstance(pos, list) else [pos] * len(halo_ids)
        vel = vel if isinstance(vel, list) else [vel] * len(halo_ids)
        use_pot_min = (
            use_pot_min
            if isinstance(use_pot_min, list)
            else [use_pot_min] * len(halo_ids)
        )

        # now cycle through
        for index, halo_id in enumerate(halo_ids):
            assert (
                self.particle_files[halo_id] is not None
            ), f"Cannot add BH to non-existant particle dataset (id = {halo_id})."

            _parts = ClusterParticles.from_file(self.particle_files[halo_id])

            _parts.add_black_hole(
                bh_mass[index], pos[index], vel[index], use_pot_min=use_pot_min[index]
            )

            _parts.write_particles_to_h5(self.particle_files[halo_id], overwrite=True)

    def to_file(self, filename: str | Path, overwrite: bool = False):
        r"""Write the initial conditions information to a file.

        Parameters
        ----------
        filename : string
            The file to write the initial conditions information to.
        overwrite : boolean, optional
            If True, overwrite a file with the same name. Default: False

        Notes
        -----
        The file representation for :py:class:`ClusterICs` is a ``.yaml`` file effectively redirecting the information
        to other files storing the particles and the models.
        """
        if os.path.exists(filename) and not overwrite:
            raise RuntimeError(f"{filename} exists and overwrite=False!")
        from ruamel.yaml.comments import CommentedMap

        # Setting up the header
        out = CommentedMap()
        out["basename"] = self.basename
        out.yaml_add_eol_comment("base name for ICs", key="basename")
        out["num_halos"] = self.num_halos
        out.yaml_add_eol_comment("number of halos", key="num_halos")

        # Writing the model information for each of the included models.
        for halo_id, model in enumerate(self.models):
            out[f"profile{halo_id+1}"] = model
            out[f"center{halo_id+1}"] = self.centers[halo_id].tolist()
            out[f"velocity{halo_id + 1}"] = self.velocities[halo_id].tolist()

            if self.particle_files[halo_id] is not None:
                out[f"particle_file{halo_id+1}"] = self.particle_files[halo_id]
                out.yaml_add_eol_comment(
                    f"particle file for cluster {halo_id+1}",
                    key=f"particle_file{halo_id+1}",
                )

            out.yaml_add_eol_comment(
                f"profile for cluster {halo_id+1}", key=f"profile{halo_id+1}"
            )
            out.yaml_add_eol_comment(
                f"center for cluster {halo_id+1}", key=f"center{halo_id+1}"
            )
            out.yaml_add_eol_comment(
                f"velocity for cluster {halo_id+1}", key=f"velocity{halo_id+1}"
            )

        # Determining the number of particles of each type
        for particle_species in self.__class__._ic_particle_types:
            if self.num_particles.get(particle_species, 0) > 0:
                out[f"num_{particle_species}_particles"] = self.num_particles[
                    particle_species
                ]
                out.yaml_add_eol_comment(
                    f"number of {particle_species} particles.",
                    key=f"num_{particle_species}_particles",
                )

        # Checking for a mag file.
        if self.mag_file is not None:
            out["mag_file"] = self.mag_file
            out.yaml_add_eol_comment("3D magnetic field file", key="mag_file")

        out["r_max"] = self.r_max.tolist()
        out.yaml_add_eol_comment("Maximum radii of particles", key="r_max")

        if self.num_particles.get("tracer", 0) > 0:
            out["r_max_tracer"] = self.r_max_tracer.tolist()
            out.yaml_add_eol_comment("Maximum radii of tracer particles", key="r_max")

        yaml = YAML()
        with open(filename, "w") as f:
            yaml.dump(out, f)

    @classmethod
    def from_file(cls, filename: str | Path) -> Self:
        r"""Read a :py:class:`ClusterICs` instance from a ``.yaml`` file.

        Parameters
        ----------
        filename: str
            The path to the ``.yaml`` file.

        Returns
        -------
        ClusterICs
            The resulting initial conditions object.
        """
        from ruamel.yaml import YAML

        yaml = YAML()

        with open(filename, "r") as f:
            params = yaml.load(f)

        # Loading basic attributes.
        basename = params["basename"]
        num_halos = params["num_halos"]
        profiles = [params[f"profile{i}"] for i in range(1, num_halos + 1)]
        center = [np.array(params[f"center{i}"]) for i in range(1, num_halos + 1)]
        velocity = [np.array(params[f"velocity{i}"]) for i in range(1, num_halos + 1)]
        num_particles = {
            k: params.get(f"num_{k}_particles", 0) for k in ["gas", "dm", "star"]
        }
        mag_file = params.get("mag_file", None)
        particle_files = [
            params.get(f"particle_file{i}", None) for i in range(1, num_halos + 1)
        ]
        r_max = params.get("r_max", 20000.0)
        r_max_tracer = params.get("r_max_tracer", r_max)

        return cls(
            basename,
            num_halos,
            profiles,
            center,
            velocity,
            num_particles=num_particles,
            mag_file=mag_file,
            particle_files=particle_files,
            r_max=r_max,
            r_max_tracer=r_max_tracer,
        )

    def setup_particle_ics(
        self, regenerate_particles: bool = False, prng: PRNG = None
    ) -> ClusterParticles:
        r"""From a set of cluster models and their relative positions and velocities, set
        up initial conditions for use with SPH codes.

        Parameters
        ----------
        regenerate_particles: bool
            If ``True``, then existing particle files are overwritten and resampled.
        prng: RandomState or int
            The pseudo-random number generator seed or instance.

        Returns
        -------
        ClusterParticles
            The combined particle dataset ready for use in SPH codes.

        Notes
        -----

        This routine will either generate a single cluster or will combine
        two or three clusters together. If more than one cluster is
        generated, the gas particles will have their densities set by
        adding the densities from the overlap of the two particles
        together, and will have their thermal energies and velocities
        set by mass-weighting them from the two profiles.
        """
        from cluster_generator.particles import combine_clusters

        # Pulling model and particle files.
        models = [ClusterModel.from_h5_file(hf) for hf in self.models]

        self._generate_particles(regenerate_particles=regenerate_particles, prng=prng)
        parts = [ClusterParticles.from_h5_file(hf) for hf in self.particle_files]

        centers, velocities = [unyt_array(c, "kpc") for c in self.centers], [
            unyt_array(v, "km/s") for v in self.velocities
        ]

        all_parts = combine_clusters(parts, models, centers, velocities)

        return all_parts

    def resample_particle_ics(
        self, particles: ClusterParticles, passive_scalars: list[str] = None
    ) -> ClusterParticles:
        r"""Given a Gadget-HDF5-like initial conditions file which has been output from
        some type of relaxation process (such as making a glass or using MESHRELAX in
        the case of Arepo), resample the density, thermal energy, and velocity fields
        onto the gas particles/cells from the initial hydrostatic profiles.

        Parameters
        ----------
        particles: ClusterParticles
            The particle dataset to resample onto.
        passive_scalars: list of str
            Any passive scalar fields to pull from the models and map onto the particles.
        """
        from cluster_generator.particles import sample_from_clusters

        models = [ClusterModel.from_h5_file(hf) for hf in self.models]
        centers, velocities = [unyt_array(c, "kpc") for c in self.centers], [
            unyt_array(v, "km/s") for v in self.velocities
        ]

        new_parts = sample_from_clusters(
            particles,
            models,
            centers,
            velocities,
            resample=True,
            passive_scalars=passive_scalars,
        )

        return new_parts

    def create_dataset(
        self,
        filename: str | Path,
        domain_dimensions: Collection[int] = (512, 512, 512),
        left_edge: Collection[Number] | unyt_array | None = None,
        box_size: Collection[Number] | unyt_array | None = None,
        overwrite: bool = False,
        chunksize: int = 64,
    ) -> str | Path:
        r"""Construct a ``yt`` dataset object from this model on a uniformly spaced grid.

        Parameters
        ----------
        filename: str or :py:class:`pathlib.Path`
            The path at which to generate the underlying HDF5 datafile.
        domain_dimensions: Collection of int, optional
            The size of the uniform grid along each axis of the domain. If specified, the argument must be an iterable type with
            shape ``(3,)``. Each element should be an ``int`` specifying the number of grid cells to place along that axis. By default,
            the selected value is ``(512,512,512)``.
        left_edge: Collection of float or :py:class:`unyt.unyt_array`, optional
            The left-most edge of the uniform grid's domain. In conjunction with ``box_size``, this attribute specifies the position of
            the model in the box and the amount of the model which is actually written to the disk. If specified, ``left_edge`` should be a
            length 3 iterable with each of the entries representing the minimum value of the respective axis. If elements of the iterable have units, or
            the array is a :py:class:`unyt.unyt_array` instance, then the units will be interpreted automatically; otherwise, units are assumed to be
            kpc. By default, the left edge is determined such that the resulting grid contains the full radial domain of the :py:class:`ClusterModel`.
        box_size: Collection of float or :py:class:`unyt.unyt_array`, optional
            The length of the grid along each of the physical axes. Along with ``left_edge``, this argument determines the positioning of the grid and
            the model within it. If specified, ``box_size`` should be a length 3 iterable with each of the entries representing the length
            of the grid along the respective axis. If elements of the iterable have units, or the array is a :py:class:`unyt.unyt_array` instance,
             then the units will be interpreted automatically; otherwise, units are assumed to be kpc.
            By default, the ``box_size`` is determined such that the resulting grid contains the full radial domain of the :py:class:`ClusterModel`.
        overwrite: bool, optional
            If ``False`` (default), then an error is raised if ``filename`` already exists. Otherwise, ``filename`` will be deleted and overwritten
            by this method.
        chunksize: int, optional
            The maximum chunksize for subgrid operations. Lower values with increase the execution time but save memory. By default,
            chunks contain no more that :math:`64^3` cells (``chunksize=64``).

        Returns
        -------
        str
            The path to the output dataset file.

        Notes
        -----

        Generically, converting a :py:class:`ClusterModel` instance to a valid ``yt`` dataset occurs in two steps. In the first step,
        the dataset is written to disk on a uniform grid (or, more generally, an AMR grid). From this grid, ``yt`` can then interpret the
        data and construct a dataset from there.

        Because constructing the underlying grid is a memory intensive procedure, this method utilizes the HDF5 structure as an intermediary
        (effectively using the disk for VRAM).
        """
        from cluster_generator.data_structures import YTHDF5

        if not left_edge:
            left_edge = 3 * [-np.amax(self.r_max)]
        if not box_size:
            box_size = 2 * np.amax(self.r_max)

        bbox = [le + box_size for le in left_edge]

        ds_obj = YTHDF5.build(
            filename,
            domain_dimensions,
            bbox,
            chunksize=chunksize,
            overwrite=overwrite,
        )
        ds_obj.add_ICs(self)

        return ds_obj.filename
