"""Initial conditions and cluster model particle management module."""

from collections import OrderedDict, defaultdict
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Collection

import h5py
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from unyt import Unit, unyt_array

from cluster_generator.utilities.docs import deprecated
from cluster_generator.utilities.logging import mylog
from cluster_generator.utilities.types import (
    MaybeUnitScalar,
    MaybeUnitVector,
    Self,
    ensure_list,
    ensure_ytarray,
    ensure_ytquantity,
)

if TYPE_CHECKING:
    from cluster_generator import ClusterModel


recognized_particle_types: list[str] = ["dm", "gas", "star", "tracer", "black_hole"]
""" list of str: The 5 standard particle types in ``cluster_generator``."""

gadget_fields = {
    "dm": ["Coordinates", "Velocities", "Masses", "ParticleIDs", "Potential"],
    "gas": [
        "Coordinates",
        "Velocities",
        "Masses",
        "ParticleIDs",
        "InternalEnergy",
        "MagneticField",
        "Density",
        "Potential",
        "PassiveScalars",
    ],
    "star": ["Coordinates", "Velocities", "Masses", "ParticleIDs", "Potential"],
    "black_hole": ["Coordinates", "Velocities", "Masses", "ParticleIDs"],
    "tracer": ["Coordinates"],
}

gadget_field_map = {
    "Coordinates": "particle_position",
    "Velocities": "particle_velocity",
    "Masses": "particle_mass",
    "Density": "density",
    "Potential": "potential_energy",
    "InternalEnergy": "thermal_energy",
    "MagneticField": "magnetic_field",
}

gadget_field_units = {
    "Coordinates": "kpc",
    "Velocities": "km/s",
    "Masses": "1e10*Msun",
    "Density": "1e10*Msun/kpc**3",
    "InternalEnergy": "km**2/s**2",
    "Potential": "km**2/s**2",
    "PassiveScalars": "",
    "MagneticField": "1e5*sqrt(Msun)*km/s/(kpc**1.5)",
}

ptype_map = OrderedDict(
    [
        ("PartType0", "gas"),
        ("PartType1", "dm"),
        ("PartType2", "tracer"),
        ("PartType4", "star"),
        ("PartType5", "black_hole"),
    ]
)

rptype_map = OrderedDict([(v, k) for k, v in ptype_map.items()])


class ClusterParticles:
    """Class representation for particles in galaxy cluster models.

    Notes
    -----
    The :py:class:`ClusterParticles` class is effectively a collection of fields similar to the :py:class:`model.ClusterModel` class.
    For each particle type, particular fields can be specified. Each field entry is a ``(N,)`` array of values where ``N`` is the number
    of such particles being represented.
    """

    def __init__(
        self, particle_types: list[str] | str, fields: dict[tuple[str, str], unyt_array]
    ):
        """Initialize a :py:class:`ClusterParticles` object.

        Parameters
        ----------
        particle_types: list
            List of included particle types. May include ``dm``, ``gas``, ``tracer``, ``star``, or ``black_hole``.
        fields: dict
            The fields for each of the particle types.
        """
        # Setting the particle class's attributes
        self.particle_types: list[str] = ensure_list(particle_types)
        """ list of str: The types of particles included in this object.

        The valid particle types are ``['dm','gas','tracer','star','black_hole']``.
        """
        self.fields: dict[tuple[str, str], unyt_array] = fields
        """ dict: The fields for particles.

        Fields are formatted as ``particle_type, field_name`` with each value being an :py:class:`unyt_array` instance.
        """

        # enforce the validity of the particle types.
        assert all(
            pt in recognized_particle_types for pt in self.particle_types
        ), f"{[pt for pt in self.particle_types if pt not in recognized_particle_types]} are not recognized particle types."

        self.passive_scalars = []

    def __getitem__(self, key: tuple[str, str]) -> unyt_array:
        return self.fields[key]

    def __setitem__(self, key: tuple[str, str], value: unyt_array):
        # Ensure that the particle type is allowed
        assert (
            key[0] in recognized_particle_types
        ), f"{key[0]} is not a valid particle type."

        # Add the particle type to our included particle types.
        if key[0] not in self.particle_types:
            self.particle_types.append(key[0])

        # set the field
        self.fields[key] = value

    def __delitem__(self, key: tuple[str, str]):
        del self.fields[key]

    def __repr__(self):
        return "<ClusterParticles Object>"

    def __str__(self):
        return f"<ClusterParticles PTypes={len(self.particle_types)}>"

    def __add__(self, other: Self | int) -> Self:
        # __add__ compliance with np.sum / sum built-in function - begins with 0.
        if other == 0:
            return self

        base_fields = self.fields.copy()

        # Concatenate fields, creating them if necessary
        for field in other.fields:
            if field in base_fields:
                base_fields[field] = ensure_ytarray(
                    np.concatenate(
                        (
                            base_fields[field].d,
                            other.fields[field].to_value(base_fields[field].units),
                        )
                    ),
                    base_fields[field].units,
                )
                # In unyt>3.0, unit wrangling here isn't needed as np.concatenate handles units properly, but
                # for backwards compatibility, we use the verbose approach.

            else:
                base_fields[field] = other[field]
        particle_types = list(set(self.particle_types + other.particle_types))
        return ClusterParticles(particle_types, base_fields)

    def __radd__(self, other):
        return self.__add__(other)

    def __contains__(self, item: tuple[str, str]) -> bool:
        return item in self.fields

    def __len__(self) -> dict[str, int]:
        return self.num_particles

    @property
    def field_names(self) -> dict[str, list[str]]:
        """Dictionary of fields for each of the included particle types.

        For each of the particle types, any of a number of included fields may exist. The :py:attr:`ClusterParticles.field_names` attribute
        collects all of these included field names for each of the particle types.
        """
        fns = defaultdict(list)

        for field in self.fields:
            fns[field[0]].append(field[1])

        return fns

    @property
    def num_particles(self) -> dict[str, int]:
        """The number of particles (for each particle type) contained in this
        :py:class:`ClusterParticles` object."""
        np = {k: 0 for k in recognized_particle_types}  # --> empty buffer

        for ptype in self.particle_types:
            np[ptype] = self.fields[ptype, "particle_mass"].size

        return np

    @property
    def num_passive_scalars(self):
        return len(self.passive_scalars)

    def keys(self) -> Collection[tuple[str, str]]:
        """Get the field keys for the particle object."""
        return self.fields.keys()

    def _clip_to_box(self, ptype: str, box_size: Number):
        """Create a mask to cut out particles not within a specific box size."""
        pos = self.fields[ptype, "particle_position"].to_value(
            "kpc"
        )  # -> convert to kpc to enforce our unit convention.
        return ~np.logical_or((pos < 0.0).any(axis=1), (pos > box_size).any(axis=1))

    def drop_ptypes(self, ptypes: list[str] | str):
        """Drop one or several particle types from this object.

        Parameters
        ----------
        ptypes: list of str or str
            The particle types to drop.
        """
        ptypes = ensure_list(ptypes)

        for ptype in ptypes:
            self.particle_types.remove(ptype)

            _removed_fields = [(ptype, f) for f in self.field_names[ptype]]
            for field in _removed_fields:
                del self.fields[field]

    def make_radial_cut(
        self,
        r_max: MaybeUnitScalar,
        center: MaybeUnitVector = None,
        ptypes: list[str] | str = None,
    ):
        """Make a radial cut on particles. All particles outside a certain radius will
        be removed.

        Parameters
        ----------
        r_max : float
            The maximum radius of the particles in kpc.
        center : array-like, optional
            The center coordinate of the system of particles to define
            the radius from, in units of kpc. Default: [0.0, 0.0, 0.0]
        ptypes : list of strings, optional
            The particle types to perform the radial cut on. If
            not set, all will be exported.
        """
        # Enforce units and then reduce to scalar (this ensures that non-kpc units are converted correctly).
        r_max, center = (
            ensure_ytquantity(r_max, "kpc").d,
            ensure_ytarray(center, "kpc").d,
        )

        if center is None:
            center = np.array([0.0] * 3)
        if ptypes is None:
            ptypes = self.particle_types

        ptypes = ensure_list(ptypes)

        # Make the radial cuts. Identify the relevant ids and then cut from all the fields.
        for part in ptypes:
            # Identify the kept ids.
            cidx = (
                (self[part, "particle_position"].to_value("kpc") - center) ** 2
            ).sum(axis=1) <= r_max**2

            for field in self.field_names[part]:
                self.fields[part, field] = self.fields[part, field][cidx]

    def make_box_cut(
        self,
        bbox: MaybeUnitVector,
        ptypes: list[str] | str = None,
    ):
        """Make a radial cut on particles. All particles outside a certain radius will
        be removed.

        Parameters
        ----------
        bbox: array-like
            A ``(3,2)`` array (either with or without units) for the bounding box to constrain the particles to. If a
            unit is not given, it is assumed to be kpc.
        ptypes : list of strings, optional
            The particle types to perform the radial cut on. If
            not set, all will be exported.
        """
        # Enforce units and then reduce to scalar (this ensures that non-kpc units are converted correctly).
        bbox = ensure_ytarray(bbox, "kpc")

        if ptypes is None:
            ptypes = self.particle_types

        ptypes = ensure_list(ptypes)

        # Make the radial cuts. Identify the relevant ids and then cut from all the fields.
        for part in ptypes:
            # Identify the kept ids.
            cidx = (
                np.sum(
                    (self[part, "particle_position"].to_value("kpc") - bbox[:, 0])
                    > bbox[:, 1] - bbox[:, 0],
                    axis=1,
                )
                < 1
            )

            for field in self.field_names[part]:
                self.fields[part, field] = self.fields[part, field][cidx]

    def make_boxsize_cut(
        self, boxsize: MaybeUnitVector, ptypes: list = None, centered: bool = False
    ):
        """Make a box cut using a specified boxsize.

        Parameters
        ----------
        boxsize: unyt_quantity
            The boxsize to use to make the cut.
        ptypes: list of str, optional
            The particle types to include. If not set, all will be used.
        centered: bool, optional
            If ``True``, then the box is placed centered on the axis. If ``False`` (default), then the box is constrained
            to the positive octant.
        """
        boxsize = ensure_ytquantity(boxsize, "kpc")
        bbox = np.array([[0, 0, 0], 3 * [boxsize.d]]).T

        # correct for the bounding box.
        if centered:
            bbox -= boxsize.d / 2

        # Process this as a standard boxcut.
        self.make_box_cut(bbox, ptypes=ptypes)

    def add_black_hole(
        self,
        bh_mass: MaybeUnitScalar,
        pos: MaybeUnitVector = None,
        vel: MaybeUnitVector = None,
        use_pot_min: bool = False,
    ):
        r"""Add a black hole particle to the set of cluster particles.

        Parameters
        ----------
        bh_mass : unyt_quantity
            The mass of the black hole particle. If specified without units, they are
            assumed to be solar masses.
        pos : array-like, optional
            The position of the particle, assumed to be in units of
            kpc if units are not given. If use_pot_min=True this
            argument is ignored. Default: None, in which case the
            particle position is [0.0, 0.0, 0.0] kpc.
        vel : array-like, optional
            The velocity of the particle, assumed to be in units of
            kpc/Myr if units are not given. If use_pot_min=True this
            argument is ignored. Default: None, in which case the
            particle velocity is [0.0, 0.0, 0.0] kpc/Myr.
        use_pot_min : boolean, optional
            If True, use the dark matter particle with the minimum
            value of the gravitational potential to determine the
            position and velocity of the black hole particle. Default:
            False
        """
        mass = ensure_ytquantity(bh_mass, "msun")

        # Determining the particle's positioning
        if use_pot_min:
            if ("dm", "potential_energy") not in self.fields:
                raise KeyError("('dm', 'potential_energy') is not available!")

            idx = np.argmin(self.fields["dm", "potential_energy"])
            pos = unyt_array(self.fields["dm", "particle_position"][idx]).reshape(1, 3)
            vel = unyt_array(self.fields["dm", "particle_velocity"][idx]).reshape(1, 3)
        else:
            if pos is None:
                pos = unyt_array(np.zeros((1, 3)), "kpc")
            if vel is None:
                vel = unyt_array(np.zeros((1, 3)), "kpc/Myr")

            pos = ensure_ytarray(pos, "kpc").reshape(1, 3)
            vel = ensure_ytarray(vel, "kpc/Myr").reshape(1, 3)

        # Determining / setting the fields.
        if "black_hole" not in self.particle_types:
            self.particle_types.append("black_hole")
            self.fields["black_hole", "particle_position"] = pos
            self.fields["black_hole", "particle_velocity"] = vel
            self.fields["black_hole", "particle_mass"] = mass
        else:
            uappend = lambda x, y: unyt_array(np.append(x, y, axis=0).v, x.units)

            self.fields["black_hole", "particle_position"] = uappend(
                self.fields["black_hole", "particle_position"], pos
            )
            self.fields["black_hole", "particle_velocity"] = uappend(
                self.fields["black_hole", "particle_velocity"], vel
            )
            self.fields["black_hole", "particle_mass"] = uappend(
                self.fields["black_hole", "particle_mass"], mass
            )

    @classmethod
    def from_fields(cls, fields: dict[tuple[str, str], unyt_array]) -> Self:
        """Initialize a :py:class:`ClusterParticles` instance from existing fields.

        Parameters
        ----------
        fields: dict
            The fields from which to load this instance from.

        Returns
        -------
        ClusterParticles
            The resulting particle instance.
        """
        particle_types = []
        for key in fields:
            if key[0] not in particle_types:
                particle_types.append(key[0])

        return cls(particle_types, fields)

    @classmethod
    def from_file(cls, filename: str | Path, ptypes: list[str] | str = None) -> Self:
        r"""Generate cluster particles from an HDF5 file.

        Parameters
        ----------
        filename : string
            The name of the file to read the model from.
        ptypes: list of str or str
            The particle types to load from file.

        Examples
        --------

        .. code-block:: python

            from cluster_generator import ClusterParticles
            dm_particles = ClusterParticles.from_file("dm_particles.h5")
        """
        filename = Path(filename)
        assert filename.exists(), f"{filename} does not exist!"

        names = {}
        with h5py.File(filename, "r") as f:
            if ptypes is None:
                ptypes = list(f.keys())
            ptypes = ensure_list(ptypes)
            for ptype in ptypes:
                names[ptype] = list(f[ptype].keys())
        fields = OrderedDict()
        for ptype in ptypes:
            for field in names[ptype]:
                if field == "particle_index":
                    with h5py.File(filename, "r") as f:
                        fields[ptype, field] = f[ptype][field][:]
                else:
                    a = unyt_array.from_hdf5(
                        str(filename), dataset_name=field, group_name=ptype
                    )
                    fields[ptype, field] = unyt_array(
                        a.d.astype("float64"), str(a.units)
                    ).in_base("galactic")
        return cls(ptypes, fields)

    @classmethod
    def from_h5_file(cls, filename: str | Path, ptypes: list[str] | str = None) -> Self:
        """Load the particles from an HDF5 file.

        Parameters
        ----------
        filename: str
            The filename to load from.
        ptypes: list of str, optional
            The particle types to load.

        Returns
        -------
        ClusterParticles
        """
        return cls.from_file(filename, ptypes=ptypes)

    @classmethod
    def from_gadget_file(
        cls, filename: str | Path, ptypes: list[str] | str = None
    ) -> Self:
        """Read in particle data from a Gadget (or Arepo, GIZMO, etc.) snapshot.

        Parameters
        ----------
        filename : string
            The name of the file to read from.
        ptypes : string or list of strings, optional
            The particle types to read from the file, either
            a single string or a list of strings. If None,
            all particle types will be read from the file.

        Examples
        --------

        .. code-block:: python

            from cluster_generator import ClusterParticles
            ptypes = ["gas", "dm"]
            particles = ClusterParticles.from_gadget_file("snapshot_060.h5", ptypes=ptypes)
        """
        fields = OrderedDict()
        f = h5py.File(filename, "r")
        particle_types = []
        if ptypes is None:
            ptypes = [k for k in f if k.startswith("PartType")]
        else:
            ptypes = ensure_list(ptypes)
            ptypes = [rptype_map[k] for k in ptypes]
        for ptype in ptypes:
            my_ptype = ptype_map[ptype]
            particle_types.append(my_ptype)
            g = f[ptype]
            for field in gadget_fields[my_ptype]:
                if field in g:
                    if field == "ParticleIDs":
                        fields[my_ptype, "particle_index"] = g[field][:]
                    else:
                        fd = gadget_field_map[field]
                        units = gadget_field_units[field]
                        fields[my_ptype, fd] = unyt_array(
                            g[field], units, dtype="float64"
                        ).in_base("galactic")
            if "Masses" not in g:
                n_ptype = g["ParticleIDs"].size
                units = gadget_field_units["Masses"]
                n_type = int(ptype[-1])
                fields[my_ptype, "particle_mass"] = unyt_array(
                    [f["Header"].attrs["MassTable"][n_type]] * n_ptype, units
                ).in_base("galactic")
        f.close()
        return cls(particle_types, fields)

    def write_particles(self, output_filename: str | Path, overwrite: bool = False):
        """Write the particles to an HDF5 file.

        Parameters
        ----------
        output_filename : string
            The file to write the particles to.
        overwrite : boolean, optional
            Overwrite an existing file with the same name. Default False.
        """
        if Path(output_filename).exists() and not overwrite:
            raise IOError(
                f"Cannot create {output_filename}. It exists and overwrite=False."
            )

        with h5py.File(output_filename, "w") as f:
            for ptype in self.particle_types:
                f.create_group(ptype)
        for field in self.fields:
            if field[1] == "particle_index":
                with h5py.File(output_filename, "r+") as f:
                    g = f[field[0]]
                    g.create_dataset("particle_index", data=self.fields[field])
            else:
                self.fields[field].write_hdf5(
                    output_filename, dataset_name=field[1], group_name=field[0]
                )

    def write_particles_to_h5(self, output_filename, overwrite=False):
        """Create an HDF5 file representing this :py:class:`ClusterParticles` instance.

        Parameters
        ----------
        output_filename: str
            The output filename.
        overwrite: boolean, optional
            If ``True``, allow overwritting of existing file.
        """
        self.write_particles(output_filename, overwrite=overwrite)

    def set_field(
        self,
        ptype: str,
        name: str,
        value: MaybeUnitVector,
        units: str | Unit = None,
        add: bool = False,
        passive_scalar: bool = False,
    ):
        """Add or update a particle field using a unyt_array. The array will be checked
        to make sure that it has the appropriate size.

        Parameters
        ----------
        ptype : string
            The particle type of the field to add or update.
        name : string
            The name of the field to add or update.
        value : unyt_array
            The particle field itself--an array with the same
            shape as the number of particles.
        units : string, optional
            The units to convert the field to. Default: None,
            indicating the units will be preserved.
        add : boolean, optional
            If True and the field already exists, the values
            in the array will be added to the already existing
            field array.
        passive_scalar : boolean, optional
            If set, the field to be added is a passive scalar.
            Default: False
        """
        if not isinstance(value, unyt_array):
            value = unyt_array(value, "dimensionless")
        num_particles = self.num_particles[ptype]
        exists = (ptype, name) in self.fields
        if value.shape[0] == num_particles:
            if exists:
                if add:
                    self.fields[ptype, name] += value
                else:
                    mylog.warning(f"Overwriting field ({ptype}, {name}).")
                    self.fields[ptype, name] = value
            else:
                if add:
                    raise RuntimeError(
                        f"Field ({ptype}, {name}) does not " f"exist and add=True!"
                    )
                else:
                    self.fields[ptype, name] = value
                if passive_scalar and ptype == "gas":
                    self.passive_scalars.append(name)
            if units is not None:
                self.fields[ptype, name].convert_to_units(units)
        else:
            raise ValueError(
                f"The length of the array needs to be {num_particles} particles!"
            )

    def add_offsets(
        self,
        r_ctr: MaybeUnitVector,
        v_ctr: MaybeUnitVector,
        ptypes: list[str] | str = None,
    ):
        """Add offsets in position and velocity to the cluster particles, which can be
        added to one or more particle types.

        Parameters
        ----------
        r_ctr : array-like
            A 3-element list, NumPy array, or unyt_array of the coordinates
            of the new center of the particle distribution. If units are not
            given, they are assumed to be in kpc.
        v_ctr : array-like
            A 3-element list, NumPy array, or unyt_array of the coordinates
            of the new bulk velocity of the particle distribution. If units
            are not given, they are assumed to be in kpc/Myr.
        ptypes : string or list of strings, optional
            A single string or list of strings indicating the particle
            type(s) to be offset. Default: None, meaning all of the
            particle types will be offset. This should not be used in
            normal circumstances.
        """
        if ptypes is None:
            ptypes = self.particle_types

        ptypes = ensure_list(ptypes)
        r_ctr = ensure_ytarray(r_ctr, "kpc")
        v_ctr = ensure_ytarray(v_ctr, "kpc/Myr")
        for ptype in ptypes:
            self.fields[ptype, "particle_position"] += r_ctr
            self.fields[ptype, "particle_velocity"] += v_ctr

    def _write_gadget_fields(self, ptype, h5_group, idxs, dtype):
        for field in gadget_fields[ptype]:
            if field == "ParticleIDs":
                continue
            if field == "PassiveScalars" and ptype == "gas":
                if self.num_passive_scalars > 0:
                    data = np.stack(
                        [self[ptype, s].d for s in self.passive_scalars], axis=-1
                    )
                    h5_group.create_dataset("PassiveScalars", data=data)
            else:
                my_field = gadget_field_map[field]
                if (ptype, my_field) in self.fields:
                    units = gadget_field_units[field]
                    fd = self.fields[ptype, my_field]
                    data = fd[idxs].to(units).d.astype(dtype)
                    h5_group.create_dataset(field, data=data)

    def write_to_gadget_file(
        self,
        ic_filename: str | Path,
        box_size: MaybeUnitScalar,
        dtype: str = "float32",
        overwrite: bool = False,
        code: str = None,
    ):
        """Write the particles to a file in the HDF5 Gadget format which can be used as
        initial conditions for a simulation.

        Parameters
        ----------
        ic_filename : string
            The name of the file to write to.
        box_size : float
            The width of the cubical box that the initial condition
            will be within in units of kpc.
        dtype : string, optional
            The datatype of the fields to write, either 'float32' or
            'float64'. Default: 'float32'
        overwrite : boolean, optional
            Whether to overwrite an existing file. Default: False
        code : string, optional
            If specified, additional information will be written to
            the file so that it can be identified by yt as belonging
            to a specific frontend. Default: None
        """
        box_size = ensure_ytquantity(box_size, "kpc").d
        if Path(ic_filename).exists() and not overwrite:
            raise IOError(
                f"Cannot create {ic_filename}. It exists and " f"overwrite=False."
            )
        num_particles = {}
        npart = 0
        mass_table = np.zeros(6)
        f = h5py.File(ic_filename, "w")
        for ptype in self.particle_types:
            gptype = rptype_map[ptype]
            idxs = self._clip_to_box(ptype, box_size)
            num_particles[ptype] = idxs.sum()
            g = f.create_group(gptype)
            self._write_gadget_fields(ptype, g, idxs, dtype)
            ids = np.arange(num_particles[ptype]) + 1 + npart
            g.create_dataset("ParticleIDs", data=ids.astype("uint32"))
            npart += num_particles[ptype]
            if ptype in ["star", "dm", "black_hole"]:
                mass_table[int(rptype_map[ptype][-1])] = g["Masses"][0]
        f.flush()
        hg = f.create_group("Header")
        hg.attrs["Time"] = 0.0
        hg.attrs["Redshift"] = 0.0
        hg.attrs["BoxSize"] = box_size
        hg.attrs["Omega0"] = 0.0
        hg.attrs["OmegaLambda"] = 0.0
        hg.attrs["HubbleParam"] = 1.0
        hg.attrs["NumPart_ThisFile"] = np.array(
            [
                num_particles.get("gas", 0),
                num_particles.get("dm", 0),
                num_particles.get("tracer", 0),
                0,
                num_particles.get("star", 0),
                num_particles.get("black_hole", 0),
            ],
            dtype="uint32",
        )
        hg.attrs["NumPart_Total"] = hg.attrs["NumPart_ThisFile"]
        hg.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype="uint32")
        hg.attrs["NumFilesPerSnapshot"] = 1
        hg.attrs["MassTable"] = mass_table
        hg.attrs["Flag_Sfr"] = 0
        hg.attrs["Flag_Cooling"] = 0
        hg.attrs["Flag_StellarAge"] = 0
        hg.attrs["Flag_Metals"] = 0
        hg.attrs["Flag_Feedback"] = 0
        hg.attrs["Flag_DoublePrecision"] = 0
        hg.attrs["Flag_IC_Info"] = 0
        if code == "arepo":
            cg = f.create_group("Config")
            cg.attrs["VORONOI"] = 1
        f.flush()
        f.close()

    def to_yt_dataset(self, box_size: float, ptypes: list[str] | str = None):
        """Create an in-memory yt dataset for the particles.

        Parameters
        ----------
        box_size : float
            The width of the domain on a side, in kpc.
        ptypes : string or list of strings, optional
            The particle types to export to the dataset. If
            not set, all will be exported.
        """
        from yt import load_particles

        data = self.fields.copy()
        if ptypes is None:
            ptypes = self.particle_types
        ptypes = ensure_list(ptypes)
        for ptype in ptypes:
            pos = data.pop((ptype, "particle_position"))
            vel = data.pop((ptype, "particle_velocity"))
            for i, ax in enumerate("xyz"):
                data[ptype, f"particle_position_{ax}"] = pos[:, i]
                data[ptype, f"particle_velocity_{ax}"] = vel[:, i]
        return load_particles(
            data,
            length_unit="kpc",
            bbox=[[0.0, box_size]] * 3,
            mass_unit="Msun",
            time_unit="Myr",
        )


def sample_from_clusters(
    particles: ClusterParticles,
    models: list["ClusterModel"],
    center: MaybeUnitVector,
    velocity: MaybeUnitVector,
    radii: Collection[float] | float = None,
    resample: bool = False,
    passive_scalars: Collection[str] = None,
) -> ClusterParticles:
    """Given an input ``particles``, use the provided :py:class:`model.ClusterModel`
    instances (``models``) to redetermine the values of the ``particles`` gas fields
    from the provided models.

    Parameters
    ----------
    particles: ClusterParticles
        The particle dataset to sample from the selected models.
    models: list of ClusterModel
    center: array-like
        The centers of the models.
    velocity: array-like
        The velocities of the models.
    radii: list of float or float, optional
        The cutoff radii for each of the models. Optional. Must be in kpc.
    resample: bool
        If ``True``, then the particle masses (subject to the ``radii`` constraints) are adjusted to accurately reflect the total
        density of particles and the interpolated density obtained from ``models``.

    passive_scalars: list of str, optional
        List of additional passive scalar fields to sample from. These fields must be valid fields of all of the
        :py:class:`model.ClusterModel` instances in ``models``. By default, this argument is ``None`` and no scalar fields
        are propagated.

    Returns
    -------
    ClusterParticles

    Notes
    -----
    Effectively, this function can be used to map the relevant properties of an assortment of ``models`` onto an existing
    set of :py:class:`ClusterParticles`. For each of the relevant gas fields, the constituent ``models`` are interpolated and their
    contributions summed. The values in the analogous particle fields are then set based on those values.

    It should be noted that, by default, this function does not alter ``particle_mass``. As such, it may be the case (for gas particles)
    that the density reflected by the particle fields do not reflect the actual density of particles in physical space. Ideally,
    this would be resolved by re-sampling the actual particles; however, this is not generally possible. As such, the ``resample`` flag
    may be used to make sure that the ``particle_mass`` field of the particles accurately reflects the density and volume per cell of the
    particles.
    """
    # Setting up the mathematics backbone. Get centers, velocities, and particle radii from each model.
    num_halos = len(models)
    center = [ensure_ytarray(c, "kpc") for c in center]
    velocity = [ensure_ytarray(v, "kpc/Myr") for v in velocity]

    # compute the radial displacement from each of the centers for each of the particles.
    r = np.zeros((num_halos, particles.num_particles["gas"]))
    for i, c in enumerate(center):
        r[i, :] = ((particles["gas", "particle_position"] - c) ** 2).sum(axis=1).d
    np.sqrt(r, r)

    # Allow for specific radial cuts to the considered domains if specified.
    if radii is None:
        idxs = slice(None, None, None)
    else:
        radii = np.array(radii)
        idxs = np.any(r <= radii[:, np.newaxis], axis=0)

    # Determine the field arrays
    density_buffer = np.zeros((num_halos, particles.num_particles["gas"]))
    energy_buffer = np.zeros((num_halos, particles.num_particles["gas"]))
    momentum_buffer = np.zeros((num_halos, 3, particles.num_particles["gas"]))

    num_scalars = 0
    if passive_scalars is not None:
        num_scalars = len(passive_scalars)
        scalar_buffer = np.zeros(
            (num_halos, num_scalars, particles.num_particles["gas"])
        )

    for i, model in enumerate(models):
        if "density" not in model:
            mylog.warning(f"No density field found in {model}. Skipping.")
            continue

        # Filling in the particle fields from the interpolated model fields.
        get_density = InterpolatedUnivariateSpline(model["radius"], model["density"])
        density_buffer[i, :] = get_density(r[i, :])
        e_arr = 1.5 * model["pressure"] / model["density"]  # Ideal gas.
        get_energy = InterpolatedUnivariateSpline(model["radius"], e_arr)
        energy_buffer[i, :] = get_energy(r[i, :]) * density_buffer[i, :]
        momentum_buffer[i, :, :] = velocity[i].d[:, np.newaxis] * density_buffer[i, :]

        if num_scalars > 0:
            for j, name in enumerate(passive_scalars):
                get_scalar = InterpolatedUnivariateSpline(model["radius"], model[name])
                scalar_buffer[i, j, :] = get_scalar(r[i, :]) * density_buffer[i, :]

    dens = density_buffer.sum(axis=0)
    eint = (
        energy_buffer.sum(axis=0) / dens
    )  # --> eint and momentum should be density weighted.
    mom = momentum_buffer.sum(axis=0) / dens

    if num_scalars > 0:
        ps = scalar_buffer.sum(axis=0) / dens

    if resample:
        vol = particles["gas", "particle_mass"] / particles["gas", "density"]
        particles["gas", "particle_mass"][idxs] = dens[idxs] * vol.d[idxs]
    particles["gas", "density"][idxs] = dens[idxs]
    particles["gas", "thermal_energy"][idxs] = eint[idxs]
    particles["gas", "particle_velocity"][idxs] = mom.T[idxs]
    if num_scalars > 0:
        for j, name in enumerate(passive_scalars):
            particles["gas", name][idxs] = ps[j, idxs]
    return particles


def combine_clusters(
    particles: list[ClusterParticles],
    models: list["ClusterModel"],
    centers: list[MaybeUnitVector],
    velocities: list[MaybeUnitVector],
) -> ClusterParticles:
    """Combine particle representations of any number of galaxy clusters into a single
    representation.

    Parameters
    ----------
    particles: list of :py:class:`ClusterParticles`
        The :py:class:`ClusterParticles` instances to combine.
    models: list of :py:class:`model.ClusterModel`
        The underlying models to combine.
    centers: unyt_array
        The centers of each of the constituent clusters.
    velocities: unyt_array
        The velocities of each of the constituent clusters.

    Returns
    -------
    ClusterParticles
        The resulting combination of the particles

    Notes
    -----

    As opposed to a naive merger of the fields of the different :py:class:`ClusterParticle` instances, this function combines
    particle datasets and then resamples from their underlying datasets to ensure that the physics is self-consistent.
    """
    centers = [ensure_ytarray(c, "kpc") for c in centers]
    velocities = [ensure_ytarray(v, "km/s") for v in velocities]

    if not (len(centers) == len(velocities) == len(particles) == len(models)):
        raise ValueError(
            f"Particles ({len(particles)}), models ({len(models)}), centers ({len(centers)}), and velocities ({len(velocities)}) are not "
            f"the same."
        )

    # Applying necessary offsets to the particle datasets
    for pid, particle_obj in enumerate(particles):
        _ptypes_copy = (
            particle_obj.particle_types.copy()
        )  # --> necessary so we can remove gas from it.

        if "gas" in _ptypes_copy:
            # apply the gas specific offset (no velocity included).
            particle_obj.add_offsets(centers[pid], [0.0] * 3, ptypes=["gas"])
            _ptypes_copy.remove("gas")

        particle_obj.add_offsets(centers[pid], velocities[pid], ptypes=_ptypes_copy)

    output_particles = np.sum(particles)

    # Resampling if gas was involved.
    if "gas" in output_particles.particle_types:
        output_particles = sample_from_clusters(
            output_particles, models, centers, velocities
        )

    return output_particles


@deprecated(
    version_deprecated="0.1.0",
    alternative=sample_from_clusters,
    reason="Generalized by alternative",
)
def combine_two_clusters(
    particles1, particles2, hse1, hse2, center1, center2, velocity1, velocity2
):
    return combine_clusters(
        [particles1, particles2],
        [hse1, hse2],
        [center1, center2],
        [velocity1, velocity2],
    )


@deprecated(
    version_deprecated="0.1.0",
    alternative=sample_from_clusters,
    reason="Generalized by alternative",
)
def combine_three_clusters(
    particles1,
    particles2,
    particles3,
    hse1,
    hse2,
    hse3,
    center1,
    center2,
    center3,
    velocity1,
    velocity2,
    velocity3,
):
    return combine_clusters(
        [particles1, particles2, particles3],
        [hse1, hse2, hse3],
        [center1, center2, center3],
        [velocity1, velocity2, velocity3],
    )


@deprecated(
    version_deprecated="0.1.0",
    alternative=sample_from_clusters,
    reason="Generalized by alternative",
)
def resample_one_cluster(
    particles, hse, center, velocity, radius=None, passive_scalars=None
):
    return sample_from_clusters(
        particles,
        [hse],
        [center],
        [velocity],
        resample=True,
        radii=[radius],
        passive_scalars=[passive_scalars],
    )


@deprecated(
    version_deprecated="0.1.0",
    alternative=sample_from_clusters,
    reason="Generalized by alternative",
)
def resample_two_clusters(
    particles,
    hse1,
    hse2,
    center1,
    center2,
    velocity1,
    velocity2,
    radii,
    passive_scalars=None,
):
    particles = sample_from_clusters(
        particles,
        [hse1, hse2],
        [center1, center2],
        [velocity1, velocity2],
        radii=radii,
        resample=True,
        passive_scalars=passive_scalars,
    )
    return particles


@deprecated(
    version_deprecated="0.1.0",
    alternative=sample_from_clusters,
    reason="Generalized by alternative",
)
def resample_three_clusters(
    particles,
    hse1,
    hse2,
    hse3,
    center1,
    center2,
    center3,
    velocity1,
    velocity2,
    velocity3,
    radii,
    passive_scalars=None,
):
    particles = sample_from_clusters(
        particles,
        [hse1, hse2, hse3],
        [center1, center2, center3],
        [velocity1, velocity2, velocity3],
        radii=radii,
        resample=True,
        passive_scalars=passive_scalars,
    )
    return particles
