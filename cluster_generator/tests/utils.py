from pathlib import Path

import h5py
from numpy.testing import assert_allclose, assert_equal

from cluster_generator.model import ClusterModel
from cluster_generator.particles import ClusterParticles
from cluster_generator.radial_profiles import (
    find_overdensity_radius,
    find_radius_mass,
    rescale_profile_by_mass,
    snfw_density_profile,
    snfw_mass_profile,
    snfw_total_mass,
    vikhlinin_density_profile,
)


def generate_model():
    z = 0.1
    M200 = 1.5e15
    conc = 4.0
    r200 = find_overdensity_radius(M200, 200.0, z=z)
    a = r200 / conc
    M = snfw_total_mass(M200, r200, a)
    rhot = snfw_density_profile(M, a)
    Mt = snfw_mass_profile(M, a)
    r500, M500 = find_radius_mass(Mt, z=z, delta=500.0)
    f_g = 0.12
    rhog = vikhlinin_density_profile(1.0, 100.0, r200, 1.0, 0.67, 3)
    rhog = rescale_profile_by_mass(rhog, f_g * M500, r500)
    rhos = 0.02 * rhot
    rmin = 0.1
    rmax = 10000.0
    m = ClusterModel.from_dens_and_tden(rmin, rmax, rhog, rhot, stellar_density=rhos)
    m.set_magnetic_field_from_beta(100.0, gaussian=True)

    return m


def model_answer_testing(model, filename, answer_store, answer_dir):
    p = Path(answer_dir) / filename
    if answer_store:
        model.write_model_to_h5(p, overwrite=True)
    else:
        old_model = ClusterModel.from_h5_file(p)
        for field in old_model.fields:
            assert_equal(old_model[field], model[field])
        assert_equal(old_model.dm_virial.df, model.dm_virial.df)
        assert_equal(old_model.star_virial.df, model.star_virial.df)


def particle_answer_testing(parts, filename, answer_store, answer_dir):
    p = Path(answer_dir) / filename
    if answer_store:
        parts.write_particles(p, overwrite=True)
    else:
        old_parts = ClusterParticles.from_file(p)
        for field in old_parts.fields:
            assert_equal(old_parts[field], parts[field])


def _h5_recursive_check(
    primary_object: h5py.File | h5py.Group | h5py.Dataset,
    secondary_object: h5py.File | h5py.Group | h5py.Dataset,
):
    # Check that they are the same type
    assert type(primary_object) is type(
        secondary_object
    ), f"{type(primary_object)} != {type(secondary_object)}"

    # -- Attribute checking -- #
    assert all(
        k in primary_object.attrs.keys() for k in secondary_object.attrs.keys()
    ), f"Attribute keys of {primary_object} and {secondary_object} don't match."

    for k in primary_object.attrs.keys():
        assert (
            primary_object.attrs[k] == secondary_object.attrs[k]
        ), f"Attribute {k} of {primary_object} and {secondary_object} don't match."

    # -- Dataset -- #
    # If the object is a dataset, then we need to check the array equality and exit without passing through the
    # rest of this function.
    if isinstance(primary_object, h5py.Dataset):
        assert_allclose(primary_object[...], secondary_object[...], rtol=1e-7)
        return

    # -- Groups or Files -- #
    # In each case, we can simply check keys are equal again and then recursively check their sub-objects.
    assert all(
        k in primary_object.keys() for k in secondary_object.keys()
    ), f"Keys of {primary_object} and {secondary_object} don't match."

    for key in primary_object.keys():
        _h5_recursive_check(primary_object[key], secondary_object[key])


def h5_answer_testing(
    new_path: str, filename: str, answer_store: bool, answer_dir: str
):
    """Recursively test an HDF5 file against another.

    Parameters
    ----------
    new_path: str
        The newly generated model to test against.
    filename: str
        The name of the old model (as was saved in the answer directory).
    answer_store: bool
        If ``True``, the new model will replace the old and no checks are run.
    answer_dir: str
        The directory where the answers are stored.
    """
    import os

    p = Path(answer_dir) / filename
    if answer_store or not os.path.exists(p):
        from shutil import copy

        try:
            copy(new_path, p)  # --> Overwrite the existing file with the new path.
        except Exception:
            pass
    else:
        with h5py.File(new_path, "r") as nfio, h5py.File(p, "r") as ofio:
            # recursively check these against one another.
            _h5_recursive_check(nfio, ofio)
