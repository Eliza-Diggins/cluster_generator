from pathlib import Path
from numpy.testing import assert_allclose, assert_equal

from cluster_generator.cluster_model import ClusterModel
from cluster_generator.hydrostatic import HydrostaticEquilibrium
from cluster_generator.virial import VirialEquilibrium
from cluster_generator.radial_profiles import find_overdensity_radius, \
    snfw_density_profile, snfw_total_mass, vikhlinin_density_profile, \
    rescale_profile_by_mass, find_radius_mass, snfw_mass_profile


def generate_profile():
    z = 0.1
    M200 = 1.5e15
    conc = 4.0
    r200 = find_overdensity_radius(M200, 200.0, z=z)
    a = r200/conc
    M = snfw_total_mass(M200, r200, a)
    rhot = snfw_density_profile(M, a)
    Mt = snfw_mass_profile(M, a)
    r500, M500 = find_radius_mass(Mt, z=z, delta=500.0)
    f_g = 0.12
    rhog = vikhlinin_density_profile(1.0, 100.0, r200, 1.0, 0.67, 3)
    rhog = rescale_profile_by_mass(rhog, f_g*M500, r500)
    rhos = 0.02*rhot
    rmin = 0.1
    rmax = 10000.0
    p = HydrostaticEquilibrium.from_dens_and_tden(rmin, rmax, rhog, rhot,
                                                  stellar_density=rhos)
    p.set_magnetic_field_from_beta(100.0, gaussian=True)
    vd = VirialEquilibrium.from_hse_model(p, ptype="dark_matter")
    vs = VirialEquilibrium.from_hse_model(p, ptype="stellar")

    return p, vd, vs


def answer_testing(model, filename, answer_store, answer_dir):
    p = Path(answer_dir) / filename
    if answer_store:
        model.write_model_to_h5(p, overwrite=True)
    else:
        old_model = ClusterModel.from_h5_file(p)
        for field in old_model.fields:
            assert_equal(old_model[field], model[field])