"""
Tests for gravity classes
"""
import os
from abc import ABC, abstractmethod
from collections import OrderedDict

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose
from unyt import unyt_array, unyt_quantity

from cluster_generator.gravity import Newtonian
from cluster_generator.utils import G


class Constructor(ABC):
    """
    Contains all of the data generation routines for each gravitational theory.
    """

    @property
    @abstractmethod
    def name(self):
        pass

    @classmethod
    def fetch_setup(cls, answer_dir, answer_store, method, sub_method, **kwargs):
        """Fetches the setups"""
        path = os.path.join(answer_dir, f"{cls.name}-{method}-{sub_method}.h5")

        if answer_store or (not os.path.exists(path)):
            fields = getattr(cls, f"setup_{method}")(sub_method, **kwargs)

            f = h5py.File(path, "w")
            f.close()

            r_min = 0.0
            r_max = fields["radius"][-1].d * 2
            mask = np.logical_and(
                fields["radius"].d >= r_min, fields["radius"].d <= r_max
            )

            for k, v in fields.items():
                fd = v[mask]
                fd.write_hdf5(path, dataset_name=k, group_name="fields")
        else:
            with h5py.File(path, "r") as f:
                fnames = list(f["fields"].keys())

            fields = OrderedDict()
            for field in fnames:
                a = unyt_array.from_hdf5(path, dataset_name=field, group_name="fields")
                fields[field] = unyt_array(a.d, str(a.units))

        return fields

    @abstractmethod
    def setup_compute_dynamical_mass(self, sub_method, **kwargs):
        pass

    @abstractmethod
    def setup_compute_potential(self, sub_method, **kwargs):
        pass

    @abstractmethod
    def setup_compute_gravitational_field(self, sub_method, **kwargs):
        pass


@pytest.mark.usefixtures("answer_store", "answer_dir")
class GravityTest(ABC):
    """
    Core testing structure for gravitational testing.
    """

    @property
    @abstractmethod
    def gravity_class(self):
        """The gravity class associated with these tests."""
        return object

    @property
    @abstractmethod
    def constructor_class(self):
        """The constructor class associated with these tests."""
        return object

    nmethods = {
        "compute_potential": 1,
        "compute_dynamical_mass": 1,
        "compute_gravitational_field": 1,
    }

    @classmethod
    def test_compute_potential(cls, answer_dir, answer_store, method, **kwargs):
        """Tests the potential computation"""
        fields = cls.constructor_class.fetch_setup(
            answer_dir, answer_store, "compute_potential", method, **kwargs
        )
        fields["gravitational_potential"] = cls.gravity_class.compute_potential(
            fields, method=method, **kwargs
        )
        cls.check_potential(fields, method, answer_dir, answer_store)

    @classmethod
    def test_compute_dynamical_mass(cls, answer_dir, answer_store, method, **kwargs):
        "Tests the dynamical mass computation"
        fields = cls.constructor_class.fetch_setup(
            answer_dir, answer_store, "compute_dynamical_mass", method, **kwargs
        )
        fields["total_mass"] = cls.gravity_class.compute_dynamical_mass(
            fields, method=method, **kwargs
        )
        cls.check_dynamical_mass(fields, method, answer_dir, answer_store)

    @classmethod
    def test_compute_gravitational_field(
        cls, answer_dir, answer_store, method, **kwargs
    ):
        "Tests the gravitational_field"
        fields = cls.constructor_class.fetch_setup(
            answer_dir, answer_store, "compute_gravitational_field", method, **kwargs
        )
        fields["gravitational_field"] = cls.gravity_class.compute_gravitational_field(
            fields, method=method, **kwargs
        )
        cls.check_gravitational_field(fields, method, answer_dir, answer_store)

    @abstractmethod
    def check_potential(self, *args, **kwargs):
        pass

    @abstractmethod
    def check_dynamical_mass(self, *args, **kwargs):
        pass

    @abstractmethod
    def check_gravitational_field(self, *args, **kwargs):
        pass


class ConstructorNewtonian(Constructor):
    """Constructor class for the Newtonian tests"""

    name = "Newtonian"

    @classmethod
    def setup_compute_dynamical_mass(cls, sub_method, **kwargs):
        fields = {}
        rr = np.geomspace(1, 10000, 1000)
        fields["radius"] = unyt_array(rr, "kpc")

        if sub_method == 1:
            # constant density profile
            grav_field_funct = lambda x: (-1 / x**2) * unyt_quantity(
                1, "kpc**3/(Myr**2)"
            )

            fields["gravitational_field"] = grav_field_funct(fields["radius"])

        else:
            raise ValueError("No such sub-method")

        return fields

    @classmethod
    def setup_compute_potential(cls, sub_method, **kwargs):
        fields = {}
        rr = np.geomspace(1, 10000, 1000)
        fields["radius"] = unyt_array(rr, "kpc")

        if sub_method == 1:
            # constant density profile
            from cluster_generator.radial_profiles import (
                hernquist_density_profile,
                hernquist_mass_profile,
            )

            m_func = hernquist_mass_profile(1, 1)
            d_func = hernquist_density_profile(1, 1)

            fields["total_mass"] = unyt_array(m_func(rr), "Msun")
            fields["total_density"] = unyt_array(d_func(rr), "Msun/kpc**3")

        else:
            raise ValueError("No such sub-method")

        return fields

    @classmethod
    def setup_compute_gravitational_field(cls, sub_method, **kwargs):
        fields = {}
        rr = np.geomspace(1, 10000, 1000)
        fields["radius"] = unyt_array(rr, "kpc")

        if sub_method == 1:
            # constant density profile
            rho_constant = unyt_quantity(1, "Msun/kpc**3")
            mass_array = (4 / 3) * np.pi * (fields["radius"] ** 3) * rho_constant

            fields["total_mass"] = mass_array

        elif sub_method == 2:
            a, b = 1, 1
            potential_function = lambda x, alpha=a, beta=b: (-alpha / x) * (
                np.log(1 + (x / beta))
            )

            fields["gravitational_potential"] = unyt_array(
                potential_function(rr), "kpc**2/Myr**2"
            )
        else:
            raise ValueError("No such sub-method")

        return fields


class TestNewtonian(GravityTest):
    constructor_class = ConstructorNewtonian
    gravity_class = Newtonian

    nmethods = {k: len(v) for k, v in gravity_class._method_requirements.items()}

    @classmethod
    @pytest.mark.parametrize(
        "method", range(1, nmethods["compute_gravitational_field"] + 1)
    )
    def test_compute_gravitational_field(
        cls, answer_dir, answer_store, method, **kwargs
    ):
        super().test_compute_gravitational_field(
            answer_dir, answer_store, method, **kwargs
        )

    @classmethod
    @pytest.mark.parametrize("method", range(1, nmethods["compute_potential"] + 1))
    def test_compute_potential(cls, answer_dir, answer_store, method, **kwargs):
        super().test_compute_potential(answer_dir, answer_store, method, **kwargs)

    @classmethod
    @pytest.mark.parametrize("method", range(1, nmethods["compute_dynamical_mass"] + 1))
    def test_compute_dynamical_mass(cls, answer_dir, answer_store, method, **kwargs):
        super().test_compute_dynamical_mass(answer_dir, answer_store, method, **kwargs)

    @classmethod
    def check_potential(cls, fields, sub_method, answer_dir, answer_store):
        if sub_method == 1:
            M, R_0 = unyt_quantity(1, "Msun"), unyt_quantity(1, "kpc")
            func = lambda x: -(G * M) / (x + R_0)

            check_field = func(fields["radius"])
            assert_allclose(
                fields["gravitational_potential"].d, check_field.d, rtol=1e-4
            )
            assert check_field.units == fields["gravitational_potential"].units
        else:
            raise NotImplementedError

    @classmethod
    def check_dynamical_mass(cls, fields, sub_method, answer_dir, answer_store):
        if sub_method == 1:
            test_function = lambda x: np.ones(x.size) * (
                unyt_quantity(1, "kpc**3/(Myr**2)") / G
            ).to("Msun")

            check_field = unyt_array(test_function(fields["radius"].d), "Msun")

            assert_allclose(fields["total_mass"].to("Msun").d, check_field.d, rtol=1e-7)
            assert check_field.units == fields["total_mass"].units
        else:
            raise NotImplementedError

    @classmethod
    def check_gravitational_field(cls, fields, sub_method, answer_dir, answer_store):
        if sub_method == 1:
            test_function = (
                lambda r: (-4 / 3) * G * (np.pi * r) * unyt_quantity(1, "Msun/kpc**3")
            )
            check_field = test_function(fields["radius"]).to("kpc/Myr**2")

            assert_allclose(fields["gravitational_field"].d, check_field.d, rtol=1e-7)
            assert check_field.units == fields["gravitational_field"].units
        elif sub_method == 2:
            test_function = lambda x: (np.log(1 + x) / x**2) - (1 / (x * (1 + x)))

            check_field = -unyt_array(test_function(fields["radius"].d), "kpc/Myr**2")

            assert_allclose(fields["gravitational_field"].d, check_field.d, rtol=1e-5)
            assert check_field.units == fields["gravitational_field"].units
        else:
            raise NotImplementedError


if __name__ == "__main__":
    pass
