"""
Tests for gravity classes
"""
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from itertools import product

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose
from unyt import unyt_array, unyt_quantity

from cluster_generator.gravity import AQUAL, Newtonian
from cluster_generator.utils import G


class Constructor(ABC):
    """
    Abstract class representation of the construction phase of testing. Each should have a fetch setup function
    and a setup method for each of the relevant testing methodologies.
    """

    @property
    @abstractmethod
    def name(self):
        pass

    @classmethod
    def fetch_setup(cls, answer_dir, answer_store, method, sub_method, **kwargs):
        """Fetches the setups"""
        sm_string = str(sub_method)
        for _, v in kwargs.items():
            sm_string += f"_{v}"
        path = os.path.join(answer_dir, f"{cls.name}-{method}-{sm_string}.h5")

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
            fields, method=method
        )
        cls.check_potential(fields, method, answer_dir, answer_store, **kwargs)

    @classmethod
    def test_compute_dynamical_mass(cls, answer_dir, answer_store, method, **kwargs):
        "Tests the dynamical mass computation"
        fields = cls.constructor_class.fetch_setup(
            answer_dir, answer_store, "compute_dynamical_mass", method, **kwargs
        )
        fields["total_mass"] = cls.gravity_class.compute_dynamical_mass(
            fields, method=method
        )
        cls.check_dynamical_mass(fields, method, answer_dir, answer_store, **kwargs)

    @classmethod
    def test_compute_gravitational_field(
        cls, answer_dir, answer_store, method, **kwargs
    ):
        "Tests the gravitational_field"
        fields = cls.constructor_class.fetch_setup(
            answer_dir, answer_store, "compute_gravitational_field", method, **kwargs
        )
        fields["gravitational_field"] = cls.gravity_class.compute_gravitational_field(
            fields, method=method
        )
        cls.check_gravitational_field(
            fields, method, answer_dir, answer_store, **kwargs
        )

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
            # This is simple integration, doesn't need a test
            fields["gravitational_field"] = unyt_array(np.ones(rr.size), "kpc/Myr**2")
        elif sub_method == 2:
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
        if sub_method == 2:
            M, R_0 = unyt_quantity(1, "Msun"), unyt_quantity(1, "kpc")
            func = lambda x: -(G * M) / (x + R_0)

            check_field = func(fields["radius"])
            assert_allclose(
                fields["gravitational_potential"].d, check_field.d, rtol=1e-4
            )
            assert check_field.units == fields["gravitational_potential"].units
        elif sub_method == 1:
            pass
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


class ConstructorAQUAL(Constructor):
    """Constructor class for the AQUAL tests"""

    name = "AQUAL"

    @classmethod
    def setup_compute_dynamical_mass(cls, sub_method, **kwargs):
        """
        We use a field (beta r^-2) with different values of beta to comprehensively test the dynamical mass
        computations.
        """
        fields = {}
        rr = np.geomspace(1, 10000, 1000)
        fields["radius"] = unyt_array(rr, "kpc")

        is_asymptotic = kwargs["is_asymptotic"]

        if sub_method == 1:
            # constant density profile
            if is_asymptotic:
                rc = 3.724e10  # --> minimum relative acceleration is 1000
            else:
                rc = 3724  # --> minimum relative acceleration is 0.01
            beta = unyt_quantity(rc, "kpc**3/Myr**2")
            field_array = -beta * fields["radius"] ** (-2)

            fields["gravitational_field"] = field_array
        else:
            raise ValueError("No such sub-method")

        return fields

    @classmethod
    def setup_compute_potential(cls, sub_method, **kwargs):
        fields = {}
        rr = np.geomspace(1, 10000, 1000)
        fields["radius"] = unyt_array(rr, "kpc")

        is_asymptotic = kwargs["is_asymptotic"]

        if sub_method == 2:
            # constant density profile
            if is_asymptotic:
                rc = 100000  # --> minimum relative acceleration is 1000
            else:
                rc = 1  # --> minimum relative acceleration is 0.01
            sigma = unyt_quantity(1.2e-10, "m/s**2") * rc / G

            fields["total_mass"] = sigma * fields["radius"] ** 2
            fields["total_mass"].convert_to_units("Msun")
            fields["total_density"] = unyt_array(
                np.gradient(fields["total_mass"].d, fields["radius"].d)
                * (1 / (4 * np.pi * fields["radius"].d ** 2)),
                "Msun/kpc**3",
            )

        elif sub_method == 1:
            fields["gravitational_field"] = unyt_array(np.ones(rr.size), "kpc/Myr**2")
        else:
            raise ValueError("No such sub-method")

        return fields

    @classmethod
    def setup_compute_gravitational_field(cls, sub_method, **kwargs):
        fields = {}
        rr = np.geomspace(1, 10000, 1000)
        fields["radius"] = unyt_array(rr, "kpc")

        is_asymptotic = kwargs["is_asymptotic"]

        if sub_method == 1:
            # constant density profile
            if is_asymptotic:
                rc = 1e17  # --> ranges from interp 487 to 4.87e6
            else:
                rc = 1e6  # --> ranges from interp 0.00487 to 48.7
            rho_constant = unyt_quantity(rc, "Msun/kpc**3")
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


class TestAQUAL(GravityTest):
    constructor_class = ConstructorAQUAL
    gravity_class = AQUAL

    nmethods = {k: len(v) for k, v in gravity_class._method_requirements.items()}

    @classmethod
    @pytest.mark.parametrize(
        "method,is_asymptotic",
        [
            (m, k)
            for m, k in product(
                range(1, nmethods["compute_gravitational_field"] + 1), [True, False]
            )
        ],
    )
    def test_compute_gravitational_field(
        cls, answer_dir, answer_store, method, is_asymptotic, **kwargs
    ):
        super().test_compute_gravitational_field(
            answer_dir, answer_store, method, is_asymptotic=is_asymptotic, **kwargs
        )

    @classmethod
    @pytest.mark.parametrize(
        "method,is_asymptotic",
        [
            (m, k)
            for m, k in product(
                range(1, nmethods["compute_potential"] + 1), [True, False]
            )
        ],
    )
    def test_compute_potential(
        cls, answer_dir, answer_store, method, is_asymptotic, **kwargs
    ):
        super().test_compute_potential(
            answer_dir, answer_store, method, is_asymptotic=is_asymptotic, **kwargs
        )

    @classmethod
    @pytest.mark.parametrize(
        "method,is_asymptotic",
        [
            (m, k)
            for m, k in product(
                range(1, nmethods["compute_dynamical_mass"] + 1), [True, False]
            )
        ],
    )
    def test_compute_dynamical_mass(
        cls, answer_dir, answer_store, method, is_asymptotic, **kwargs
    ):
        super().test_compute_dynamical_mass(
            answer_dir, answer_store, method, is_asymptotic=is_asymptotic, **kwargs
        )

    @classmethod
    def check_potential(cls, fields, sub_method, answer_dir, answer_store, **kwargs):
        from scipy.interpolate import InterpolatedUnivariateSpline

        from cluster_generator.utils import integrate

        is_asymptotic = kwargs["is_asymptotic"]

        if sub_method == 2:
            # constant density profile
            if is_asymptotic:
                rc = 100000  # --> minimum relative acceleration is 1000
            else:
                rc = 1  # --> minimum relative acceleration is 0.01
            dphi = (
                unyt_quantity(1.2e-10, "m/s**2").to("kpc/Myr**2")
                * (0.5 * rc)
                * (1 + np.sqrt(1 + (4 / rc)))
                * np.ones(fields["radius"].size)
            )
            dphi_func = InterpolatedUnivariateSpline(fields["radius"].d, dphi.d)
            check_field = unyt_array(
                -integrate(dphi_func, fields["radius"].d), "kpc**2/Myr**2"
            )

            assert_allclose(
                fields["gravitational_potential"].d, check_field.d, rtol=1e-4
            )
            assert check_field.units == fields["gravitational_potential"].units
        elif sub_method == 1:
            pass
        else:
            raise NotImplementedError

    @classmethod
    def check_dynamical_mass(
        cls, fields, sub_method, answer_dir, answer_store, **kwargs
    ):
        is_asymptotic = kwargs["is_asymptotic"]

        if sub_method == 1:
            # constant density profile
            if is_asymptotic:
                rc = 3.724e10  # --> minimum relative acceleration is 1000
            else:
                rc = 3724  # --> minimum relative acceleration is 0.01
            beta = unyt_quantity(rc, "kpc**3/Myr**2")

        if sub_method == 1 and is_asymptotic:
            test_function = lambda r: beta / G + (0 * r.d)
            check_field = test_function(fields["radius"]).to("Msun")

            assert_allclose(fields["total_mass"].d, check_field.d, rtol=1e-4)
            assert check_field.units == fields["total_mass"].units
        elif sub_method == 1 and not is_asymptotic:
            test_function = lambda r: (
                beta / G
            ) * cls.gravity_class.interpolation_function(
                (beta / cls.gravity_class.get_a0()) * (r ** (-2))
            )
            check_field = test_function(fields["radius"]).to("Msun")

            assert_allclose(fields["total_mass"].d, check_field.d, rtol=1e-7)
            assert check_field.units == fields["total_mass"].units

    @classmethod
    def check_gravitational_field(
        cls, fields, sub_method, answer_dir, answer_store, **kwargs
    ):
        """Confirm that the gravitational field is computed accurately."""
        is_asymptotic = kwargs["is_asymptotic"]

        if sub_method == 1:
            # constant density profile
            if is_asymptotic:
                rc = 1e17  # --> ranges from interp 487e6 to 4.87e12
            else:
                rc = 1e6  # --> ranges from interp 0.00487 to 48.7

        if sub_method == 1 and is_asymptotic:
            test_function = (
                lambda r: (-4 / 3) * G * (np.pi * r) * unyt_quantity(rc, "Msun/kpc**3")
            )
            check_field = test_function(fields["radius"]).to("kpc/Myr**2")

            assert_allclose(fields["gravitational_field"].d, check_field.d, rtol=1e-7)
            assert check_field.units == fields["gravitational_field"].units
        elif sub_method == 1 and not is_asymptotic:
            test_function = (
                lambda r: (4 / (3 * cls.gravity_class.get_a0()))
                * G
                * (np.pi * r)
                * unyt_quantity(rc, "Msun/kpc**3")
            )
            base_field = test_function(fields["radius"])

            field_function = (
                lambda x: -cls.gravity_class.get_a0()
                * (x / 2)
                * (1 + np.sqrt(1 + (4 / x)))
            )

            check_field = field_function(base_field).to("kpc/Myr**2")

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
