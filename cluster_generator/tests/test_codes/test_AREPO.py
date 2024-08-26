"""Testing suite for the AREPO simulation frontend.

This module provides test cases and utilities for validating the AREPO simulation code,
including checks for initial condition files (ICs) and runtime parameter files (RTPs).
"""

from pathlib import Path
from typing import ClassVar

import h5py
import pytest
import unyt

from cluster_generator.codes.arepo.arepo import Arepo
from cluster_generator.codes.arepo.io import Arepo_HDF5
from cluster_generator.tests.test_codes.utils import Test_Code


def validate_arepo_ic_file(path: str):
    """Validate an AREPO initial condition (IC) file to ensure it follows AREPO
    conventions.

    Parameters
    ----------
    path : str
        The path to the AREPO IC file to validate.

    Raises
    ------
    AssertionError
        If the file does not conform to expected AREPO conventions.
    """
    file = Arepo_HDF5(path)

    # Enforcing particle-specific details.
    particles = file.particles
    assert (
        file._group_prefix == "PartType"
    ), f"Wrong part type detected: {file._group_prefix}."
    assert {"gas", "star", "dm"} == set(
        particles.particle_types
    ), f"Inconsistent particle types: {particles.particle_types}"

    # Check that gas has the correct/expected fields
    expected_gas_fields = {
        ("gas", k)
        for k in [
            "particle_position",
            "particle_velocity",
            "particle_mass",
            "thermal_energy",
            "particle_index",
        ]
    }
    actual_gas_fields = set([j for j in particles.fields.keys() if j[0] == "gas"])
    assert (
        expected_gas_fields == actual_gas_fields
    ), f"Gas has wrong fields: {actual_gas_fields}"

    # Check that dm has the correct/expected fields
    expected_dm_fields = {
        ("dm", k)
        for k in [
            "particle_position",
            "particle_velocity",
            "particle_mass",
            "particle_index",
        ]
    }
    actual_dm_fields = set([j for j in particles.fields.keys() if j[0] == "dm"])
    assert (
        expected_dm_fields == actual_dm_fields
    ), f"DM has wrong fields: {actual_dm_fields}"

    # Check that star has the correct/expected fields
    expected_star_fields = {
        ("star", k)
        for k in [
            "particle_position",
            "particle_velocity",
            "particle_mass",
            "particle_index",
        ]
    }
    actual_star_fields = set([j for j in particles.fields.keys() if j[0] == "star"])
    assert (
        expected_star_fields == actual_star_fields
    ), f"Star has wrong fields: {actual_star_fields}"

    # Checking the HDF5 file's structure
    with h5py.File(path) as fio:
        expected_keys = {"Header", "PartType0", "PartType1", "PartType4"}
        actual_keys = set(fio.keys())
        assert (
            expected_keys == actual_keys
        ), f"Keys in {path} don't match expectation: {actual_keys}."


@pytest.mark.usefixtures("answer_dir", "answer_store", "temp_dir")
class TestAREPO(Test_Code):
    """Test suite for the AREPO frontend code.

    This class tests the AREPO frontend by generating initial condition files (ICs) and
    runtime parameter files (RTPs) and comparing them against expected answers.
    """

    ANSWER_SUBDIR: ClassVar[str | None] = "AREPO"
    """Subdirectory within the `answer_dir` where AREPO answer files are stored."""

    CODE_CLASS = Arepo
    INIT_INFO = {
        "OUTPUT_STYLE": (
            unyt.unyt_quantity(0.0, "Myr"),
            unyt.unyt_quantity(1.0, "Myr"),
        ),
        "TIME_MAX": unyt.unyt_quantity(10, "Gyr"),
        "SOFTENING_COMOVING": {k: unyt.unyt_quantity(10, "pc") for k in range(6)},
    }

    # Validators are used to check the correctness of the generated files
    VALIDATORS = Test_Code.VALIDATORS.copy()
    VALIDATORS["IC"] = validate_arepo_ic_file

    def setup_INIT_INFO(self, answer_dir, answer_store, temp_dir):
        """Set up additional initialization information specific to AREPO.

        Parameters
        ----------
        answer_dir : str
            The directory where answer files are stored.
        answer_store : bool
            Whether to store newly generated outputs as answer files.
        temp_dir : str
            The directory for storing temporary files during tests.
        """
        self.__class__.INIT_INFO["IC_PATH"] = (
            Path(temp_dir) / (self.__class__.ANSWER_SUBDIR or "") / "IC"
        )

    def compare_outputs(self, generated_file: Path, answer_file: Path):
        """Compare generated files against stored answer files.

        Parameters
        ----------
        generated_file : Path
            The path to the generated file to be validated.
        answer_file : Path
            The path to the answer file to compare against.

        Raises
        ------
        ValueError
            If the output type of the generated file is unrecognized.
        """
        if generated_file.name == "RTP":
            self._check_rtp(generated_file, answer_file)
        elif generated_file.name == "IC":
            self._check_ic(generated_file, answer_file)
        else:
            raise ValueError(f"Unrecognized output type: {generated_file.name}")

    def _check_rtp(self, gen_file: Path, ans_file: Path):
        """Validate the runtime parameter (RTP) files.

        This method should implement the logic to compare RTP files.

        Parameters
        ----------
        gen_file : Path
            The path to the generated RTP file.
        ans_file : Path
            The path to the answer RTP file.
        """
        # Implement RTP validation logic here
        pass

    def _check_ic(self, gen_file: Path, ans_file: Path):
        """Validate the initial condition (IC) files.

        This method should implement the logic to compare IC files.

        Parameters
        ----------
        gen_file : Path
            The path to the generated IC file.
        ans_file : Path
            The path to the answer IC file.
        """
        # Implement IC validation logic here
        pass
