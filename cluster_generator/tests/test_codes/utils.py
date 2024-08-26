"""Testing utilities for simulation code frontends.

This module provides a base class `Test_Code` for testing different frontend codes
that extend the `SimulationCode` class. It sets up a generic testing framework
to validate the correct behavior of simulation code implementations by comparing
generated outputs against stored answer files.

Overview
--------
The primary purpose of this module is to facilitate the testing of various frontend
simulation codes that extend the `SimulationCode` abstract base class. It provides
a structured approach to testing by ensuring that initial conditions (ICs) are
consistently generated, simulation codes are correctly instantiated, and outputs
are accurately compared against expected results.

Key Components
--------------
1. **Base Test Class (`Test_Code`)**:
   - An abstract base class for testing different simulation code frontends.
   - Subclasses should define specific simulation code classes (`CODE_CLASS`)
     and implement the `compare_outputs` method to handle the output comparison logic.

2. **Class Variables**:
   - `ANSWER_SUBDIR`: Optional subdirectory within `answer_dir` where specific
     answer files are stored. This allows organizing answers by code type or test case.
   - `CODE_CLASS`: The `SimulationCode` subclass being tested. Subclasses must
     override this with the appropriate code class.
   - `CODE`: Instance of the simulation code being tested, instantiated within
     the `setup` method.
   - `ICS`: Common initial conditions object used across all tests, created in
     the `setup` method.
   - `INIT_INFO`: Dictionary of initialization parameters passed when creating
     the code instance.

3. **Methods to Implement in Subclasses**:
   - `setup_INIT_INFO(answer_dir, answer_store, temp_dir)`: Set up additional
     initialization parameters required by the specific code class.
   - `compare_outputs(generated_file, answer_file)`: Compare a generated file
     with the corresponding stored answer file. Subclasses must implement the
     logic for comparing files (e.g., binary diff, numerical tolerance checks).

4. **Testing Workflow**:
   - The `setup` method is automatically invoked before each test, setting up
     initial conditions and the code instance.
   - The `test_generate_ics` method tests the generation of initial condition
     files using the specific code instance and checks them against stored answers.
   - The `test_rtp` method tests the generation of runtime parameter files
     (`RTPs`) and checks them against stored answers.
   - Additional tests can be implemented in subclasses as needed.

5. **Utility Methods**:
   - `check_single_answer(answer_dir, answer_store, temp_dir, filename)`: Checks
     a single generated file against the stored answer file. If the answer file
     does not exist, it stores the generated output as the answer.
   - `check_answers(answer_dir, answer_store, temp_dir)`: Checks all generated
     files in a directory against stored answer files, providing a comprehensive
     check for a series of outputs.

How to Implement Additional Tests
---------------------------------
1. **Subclass the `Test_Code` Base Class**:
   - Create a new class that inherits from `Test_Code`.
   - Override the `CODE_CLASS` class variable with the specific frontend simulation
     code class you are testing.

2. **Define Required Methods**:
   - Implement `setup_INIT_INFO` to provide any additional initialization details
     specific to your code class.
   - Implement `compare_outputs` to define how the outputs should be compared.
     This could involve byte-wise comparisons for binary files or more nuanced
     checks (e.g., within a numerical tolerance) for scientific data.

3. **Add New Test Methods**:
   - Add new test methods as needed to cover specific functionalities or scenarios.
   - Use pytest fixtures such as `answer_dir`, `answer_store`, and `temp_dir`
     to manage test directories and files.

4. **Run the Tests**:
   - Use pytest to run your tests. The configuration provided by the base class
     and its fixtures will handle the setup, teardown, and validation steps
     automatically.

Example
-------
```python
class Test_MyCode(Test_Code):
    ANSWER_SUBDIR = "mycode_answers"
    CODE_CLASS = MySimulationCode

    def setup_INIT_INFO(self, answer_dir, answer_store, temp_dir):
        self.INIT_INFO = {
            "param1": "value1",
            "param2": "value2",
            "output_dir": temp_dir,
        }

    def compare_outputs(self, generated_file: Path, answer_file: Path):
        with open(generated_file, 'rb') as gen, open(answer_file, 'rb') as ans:
            assert gen.read() == ans.read(), f"Files {generated_file} and {answer_file} do not match."
"""
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import ClassVar, Type

import pytest

from cluster_generator.codes.abc import SimulationCode
from cluster_generator.ics import ClusterICs
from cluster_generator.tests.utils import get_base_model_path


@pytest.mark.usefixtures("answer_dir", "answer_store", "temp_dir")
class Test_Code(ABC):
    """Base test class for testing different frontend codes.

    Provides a generic structure for testing different frontend codes. Subclasses should override the
    `CODE_CLASS` class variable with the specific frontend code class and implement the `compare_outputs` method.
    """

    ANSWER_SUBDIR: ClassVar[str | None] = None
    """Subdirectory within `answer_dir` where answer files are stored.

    Generally may be the name of the code. For example, ``"AREPO"`` will store all of the outputs under ``/tmp/<temp_file>/AREPO``.
    """

    CODE_CLASS: ClassVar[Type[SimulationCode]] = SimulationCode
    """The `SimulationCode` class to be tested.

    This should be the particular code class that we're testing.
    """

    CODE: ClassVar[SimulationCode] = None
    """Instance of the simulation code being tested.

    This should **always** be ``None``. It is populated during setup.
    """

    ICS: ClassVar[ClusterICs] = None
    """Initial conditions used across all tests.

    This should **always** be ``None``. It is populated during setup.
    """

    INIT_INFO: ClassVar[dict] = {}
    """Initialization info to be passed when creating the code instance.

    This serves to fill in the user-arguments and CTP of the code when it is
    initialized. Because we are only testing file formats, these may be generic or
    nonsensical.
    """
    VALIDATORS: ClassVar[defaultdict] = defaultdict(lambda: lambda path: True, {})
    """Validators for each of the output files.

    For each output file name (``IC``, ``RTP``, etc), add a key
    to the dictionary and a value which takes only the path to the file and raises assertion errors if validation fails.
    """

    @abstractmethod
    def setup_INIT_INFO(self, answer_dir, answer_store, temp_dir):
        """Setup additional initialization info if needed.

        Generically, this should be used when setting fields of the code class that are not settable without
        the context of the fixtures. All alterations should occur directly in ``self.__class__.INIT_INFO`` as that
        is then immediately fed into ``self.__class__.CODE_CLASS`` to create ``self.__class__.CODE``.
        """
        pass

    @abstractmethod
    def compare_outputs(self, generated_file: Path, answer_file: Path):
        """Compare a generated file with the corresponding answer file.

        Parameters
        ----------
        generated_file : Path
            Path to the generated file.
        answer_file : Path
            Path to the answer file.

        Raises
        ------
        AssertionError
            If the generated file does not match the answer file.

        Notes
        -----
        It is generally good practice to let this method simply redirect based on the path to purpose-built
        checkers for each of the different generated file types.
        """
        pass

    @pytest.fixture(autouse=True)
    def setup(self, answer_dir, answer_store, temp_dir: str):
        """Setup the initial conditions and the simulation code instance."""
        generated_path = Path(temp_dir) / (self.__class__.ANSWER_SUBDIR or "")
        generated_path.mkdir(exist_ok=True)

        base_model_path = get_base_model_path(temp_dir)

        # Configure and generate the ICs
        num_particles = {k: 10000 for k in ["dm", "star", "gas"]}
        self.__class__.ICS = ClusterICs(
            "single",
            1,
            base_model_path,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            num_particles=num_particles,
        )

        self.setup_INIT_INFO(answer_dir, answer_store, temp_dir)
        self.__class__.CODE = self.__class__.CODE_CLASS(**self.__class__.INIT_INFO)

    def check_single_answer(
        self, answer_dir: str, answer_store: bool, temp_dir: str, filename: str
    ):
        """Check a single generated IC file against the stored answer file.

        Parameters
        ----------
        answer_dir : str
            The directory where answer files are stored.
        answer_store : bool
            Whether to store the newly generated output as the answer file.
        temp_dir : str
            The directory where temporary files are stored during tests.
        filename : str
            The name of the generated file to check against the answer file.
        """
        from shutil import copy

        # Paths for answers and generated file
        answer_path = Path(answer_dir) / (self.__class__.ANSWER_SUBDIR or "") / filename
        generated_path = (
            Path(temp_dir) / (self.__class__.ANSWER_SUBDIR or "") / filename
        )

        assert generated_path.exists(), (
            f"Failed to check answer for {self.CODE} because output IC file "
            f"was not at {generated_path} as expected."
        )

        # Determine if we need to store answers
        if not answer_path.exists():
            # Answer file does not exist; create directory if needed and store the generated file as the answer
            _answer_store = True
            answer_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            _answer_store = answer_store

        if _answer_store or not answer_path.exists():
            # Store the generated file as the answer
            copy(generated_path, answer_path)
        else:
            # Compare the generated file with the existing answer file
            self.compare_outputs(generated_path, answer_path)

    def check_answers(self, answer_dir: str, answer_store: bool, temp_dir: str):
        from shutil import copy

        # Paths for answers and generated files
        # We need to map the generated files in the temp_dir to the answers in the
        # answer_dir. Then we can proceed.
        answer_path = Path(answer_dir) / (self.__class__.ANSWER_SUBDIR or "")
        generated_path = Path(temp_dir) / (self.__class__.ANSWER_SUBDIR or "")

        assert generated_path.exists(), (
            f"Failed to check answers for {self.CODE} because output IC directory "
            f"was not at {generated_path} as expected."
        )

        if not answer_path.exists():
            # We have no answers, so we just generate them and then proceed.
            _answer_store = True
            answer_path.mkdir(parents=True, exist_ok=True)
        else:
            _answer_store = answer_store

        # Identify all of the answer files we have and the generated files that exist.
        # We need to check each one against the others iteratively.
        generated_files = sorted(generated_path.glob("*"))
        answer_files = sorted(answer_path.glob("*"))

        for gen_file in generated_files:
            # the generated file should match it's partner in the answer dir
            ans_file = answer_path / gen_file.name

            if _answer_store or (not ans_file.exists()):
                # We don't have the answer, so we just write it.
                copy(gen_file, ans_file)
            else:
                # We should now check these two files.
                self.compare_outputs(gen_file, ans_file)

        # If we have any answers that aren't in the generated files, that might be an indication
        # of failure in the testing suite.
        for ans_file in answer_files:
            gen_file = generated_path / ans_file.name
            assert (
                gen_file.exists()
            ), f"Failed to find generated {gen_file} matching {ans_file} in answers."

    def test_generate_ics(self, answer_dir: str, answer_store: bool, temp_dir: str):
        """Test generation of initial condition files for the specific frontend code."""
        ic_path = Path(temp_dir) / (self.ANSWER_SUBDIR or "") / "IC"
        self.CODE.determine_runtime_params(self.ICS)
        self.CODE.generate_ics(self.ICS, overwrite=True)
        self.VALIDATORS["IC"](ic_path)
        self.check_single_answer(answer_dir, answer_store, temp_dir, ic_path.name)

    def test_rtp(self, answer_dir, answer_store, temp_dir):
        # For the specified code instance, we generate rtps and then check them.
        rtp_path = Path(temp_dir) / (self.__class__.ANSWER_SUBDIR or "") / "RTP"
        self.CODE.determine_runtime_params(self.ICS)
        self.CODE.generate_rtp_template(rtp_path, overwrite=True)
        self.VALIDATORS["RTP"](rtp_path)
        self.check_single_answer(answer_dir, answer_store, temp_dir, rtp_path.name)
