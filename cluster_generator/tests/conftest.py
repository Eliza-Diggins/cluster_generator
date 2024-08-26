"""Pytest configuration module for the `cluster_generator` package.

This configuration file sets up pytest for the `cluster_generator` package, providing
custom fixtures, test ordering, and additional command-line options to facilitate
various testing scenarios. It is designed to streamline the testing process for local
development, continuous integration (CI), and remote testing environments.

Overview
--------
The configuration file serves several key purposes:

1. **Custom Test Ordering**:
   Ensures specific tests run first, optimizing the test workflow by reducing redundant
   operations, such as multiple rebuilds of toy models.

2. **Command-Line Options**:
   Adds custom command-line options to pytest, allowing users to control test behavior
   more granularly. Options include managing answer directories, storing new answers,
   packaging test results, and fetching remote data.

3. **Fixtures**:
   Provides a set of custom fixtures that manage test setup and teardown processes,
   such as handling temporary directories, packaging test results, and fetching remote
   answers.

Components
----------
1. **Custom Command-Line Options**:
   - `--answer_dir`: Specifies the directory where test answers are stored. Essential
     for answer comparison and validation.
   - `--answer_store`: If set to True, generates new answers but does not test against
     existing ones. Useful for creating baselines in new environments.
   - `--tmp`: Sets a specific temporary directory to use. If not specified, a new
     temporary directory is generated and cleaned up after the test run.
   - `--package`: If enabled, packages the test results into a `.tar.gz` file for
     external use or distribution. Helpful in CI/CD pipelines or when sharing test
     results.
   - `--fetch_remote`: If set to True, fetches test answers from a remote directory.
     Useful for ensuring consistency across different testing environments.
   - `--remote_file`: Allows specifying an explicit remote file to fetch instead of
     the default answer package.

2. **Fixtures**:
   - `packager`: Packages the test results into a `.tar.gz` file if the `--package`
     option is enabled. This fixture runs after all tests have completed.
   - `fetch_remote`: Fetches remote test answers and loads them into the local answer
     directory if the `--fetch_remote` option is enabled. This fixture runs at the
     beginning of the test session.
   - `answer_store`: Returns the value of the `--answer_store` option, determining
     whether new test answers should be generated and stored.
   - `answer_dir`: Returns the absolute path of the directory where test answers are
     stored. Creates the directory if it does not exist.
   - `temp_dir`: Manages temporary directory creation and cleanup. Returns a directory
     path, which is either user-specified or automatically generated.

3. **Helper Functions**:
   - `download_file(url, output_path)`: Downloads a file from a specified URL to a
     local path. Used primarily in the `fetch_remote` fixture to retrieve remote
     test answers.

4. **Custom Test Collection Modification**:
   - `pytest_collection_modifyitems(session, config, items)`: Modifies the order of
     collected test items to ensure that certain tests (e.g., model-building tests)
     run first. This prevents redundant model builds and optimizes the testing process.

Usage Scenarios
---------------
- **Local Development**:
  Use the default pytest command to run tests locally. The configuration ensures that
  necessary models are built first, and fixtures manage any temporary files or directories.

- **Continuous Integration (CI)**:
  In CI environments, such as GitHub Actions, disabling progress bars and ordering
  tests efficiently can reduce runtime. The `--package` option can be enabled to
  package results for further analysis or storage.

- **Answer Testing**:
  To compare current test outputs against expected results, use the `--answer_dir` to
  specify where expected answers are stored. If new baselines are needed, use the
  `--answer_store` option.

- **Fetching Remote Data**:
  When consistency across environments is critical, the `--fetch_remote` option can
  download pre-built answers from a remote host. This is useful for environments where
  direct access to the test environment is limited or when synchronizing test baselines
  across multiple teams or locations.

- **Packaging Test Results**:
  If the test results need to be shared or transferred, the `--package` option
  automatically packages the results into a `.tar.gz` file. This is particularly
  useful in CI pipelines where artifacts need to be stored or analyzed after a test run.

Setup Instructions
------------------
1. Ensure pytest and any required dependencies are installed in your environment.
2. Use the provided command-line options to customize test behavior according to your needs:
   - Example: `pytest --answer_dir=/path/to/answers --package`
3. Run pytest as usual, and the configuration file will manage the test workflow
   according to the specified options.

This configuration is designed to be flexible and extensible, allowing developers to
easily add new testing protocols, fixtures, or command-line options as needed.
"""
import os

import pytest

from cluster_generator.utilities.config import cgparams

# Disable progress bars during tests to improve compatibility with GitHub Actions.
cgparams.config.system.preferences.disable_progress_bars = True

# REMOTE HOST: This is the server from which to download specific answer keys remotely.
# Currently, hosted via University of Utah Dept. of Astronomy and Physics, but can be changed
# by hosting elsewhere.
_remote_host = "https://astro.utah.edu/~u1281896/cluster_generator"


def download_file(url, output_path):
    """Download a file from a URL to a specified output path.

    Parameters
    ----------
    url : str
        The URL of the file to be downloaded.
    output_path : str
        The local file path where the downloaded file will be saved.

    Raises
    ------
    urllib.error.URLError
        If there is an error during the file download.
    """
    import urllib.request

    urllib.request.urlretrieve(url, output_path)


def pytest_collection_modifyitems(session, config, items):
    """Modify the order of collected test items to ensure certain tests run first.

    This function ensures that tests requiring pre-built models are run after the
    necessary model-building tests, optimizing the test workflow by preventing redundant
    model builds.

    Parameters
    ----------
    session : pytest.Session
        The pytest session object.
    config : pytest.Config
        The pytest configuration object.
    items : list of pytest.Item
        The list of collected test items to be modified.

    Notes
    -----
    In order to enforce test ordering, add the test to the ``_enforced_test_ordering`` list in the correct order.
    This will ensure that it runs before all other tests except those earlier in the list than itself.
    """
    # These are the tests that are run ahead of the other tests. They should be tests which generate
    # data that is needed for testing more complex structures later in the pipeline.
    _enforced_test_ordering = [
        ("cluster_generator.tests.test_core.test_model", "test_model_build")
    ]

    _doc_tests, _standard_tests = (
        [test for test in items if isinstance(test, pytest.DoctestItem)],
        [test for test in items if not isinstance(test, pytest.DoctestItem)],
    )

    _test_dictionary = {
        (test.module.__name__, test.name): test for test in _standard_tests
    }

    items[:] = (
        [
            _test_dictionary.pop(test_name)
            for test_name in _enforced_test_ordering
            if test_name in _test_dictionary
        ]
        + list(_test_dictionary.values())
        + _doc_tests
    )


def pytest_addoption(parser):
    """Add custom command-line options to pytest.

    This function defines custom command-line options that can be used to control the
    behavior of the tests, such as specifying directories for storing or fetching answers
    and enabling the packaging of test results.

    Parameters
    ----------
    parser : pytest.Parser
        The pytest parser object to which options are added.
    """
    parser.addoption("--answer_dir", help="Directory where answers are stored.")
    parser.addoption(
        "--answer_store",
        action="store_true",
        help="Generate new answers, but don't test.",
    )
    parser.addoption("--tmp", help="The temporary directory to use.", default=None)
    parser.addoption(
        "--package",
        help="Package the results of this test for use externally.",
        action="store_true",
    )
    parser.addoption(
        "--fetch_remote",
        help="Fetch answers from remote directory.",
        action="store_true",
    )
    parser.addoption(
        "--remote_file",
        help="Explicitly set the remote file to fetch.",
        default=None,
    )


@pytest.fixture(scope="session", autouse=True)
def packager(request):
    """Fixture to package test results into a .tar.gz file for distribution.

    This fixture runs after all tests have completed and packages the results into a
    compressed file if the `--package` option is specified. This is useful for
    transferring test results for use in environments where direct access to
    the test environment is limited.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The request object for accessing fixture data.
    """
    import sys
    import tarfile
    from importlib.metadata import version
    from pathlib import Path
    from platform import system

    capmanager = request.config.pluginmanager.getplugin("capturemanager")
    # See https://github.com/pytest-dev/pytest/issues/2704

    # Fetch the options
    _answer_dir = Path(os.path.abspath(request.config.getoption("--answer_dir")))
    _answer_store = request.config.getoption("--answer_store")
    _package = request.config.getoption("--package")

    # Obligatory yield to access post-run context in following code.
    yield None

    # Everything here now runs after all tests have finished.
    if not _package:
        return None  # --> if _package is not True, we don't do anything.

    if not _answer_store:
        # we aren't storing answers so nothing else here matters.
        # These odd looking ansi code prints are to maintain the printing structure for pytest.
        print("\0337", end="", flush=True)
        print(
            "\n[cluster_generator tests]: Skipping --package directive because --answer_store was False."
        )
        print("\0338", end="", flush=True)
        return None

    # answer_store was true and package was true -> we package the results.

    # Fetching system info: need OS, python version, and cluster generator version.
    _system = system()  # --> Linux, Windows, etc.
    _py_version = sys.version.split(" ")[0]
    _cg_version = version("cluster_generator")

    package_name = f"cg_answers_{_system}_{_py_version}_{_cg_version}.tar.gz"

    with tarfile.open(package_name, "w:gz") as tar:
        for file in os.listdir(_answer_dir):
            tar.add(os.path.join(_answer_dir, file), arcname=file)

    with capmanager.global_and_fixture_disabled():
        print("\0337", end="", flush=True)
        print(
            f"\n[cluster_generator tests]: Packaged answers for distribution: {package_name}.",
            end="\n",
        )
        print("\0338", end="", flush=True)


@pytest.fixture(scope="session", autouse=True)
def fetch_remote(request):
    """Fixture to fetch remote test answers and load them into the answer directory.

    This fixture runs at the beginning of the test session if the `--fetch_remote`
    option is enabled. It downloads and extracts test answers from a remote server
    into the local answer directory, facilitating tests that rely on external data.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The request object for accessing fixture data.
    """
    import sys
    import tarfile
    from importlib.metadata import version
    from pathlib import Path
    from platform import system

    capmanager = request.config.pluginmanager.getplugin("capturemanager")

    # Fetch the options
    _answer_dir = Path(os.path.abspath(request.config.getoption("--answer_dir")))
    _answer_store = request.config.getoption("--answer_store")
    _fetch_remote = request.config.getoption("--fetch_remote")
    _remote_file = request.config.getoption("--remote_file")

    # Ensure that _answer_store is False; otherwise this is pointless.
    if (not _answer_store) and _fetch_remote:
        # Fetching system info: need OS, python version, and cluster generator version.
        _system = system()  # --> Linux, Windows, etc.
        _py_version = sys.version.split(" ")[0]
        _cg_version = version("cluster_generator")

        if _remote_file is None:
            package_name = f"cg_answers_{_system}_{_py_version}_{_cg_version}.tar.gz"
        else:
            package_name = f"{_remote_file}.tar.gz"

        remote_path = os.path.join(_remote_host, package_name)

        with capmanager.global_and_fixture_disabled():
            print("\0337", end="", flush=True)
            print(
                f"\n[cluster_generator tests]: Attempting to fetch answers from {remote_path}..."
            )
            print("\0338", end="", flush=True)

            # download from the remote path.
            if not os.path.exists(_answer_dir):
                Path(_answer_dir).mkdir(parents=True, exist_ok=True)

            try:
                download_file(remote_path, os.path.join(_answer_dir, package_name))
            except Exception as e:
                raise pytest.UsageError(
                    f"Failed to download answers {remote_path}. Error = {e.__repr__()}"
                )

            print("\0337", end="", flush=True)
            print(
                f"\n[cluster_generator tests]: Obtained answers for distribution: {package_name}."
            )
            print("\0338", end="", flush=True)

            # dump the answers into the answer directory
            with tarfile.open(os.path.join(_answer_dir, package_name), "r:gz") as tar:
                tar.extractall(path=_answer_dir)
            print("\0337", end="", flush=True)
            print(
                f"\n[cluster_generator tests]: Dumped answers into {_answer_dir}. Ready to proceed."
            )
            print("\0338", end="", flush=True)
    yield None


@pytest.fixture()
def answer_store(request) -> bool:
    """Fixture to fetch the `--answer_store` command-line option.

    This fixture returns the value of the `--answer_store` option, which determines
    whether new test answers should be generated and stored.

    Returns
    -------
    bool
        The value of the `--answer_store` option.
    """
    return request.config.getoption("--answer_store")


@pytest.fixture()
def answer_dir(request) -> str:
    """Fixture to fetch the `--answer_dir` command-line option.

    This fixture returns the absolute path of the directory where test answers are
    stored. If the directory does not exist, it is created.

    Returns
    -------
    str
        The absolute path of the answer directory.
    """
    ad = os.path.abspath(request.config.getoption("--answer_dir"))
    if not os.path.exists(ad):
        os.makedirs(ad)
    return ad


@pytest.fixture()
def temp_dir(request) -> str:
    """Fixture to handle temporary directory management.

    This fixture returns the path of a temporary directory to be used during tests.
    If a directory is specified by the user, it will not be wiped after the test run;
    otherwise, a temporary directory is generated and removed after the tests complete.

    Yields
    ------
    str
        The path to the temporary directory.
    """
    td = request.config.getoption("--tmp")

    if td is None:
        from tempfile import TemporaryDirectory

        td = TemporaryDirectory()

        yield td.name

        td.cleanup()
    else:
        yield os.path.abspath(td)
