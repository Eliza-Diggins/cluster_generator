"""Configure pytest for cluster_generator."""
import os

import pytest

from cluster_generator.utilities.config import cgparams

# Disable progress bars during tests -> GH actions cannot emulate console, prints each update on seperate line (slow).
cgparams.config.system.preferences.disable_progress_bars = True

_remote_host = "https://astro.utah.edu/~u1281896/cluster_generator"  # The remote host from which to fetch answers.


def download_file(url, output_path):
    import urllib.error
    import urllib.request

    urllib.request.urlretrieve(url, output_path)


def pytest_collection_modifyitems(session, config, items):
    # Ensuring the order of necessary tests
    # --------------------------------------#
    # This is done to avoid having to rebuild toy-models for every test when one is already generated.
    # As such, we run tests on ``test_model`` first to ensure that a model is generated and saved to disk.
    #
    # _enforced_test_ordering can be used to add order ensured tests which are then forced to run ahead of the other tests.
    _enforced_test_ordering = [
        ("cluster_generator.tests.test_model", "test_model_build")
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
    """Package test results into a .tar.gz file which can then be downloaded for test
    answers."""
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
    """Load the remote answers into the answer directory."""
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
    """Fetches the ``--answer_store`` option."""
    return request.config.getoption("--answer_store")


@pytest.fixture()
def answer_dir(request) -> str:
    """Fetches the ``--answer_dir`` option."""
    ad = os.path.abspath(request.config.getoption("--answer_dir"))
    if not os.path.exists(ad):
        os.makedirs(ad)
    return ad


@pytest.fixture()
def temp_dir(request) -> str:
    """Pull the temporary directory.

    If this is specified by the user, then it may be a non-temp directory which is not
    wiped after runtime. If not specified, then a temp directory is generated and wiped
    after runtime.
    """
    td = request.config.getoption("--tmp")

    if td is None:
        from tempfile import TemporaryDirectory

        td = TemporaryDirectory()

        yield td.name

        td.cleanup()
    else:
        yield os.path.abspath(td)
