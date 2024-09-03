"""
Utility functions for basic functionality of the py:module:`cluster_generator` package.
"""
import logging
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Union

import numpy as np
from more_itertools import always_iterable
from numpy.random import RandomState
from ruamel.yaml import YAML, SafeLoader
from ruamel.yaml.nodes import MappingNode, ScalarNode
from scipy.integrate import quad
from unyt import kpc
from unyt import physical_constants as pc
from unyt import unyt_array, unyt_quantity

# -- configuration directory -- #
_config_directory = os.path.join(Path(__file__).parents[0], "bin", "config.yaml")


# defining the custom yaml loader for unit-ed objects
# Custom YAML loader for unit-ed objects
def _yaml_unit_constructor(loader: SafeLoader, node: MappingNode):
    kw = loader.construct_mapping(node)
    i_s = kw.pop("input_scalar")
    return unyt_array(i_s, **kw)


def _yaml_lambda_loader(loader: SafeLoader, node: ScalarNode):
    return eval(loader.construct_scalar(node))


def _get_loader():
    yaml = YAML(typ="safe")
    yaml.constructor.add_constructor("!unyt", _yaml_unit_constructor)
    yaml.constructor.add_constructor("!lambda", _yaml_lambda_loader)
    return yaml


try:
    yaml_loader = _get_loader()
    with open(_config_directory, "r") as config_file:
        cgparams = yaml_loader.load(config_file)

except FileNotFoundError as er:
    raise FileNotFoundError(
        f"Couldn't find the configuration file! Is it at {_config_directory}? Error = {er}"
    )
except Exception as er:
    raise ValueError(
        f"An error occurred while loading the configuration file! Error = {er}"
    )

stream = (
    sys.stdout
    if cgparams["system"]["logging"]["main"]["stream"] in ["STDOUT", "stdout"]
    else sys.stderr
)
cgLogger = logging.getLogger("cluster_generator")

cg_sh = logging.StreamHandler(stream=stream)

# create formatter and add it to the handlers
formatter = logging.Formatter(cgparams["system"]["logging"]["main"]["format"])
cg_sh.setFormatter(formatter)
# add the handler to the logger
cgLogger.addHandler(cg_sh)
cgLogger.setLevel(cgparams["system"]["logging"]["main"]["level"])
cgLogger.propagate = False

mylog = cgLogger

# -- Setting up the developer debugger -- #
devLogger = logging.getLogger("development_logger")

if cgparams["system"]["logging"]["developer"][
    "enabled"
]:  # --> We do want to use the development logger.
    # -- checking if the user has specified a directory -- #
    if cgparams["system"]["logging"]["developer"]["output_directory"] is not None:
        from datetime import datetime

        dv_fh = logging.FileHandler(
            os.path.join(
                cgparams["system"]["logging"]["developer"]["output_directory"],
                f"{datetime.now().strftime('%m-%d-%y_%H-%M-%S')}.log",
            )
        )

        # adding the formatter
        dv_formatter = logging.Formatter(
            cgparams["system"]["logging"]["main"]["format"]
        )

        dv_fh.setFormatter(dv_formatter)
        devLogger.addHandler(dv_fh)
        devLogger.setLevel("DEBUG")
        devLogger.propagate = False

    else:
        mylog.warning(
            "User enabled development logger but did not specify output directory. Dev logger will not be used."
        )
else:
    devLogger.propagate = False
    devLogger.disabled = True

mp = (pc.mp).to("Msun")
G = (pc.G).to("kpc**3/Msun/Myr**2")
kboltz = (pc.kboltz).to("Msun*kpc**2/Myr**2/K")
kpc_to_cm = (1.0 * kpc).to_value("cm")

X_H = cgparams["physics"]["hydrogen_abundance"]
mu = 1.0 / (2.0 * X_H + 0.75 * (1.0 - X_H))
mue = 1.0 / (X_H + 0.5 * (1.0 - X_H))

# -- Utility functions -- #
_truncator_function = lambda a, r, x: 1 / (1 + (x / r) ** a)


class TimeoutException(Exception):
    def __init__(self, msg="", func=None, max_time=None):
        self.msg = f"{msg} -- {str(func)} -- max_time={max_time} s"


def _daemon_process_runner(*args, **kwargs):
    # Runs the function specified in the kwargs in a daemon process #

    send_end = kwargs.pop("__send_end")
    function = kwargs.pop("__function")

    try:
        result = function(*args, **kwargs)
    except Exception as e:
        send_end.send(e)
        return

    send_end.send(result)


def time_limit(function, max_execution_time, *args, **kwargs):
    """
    Assert a maximal time limit on functions with potentially problematic / unbounded execution times.

    .. warning::

        This function launches a daemon process.

    Parameters
    ----------
    function: callable
        The function to run under the time limit.
    max_execution_time: float
        The maximum runtime in seconds.
    args:
        arguments to pass to the function.
    kwargs: optional
        keyword arguments to pass to the function.

    """
    import time

    from tqdm import tqdm

    recv_end, send_end = multiprocessing.Pipe(False)
    kwargs["__send_end"] = send_end
    kwargs["__function"] = function

    tqdm_kwargs = {}
    for key in ["desc"]:
        if key in kwargs:
            tqdm_kwargs[key] = kwargs.pop(key)

    N = 1000

    p = multiprocessing.Process(target=_daemon_process_runner, args=args, kwargs=kwargs)
    p.start()

    for _ in tqdm(
        range(N),
        **tqdm_kwargs,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining} - {postfix}]",
        colour="green",
        leave=False,
    ):
        time.sleep(max_execution_time / 1000)

        if not p.is_alive():
            p.join()
            result = recv_end.recv()
            break

    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutException(
            "Failed to complete process within time limit.",
            func=function,
            max_time=max_execution_time,
        )
    else:
        p.join()
        result = recv_end.recv()

    if isinstance(result, Exception):
        raise result
    else:
        return result


def truncate_spline(f, r_t, a):
    r"""
    Takes the function ``f`` and returns a truncated equivalent of it, which becomes
    .. math::
    f'(x) = f(r_t) \left(\frac{x}{r_t}\right)**(r_t*df/dx(r_t)/f(r_t))
    This preserves the slope and continuity of the function be yields a monotonic power law at large :math:`r`.
    Parameters
    ----------
    f: InterpolatedUnivariateSpline
        The function to truncate
    r_t: float
        The scale radius
    a: float
        Truncation rate. Higher values cause transition more quickly about :math:`r_t`.
    Returns
    -------
    callable
        The new function.
    """
    _gamma = r_t * f(r_t, 1) / f(r_t)  # This is the slope.
    return lambda x, g=_gamma, _a=a, r=r_t: f(x) * _truncator_function(_a, r, x) + (
        1 - _truncator_function(_a, r, x)
    ) * (f(r) * _truncator_function(-g, r, x))


def integrate_mass(profile, rr):
    mass_int = lambda r: profile(r) * r * r
    mass = np.zeros(rr.shape)
    for i, r in enumerate(rr):
        mass[i] = 4.0 * np.pi * quad(mass_int, 0, r)[0]
    return mass


def integrate(profile, rr):
    ret = np.zeros(rr.shape)
    rmax = rr[-1]
    for i, r in enumerate(rr):
        ret[i] = quad(profile, r, rmax)[0]
    return ret


def integrate_toinf(profile, rr):
    ret = np.zeros(rr.shape)
    rmax = rr[-1]
    for i, r in enumerate(rr):
        ret[i] = quad(profile, r, rmax)[0]
    ret[:] += quad(profile, rmax, np.inf, limit=100)[0]
    return ret


def generate_particle_radii(r, m, num_particles, r_max=None, prng=None):
    prng = parse_prng(prng)
    if r_max is None:
        ridx = r.size
    else:
        ridx = np.searchsorted(r, r_max)
    mtot = m[ridx - 1]
    u = prng.uniform(size=num_particles)
    P_r = np.insert(m[:ridx], 0, 0.0)
    P_r /= P_r[-1]
    r = np.insert(r[:ridx], 0, 0.0)
    radius = np.interp(u, P_r, r, left=0.0, right=1.0)
    return radius, mtot


def ensure_ytquantity(x, default_units):
    if isinstance(x, unyt_quantity):
        return unyt_quantity(x.v, x.units).in_units(default_units)
    elif isinstance(x, tuple):
        return unyt_quantity(x[0], x[1]).in_units(default_units)
    else:
        return unyt_quantity(x, default_units)


def ensure_ytarray(arr, units):
    if not isinstance(arr, unyt_array):
        arr = unyt_array(arr, units)
    return arr.to(units)


def parse_prng(prng):
    if isinstance(prng, RandomState):
        return prng
    else:
        return RandomState(prng)


def ensure_list(x):
    return list(always_iterable(x))


field_label_map = {
    "density": "$\\rho_g$ (g cm$^{-3}$)",
    "temperature": "kT (keV)",
    "pressure": "P (erg cm$^{-3}$)",
    "entropy": "S (keV cm$^{2}$)",
    "dark_matter_density": "$\\rho_{\\rm DM}$ (g cm$^{-3}$)",
    "electron_number_density": "n$_e$ (cm$^{-3}$)",
    "stellar_mass": "M$_*$ (M$_\\odot$)",
    "stellar_density": "$\\rho_*$ (g cm$^{-3}$)",
    "dark_matter_mass": "$M_{\\rm DM}$ (M$_\\odot$)",
    "gas_mass": "M$_g$ (M$_\\odot$)",
    "total_mass": "M$_{\\rm tot}$ (M$_\\odot$)",
    "gas_fraction": "f$_{\\rm gas}$",
    "magnetic_field_strength": "B (G)",
    "gravitational_potential": "$\\Phi$ (kpc$^2$ Myr$^{-2}$)",
    "gravitational_field": "g (kpc Myr$^{-2}$)",
}


class ErrorGroup(Exception):
    """Special error class containing a group of exceptions.

    Attributes
    ----------
    errors : list of Exception
        A list of exceptions that are part of this error group.
    message : str
        A formatted message summarizing all errors in the group.
    """

    def __init__(self, message: str, error_list: list[Exception]):
        """Initialize the ErrorGroup with a message and a list of exceptions.

        Parameters
        ----------
        message : str
            A summary message for the error group.
        error_list : list of Exception
            A list of exceptions to include in the error group.
        """
        super().__init__(message)
        self.errors: list[Exception] = error_list or []
        self.update_message(message)

    def update_message(self, message: str):
        """Update the summary message for the error group.

        Parameters
        ----------
        message : str
            The new summary message for the error group.
        """
        self.message = f"{len(self.errors)} ERRORS: {message}\n\n"
        for err_id, err in enumerate(self.errors):
            self.message += "%(err_id)-5s[%(err_type)10s]: %(msg)s\n" % dict(
                err_id=err_id + 1, err_type=err.__class__.__name__, msg=str(err)
            )

    def append(self, error: Exception):
        """Append a new exception to the error group.

        Parameters
        ----------
        error : Exception
            The exception to add to the group.
        """
        self.errors.append(error)
        self.update_message("Added new error.")

    def extend(self, errors: list[Exception]):
        """Extend the error group with multiple exceptions.

        Parameters
        ----------
        errors : list of Exception
            A list of exceptions to add to the group.
        """
        self.errors.extend(errors)
        self.update_message("Extended with new errors.")

    def clear(self):
        """Clear all errors from the group."""
        self.errors.clear()
        self.update_message("Cleared all errors.")

    def __len__(self):
        """Return the number of errors in the group."""
        return len(self.errors)

    def __str__(self):
        """Return a formatted string representation of the error group."""
        return self.message

    def __repr__(self):
        """Return a detailed string representation of the error group."""
        return f"<ErrorGroup(errors={len(self.errors)}, message={repr(self.message)})>"

    def __iter__(self):
        """Return an iterator over the contained exceptions."""
        return iter(self.errors)

    def __getitem__(self, index: int) -> Exception:
        """Return the exception at the given index.

        Parameters
        ----------
        index : int
            The index of the exception to retrieve.

        Returns
        -------
        Exception
            The exception at the specified index.
        """
        return self.errors[index]

    def __contains__(self, error: Exception) -> bool:
        """Check if a specific exception is in the group.

        Parameters
        ----------
        error : Exception
            The exception to check.

        Returns
        -------
        bool
            True if the exception is in the group, False otherwise.
        """
        return error in self.errors

    def __add__(self, other: Union[Exception, "ErrorGroup"]) -> "ErrorGroup":
        """Add an error or another ErrorGroup to this ErrorGroup.

        Parameters
        ----------
        other : Exception or ErrorGroup
            The exception or ErrorGroup to add.

        Returns
        -------
        ErrorGroup
            A new ErrorGroup containing the combined errors.

        Raises
        ------
        TypeError
            If `other` is not an Exception or ErrorGroup.
        """
        if isinstance(other, Exception):
            new_errors = self.errors + [other]
        elif isinstance(other, ErrorGroup):
            new_errors = self.errors + other.errors
        else:
            raise TypeError(f"Cannot add type {type(other)} to ErrorGroup.")

        return ErrorGroup("Combined errors", new_errors)

    def __iadd__(self, other: Union[Exception, "ErrorGroup"]) -> "ErrorGroup":
        """Add an error or another ErrorGroup to this ErrorGroup in place.

        Parameters
        ----------
        other : Exception or ErrorGroup
            The exception or ErrorGroup to add.

        Returns
        -------
        ErrorGroup
            The modified ErrorGroup with the added errors.

        Raises
        ------
        TypeError
            If `other` is not an Exception or ErrorGroup.
        """
        if isinstance(other, Exception):
            self.append(other)
        elif isinstance(other, ErrorGroup):
            self.extend(other.errors)
        else:
            raise TypeError(f"Cannot add type {type(other)} to ErrorGroup.")

        return self


class LogDescriptor:
    def __get__(self, instance, owner) -> logging.Logger:
        # check owner for an existing logger:
        logger = logging.getLogger(owner.__name__)

        if len(logger.handlers) > 0:
            pass
        else:
            # the logger needs to be created.
            _handler = logging.StreamHandler(
                getattr(sys, cgparams["system"]["logging"]["main"]["stream"].lower())
            )
            _handler.setFormatter(
                logging.Formatter(cgparams["system"]["logging"]["main"]["format"])
            )
            logger.addHandler(_handler)
            logger.setLevel(cgparams["system"]["logging"]["main"]["level"])
            logger.propagate = False
            logger.disabled = not cgparams["system"]["display"]["progress_bars"]

        return logger


def reverse_dict(dictionary: dict) -> dict:
    """Reverse a dictionary."""
    return {v: k for k, v in dictionary.items()}


def prepare_path(path, overwrite=False):
    path = Path(path)

    if path.exists() and (not overwrite):
        raise ValueError(f"Path {path} already exists and overwrite=False.")
    elif path.exists() and overwrite:
        path.unlink()
    else:
        pass

    path.mkdir(exist_ok=True, parents=True)

    return path


class _MissingParameter:
    """A descriptor that raises an error if accessed, set, or deleted."""

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        raise AttributeError(f"{owner.__name__} does not implement {self.name}.")

    def __set__(self, instance, value):
        raise AttributeError(
            f"{type(instance).__name__} does not implement {self.name}."
        )

    def __delete__(self, instance):
        raise AttributeError(
            f"{type(instance).__name__} does not implement {self.name}."
        )

    def __repr__(self):
        return "<MissingParameter>"
