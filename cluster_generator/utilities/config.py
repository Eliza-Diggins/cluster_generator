"""Cluster generator configuration management / utilities."""
import operator
import os
import pathlib as pt
from functools import reduce
from typing import Any, Collection, Iterable, Mapping

import ruamel.yaml
from unyt import Unit, unyt_array, unyt_quantity

from cluster_generator.utilities.types import AttrDict

config_directory = os.path.join(pt.Path(__file__).parents[1], "bin", "config.yaml")
""" str: The system directory where the ``cluster_generator`` configuration is stored.

The underlying ``.yaml`` file may be altered by the user to set configuration values.
"""

# Configure the ruamel.yaml environment to allow custom / 3rd party datatypes in the configuration yaml.
yaml = ruamel.yaml.YAML()
yaml.register_class(unyt_array)
yaml.register_class(unyt_quantity)
yaml.register_class(Unit)


class YAMLConfiguration:
    """Generic YAML configuration class."""

    def __init__(self, path: pt.Path | str):
        self.path: pt.Path = pt.Path(path)
        # :py:class:`pathlib.Path`: The path to the underlying yaml file.
        self._config: ruamel.yaml.CommentedMap | None = None

    @property
    def config(self):
        if self._config is None:
            self._config = self.load()

        return AttrDict(self._config)

    @classmethod
    def load_from_path(cls, path: pt.Path) -> dict:
        """Read the configuration dictionary from disk."""
        try:
            with open(path, "r+") as cf:
                return yaml.load(cf)

        except FileNotFoundError as er:
            raise FileNotFoundError(
                f"Couldn't find the configuration file! Is it at {config_directory}? Error = {er.__repr__()}"
            )

    def load(self) -> dict:
        return self.__class__.load_from_path(self.path)

    def reload(self):
        """Reload the configuration file from disk."""
        self._config = None

    @classmethod
    def set_on_disk(cls, path: pt.Path | str, name: str | Collection[str], value: Any):
        _old = cls.load_from_path(path)

        if isinstance(name, str):
            name = name.split(".")
        else:
            pass

        setInDict(_old, name, value)

        with open(path, "w") as cf:
            yaml.dump(_old, cf)

    def set_param(self, name: str | Collection[str], value):
        self.__class__.set_on_disk(self.path, name, value)


cgparams: YAMLConfiguration = YAMLConfiguration(config_directory)
""":py:class:`YAMLConfiguration`: The ``cluster_generator`` configuration object."""


def getFromDict(dataDict: Mapping, mapList: Iterable[slice]) -> Any:
    """Fetch an object from a nested dictionary using a list of keys.

    Parameters
    ----------
    dataDict: dict
        The data dictionary to search.
    mapList: list
        The list of keys to follow.

    Returns
    -------
    Any
        The output value.
    """
    return reduce(operator.getitem, mapList, dataDict)


def setInDict(dataDict: Mapping, mapList: Iterable[slice], value: Any):
    """Set the value of an object from a nested dictionary using a list of keys.

    Parameters
    ----------
    dataDict: dict
        The data dictionary to search.
    mapList: list
        The list of keys to follow.
    value: Any
        The value to set the object to
    """
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value
