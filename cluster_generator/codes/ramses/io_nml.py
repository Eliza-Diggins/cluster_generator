"""Backend module for compatibility with Fortran90 style namelist files.

Notes
-----

These are primarily used in conjunction with the RAMSES code frontend (written in F90) to ensure that
RTP files are written correctly.
"""
import re
from pathlib import Path
from typing import Any, Collection, Literal

from cluster_generator.utilities.types import Instance


class F90NamelistGroup(dict):
    """A subgroup of an F90 namelist file.

    Notes
    -----

    Namelist files contain a number of **namelists**, each of which is formatted as

    .. code-block::

        &NAMELIST_NAME
            key = value
            key = value
            ...
        /

    The :py:class:`F90NamelistGroup` represents the key-value map of a single on of these groups.

    To initialize a :py:class:`F90NamelistGroup`, you need only provide a dictionary and it will automatically be converted.
    """

    def __init__(self, *args, name: str = None, owner: Instance = None, **kwargs):
        """Initialize the namelist dictionary.

        Parameters
        ----------
        *args
            The arguments to pass to the ``dict`` initializer.
        **kwargs
            The arguments to pass to the ``dict`` initializer.
        name: str, optional
            The name of the group. By default, ``"generic"``.
        owner: F90Namelist, optional
            The owner file. If specified, then alterations to this group will
            automatically be reflected in the file.
        """

        # super initialization
        super(F90NamelistGroup, self).__init__(*args, **kwargs)

        # Set the name
        self._name = name if name is not None else "generic"
        """ str: The name of this namelist group."""

        self._owner: Instance = owner
        """ F90Namelist: The owner of this group."""

        self._live_update = False  # flag for live updating.

    def __str__(self):
        return f"<F90nml GROUP {self.name} ({len(self)} attrs)>"

    def __repr__(self):
        return self.__str__()

    def __setitem__(self, key: str, value: Any):
        # we need to set in the active class and also check for live.

        super(F90NamelistGroup, self).__setitem__(key, value)

    @property
    def name(self):
        """str: The name of this namelist group."""
        return self._name

    @name.setter
    def name(self, value: str):
        # ensure that the name changes in the parent.
        _old_name, _new_name = self.name, value

        if self._owner is not None:
            self._owner._groups = {
                (k if k != _old_name else _new_name): v
                for k, v in self._owner.groups.items()
            }

        self._name = _new_name

    @property
    def owner(self):
        """F90Namelist: The owner of this group."""
        return self._owner

    @owner.setter
    def owner(self, value: Instance):
        if self._owner is not None:
            raise ValueError("Cannot reallocate ownership of an F90 namelist group.")

        self._owner = value

        # ensure we get added.
        if value is not None:
            self._owner[self.name] = self

    def update_file(self):
        self.owner.write()

    @staticmethod
    def _format_value(value):
        """Format the value based on its type for Fortran namelist syntax."""
        if isinstance(value, str):
            return f"'{value}'"
        elif isinstance(value, bool):
            return ".true." if value else ".false."
        else:
            return str(value)

    def write(
        self,
        path: str | Path,
        overwrite: bool = False,
        if_exists: Literal["error", "replace", "merge"] = "error",
    ):
        """Write this group to a specified ``path``. If the ``path`` corresponds to a
        pre-existing F90 nml file, then this group will be added to that file with
        behaviour depending on the ``if_exists`` kwarg.

        Parameters
        ----------
        path: str
            The path at which to write the file.
        overwrite: bool
            If ``True``, any existing data will be overwritten and the group
            will be written to a fresh file.
        if_exists: str
            If ``"error"`` (default), then an error is raised if this group already
            exists in the file. If ``"merge"``, then the groups are merged with duplicates
            taking the value of this group. Finally, if ``"replace"``, then the group will simply be
            replaced.

        Returns
        -------
        None

        Notes
        -----
        This is all managed by reading the file as a full F90 nml and then adding / removing groups.
        """
        path = Path(path)
        namelist = F90Namelist(path)  # read the entire file as an F90.

        # Dealing with the overwrite settings
        if overwrite:
            namelist._groups = {}  # Delete all groups from NML

        # adding the group to the namelist with correct if_exists
        namelist.add_group(self, if_exists=if_exists)

        # Write the namelist to disk
        namelist.write()

    def write_group_to_disk(self, fio):
        fio.write(f"&{self.name}\n")
        for var_name, var_value in self.items():
            if isinstance(var_value, list):
                # Join list elements with commas
                formatted_values = ", ".join(self._format_value(v) for v in var_value)
                fio.write(f"  {var_name} = {formatted_values}\n")
            else:
                fio.write(f"  {var_name} = {self._format_value(var_value)}\n")
        fio.write("/\n\n")


class F90Namelist:
    """Class representation of an F90 ``nml`` file.

    .. warning::

        This is a generic representation with considerable limitations. It is suitable
        for the uses of this package; however, it should not replace more complete
        tools like ``f90nml`` for complex NML management.
    """

    _group_start_pattern = re.compile(r"&(\w+)")
    _group_stop_pattern = re.compile(r"/")
    _assignment_pattern = re.compile(r"(\w+)\s*=\s*(.*)")

    def __init__(self, path: str | Path):
        """Initialize an F90 instance.

        Parameters
        ----------
        path: str
            The file path to open as an F90 namelist. May be non-existent, in which case
            a blank :py:class:`F90Namelist` is created around the path.
        """
        self.path: Path = Path(path)
        """ Path: The underlying file on disk where the NML exists."""
        self._groups = {}

        # Populate the namelist with its data.
        self._read()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.write()

    def _read(self):
        # Check for existence
        if not self.path.exists():
            return

        # Read the NML file from disk.
        with open(self.path, "r") as fio:
            for nml_line in fio:
                # Preprocessing the line. Remove all leading and trailing spaces
                # skip if its empty or starts / ends with !.
                nml_line = nml_line.strip()

                # Skip empty lines or comments (Fortran comments start with '!')
                if not nml_line or nml_line.startswith("!"):
                    continue

                # Checking for group status.
                group_match = self.__class__._group_start_pattern.match(nml_line)
                if group_match:
                    # We have a match to the group.
                    current_group = group_match.group(1)
                    self._groups[current_group] = F90NamelistGroup(
                        {}, name=current_group, owner=self
                    )
                    current_group = self._groups[current_group]
                    continue

                # Check for group end
                if self.__class__._group_stop_pattern.match(nml_line):
                    current_group = None
                    continue

                # Parse assignments
                if current_group is not None:
                    assignment_match = self.__class__._assignment_pattern.match(
                        nml_line
                    )
                    if assignment_match:
                        var_name, var_value = assignment_match.groups()
                        var_value = self._parse_value(var_value)

                        current_group[var_name] = var_value

    def __len__(self):
        return len(self._groups)

    def __getitem__(self, item: str) -> F90NamelistGroup:
        return self._groups.__getitem__(item)

    def __iter__(self):
        return iter(self.groups.values())

    def __contains__(self, item):
        return item in self._groups.keys()

    def __str__(self):
        return f"<F90nml {self.path}>"

    def __repr__(self):
        return self.__str__()

    @property
    def groups(self) -> dict[str, F90NamelistGroup]:
        """dict: The :py:class:`F90NamelistGroup` instances that make up the namelist file."""
        return self._groups

    def drop_group(self, group: str | F90NamelistGroup):
        """Delete a group from the namelist.

        Parameters
        ----------
        group: str or F90NamelistGroup
            The group to delete. May be string or namelist group.
        """
        # Retain only the group name.
        if isinstance(group, F90NamelistGroup):
            group = group.name

        # delete from the dictionary.
        try:
            del self._groups[group]
        except KeyError:
            raise ValueError(f"Not group {group} in {self}.")

    def add_group(
        self,
        group: str | F90NamelistGroup,
        if_exists: Literal["error", "replace", "merge"] = "error",
    ):
        """Add a group to this namelist.

        Parameters
        ----------
        group: F90NamelistGroup
            The group to add to the NML.
        if_exists: str
            If ``"error"`` (default), then an error is raised if this group already
            exists in the file. If ``"merge"``, then the groups are merged with duplicates
            taking the value of this group. Finally, if ``"replace"``, then the group will simply be
            replaced.
        """
        if isinstance(group, str):
            # we only have the group name as a string. Different procedure
            group_name = group
            _is_true_group = False
        else:
            group_name = group.name
            _is_true_group = True

        # Managing rules for duplication.
        if group_name in self.groups:
            if if_exists == "error":
                raise ValueError(
                    f"{group_name} is duplicated in {self} and if_exists = 'error'."
                )
            elif if_exists == "merge":
                # merge the groups, keeping the newer version
                if _is_true_group:
                    for k, v in self.items():
                        self._groups[group_name][k] = v
                else:
                    # we only got given a name. We just let it exist and move on.
                    pass
                return
            elif if_exists == "replace":
                # replace the namelist group. Remove the old group and then re-add.
                self.drop_group(group_name)
                self.add_group(group)
            else:
                raise ValueError(
                    f"Keyword argument 'if_exists' expected 'error', 'merge', or 'replace' not {if_exists}."
                )

        # proceed with adding the group
        if _is_true_group:
            self._groups[group_name] = group
            group.owner = self
        else:
            self._groups[group_name] = F90NamelistGroup({}, name=group_name, owner=self)

    def keys(self) -> Collection[str]:
        """list: the keys of the group dictionary."""
        return self._groups.keys()

    def values(self) -> Collection[F90NamelistGroup]:
        """list: the values of the group dictionary."""
        return self._groups.values()

    def items(self) -> Collection[tuple]:
        """list of tuple: the key-value pairs of the group dictionary."""
        return self._groups.items()

    @classmethod
    def _parse_value(cls, value):
        # Remove trailing comments
        value = value.split("!")[0].strip()

        # Check for list (comma-separated values)
        if "," in value:
            _ret_array = []
            for v in value.split(","):
                _u = cls._parse_value(v.strip())
                _ret_array += [_u] if not isinstance(_u, Collection) else _u
            return _ret_array

        # Check for string (enclosed in quotes)
        if value.startswith("'") and value.endswith("'"):
            return value[1:-1]

        # Check for duplication marker
        if "*" in value:
            _duplicator_int, _duplicator_value = value.split("*")
            _duplicator_int = int(_duplicator_int.strip())
            _duplicator_value = [
                cls._parse_value(_duplicator_value.strip())
            ] * _duplicator_int

            return _duplicator_value

        # Check for boolean
        if value.lower() == ".true.":
            return True
        if value.lower() == ".false.":
            return False

        # Try to parse as integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try to parse as float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string if nothing else matches
        return value

    def write(self):
        """Write this F90nml instance to file."""
        if self.path.exists():
            self.path.unlink()

        with open(self.path, "w") as fio:
            for _, group in self.items():
                group.write_group_to_disk(fio)
