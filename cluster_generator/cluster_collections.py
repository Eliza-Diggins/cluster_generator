"""
The :py:mod:`cluster_collections` module provides user access to pre-existing radial profile models of known systems.
"""
import os
from pathlib import Path

import pandas as pd
import yaml

from cluster_generator.model import ClusterModel
from cluster_generator.radial_profiles import RadialProfile
from cluster_generator.utils import _bin_directory, _get_loader, mylog

_dir = Path(os.path.join(_bin_directory, "collections"))


def _enforce_format(schema):
    """Enforces formatting norms on the schema"""
    if any([i not in schema for i in ["main", "schema"]]):
        raise SyntaxError(
            "The schema file doesn't have either section 'main' or 'schema'"
        )

    for main_header in [
        "collection_name",
        "collection_authors",
        "source_authors",
        "n_clusters",
    ]:
        if main_header not in schema["main"]:
            raise SyntaxError(f"The key {main_header} is not in schema[main].")

    assert hasattr(
        ClusterModel, schema["schema"]["build"]["method"]
    ), f"The build method {schema['schema']['build']['method']} is not valid."


class ProtoCluster:
    """
    The :py:class:`~cluster_collections.ProtoCluster` class is a precursor to the fully realized :py:class`model.ClusterModel` class.
    These are used as minimally memory intensive placeholders to allow the user to easily load the full cluster model.
    """

    def __init__(self, profiles, parameters, names, load_method):
        """
        Loads the :py:class:`~cluster_collections.ProtoCluster` instance.

        Parameters
        ----------
        profiles: dict of str: callable or dict of str: str
            A dictionary containing the profile definitions for the cluster initialization. There are two available options
            for the formatting of each element in this argument:

            - If the dictionary value is a :py:class:`str` type, then it is assumed to be a pre-defined profile already defined in the
              :py:mod:`cluster_generator.radial_profiles` module. If a corresponding built-in profile cannot be found, an error
              will be raised.
            - If the dictionary value is a ``callable`` instance of any kind, it is assumed to be a user defined function (either explicit or lambda).
              It will be wrapped in a :py:class:`cluster_generator.radial_profiles.RadialProfile` instance during initialization.

        parameters: dict of str: list
            A dictionary containing the parameters for the profiles. For each profile in ``profiles``, there should be a corresponding key-value pair
            in the ``parameters`` argument containing a :py:class:`list` with the values of each of the necessary parameters for the specific cluster
            being modeled.

            .. note::

                If the type of the elements in the list is :py:class:`float` or :py:class:`int`, then it will be assumed that these parameters are
                already following the unit conventions of the CGP. If there is any doubt, we recommend passing the parameters as :py:class:`unyt.array.unyt_quantity`
                instances instead. These will be processed to the correct units before proceeding.

        load_method: str
            The ``load_method`` should correspond to the analogous class method on :py:class:`model.ClusterModel` for loading the full model instance.
            Typically, these should be something like ``from_dens_and_temp`` or ``from_dens_and_tden``.
        """
        self.load_method = load_method
        self.profiles = {}

        for k in profiles:
            if isinstance(profiles[k], str):
                # This is a built-in and we should look it up.
                if profiles[k] in RadialProfile.builtin:
                    self.profiles[k] = RadialProfile.built_in(
                        profiles[k], *parameters[k]
                    )
                else:
                    raise ValueError(
                        f"The profile type {profiles[k]} is not recognized as a built-in profile"
                    )
            else:
                # This is a newly defined type instance.
                self.profiles[k] = RadialProfile(
                    lambda x, p=profiles[k], params=parameters[k]: p(x, *params),
                    name=names[k],
                )  # Double lambda to avoid late interpretation in lambdas.

    def load(self, rmin, rmax, additional_args=None, **kwargs):
        """
        Loads a :py:class:`model.ClusterModel` instance from this :py:class:`cluster_collections.ProtoCluster` instance.

        Parameters
        ----------
        rmin: float
            The minimum radius of the generation regime. (kpc)
        rmax: float
            The maximum radius of the generation regime. (kpc)
        additional_args: dict, optional
            The ``additional_args`` argument allows the user to pass additional arguments into the initialization function. Generally,
            there should be no reason to specify this unless some alteration has been made to the underlying source code.
        kwargs: dict, optional
            Additional key-word arguments to pass along to the initialization function.

            .. note::

                If ``kwargs`` contains a ``stellar_density`` key and corresponding profile, it will be overridden if the
                underlying :py:class:`cluster_collections.ProtoCluster` instance also has a stellar density profile.

        Returns
        -------

        """
        import inspect

        if additional_args is None:
            additional_args = {}

        load_method = getattr(ClusterModel, self.load_method)

        signature = (
            str(inspect.signature(load_method)).replace(" ", "")[1:-1].split(",")
        )  # deconstruct so that we use it to get the signature right.
        arg_sig = [i for i in signature if all(j not in i for j in ["=", "**"])]

        args = []

        for arg in arg_sig:
            if arg in self.profiles:
                args.append(self.profiles[arg])
            elif arg == "rmin":
                args.append(rmin)
            elif arg == "rmax":
                args.append(rmax)
            elif arg in additional_args:
                args.append(additional_args[arg])
            else:
                raise ValueError(
                    f"Determined that {arg} is a required item in the call signature, but it could not be found but it wasn't found in the additional_args dict."
                )

        if "stellar_density" in self.profiles:
            stellar_density = self.profiles["stellar_density"]
        elif "stellar_density" in kwargs:
            stellar_density = kwargs["stellar_density"]
            del kwargs["stellar_density"]
        else:
            stellar_density = None

        return load_method(*args, stellar_density=stellar_density, **kwargs)

    def keys(self):
        return self.profiles.keys()

    def items(self):
        return self.profiles.items()

    def value(self):
        return self.profiles.values()


class Collection:
    """
    The :py:class:`cluster_collections.Collection` class is the base class for all of the cluster collections available in the
    CGP. Generally, this class should not be instantiated but rather the specific sub-class corresponding to the user's
    desired database.
    """

    def __init__(self, data, schema):
        """
        Initializes the :py:class:`cluster_collections.Collection` instance.

        Parameters
        ----------
        data: str or :py:class:`pathlib.Path` or :py:class:`pandas.DataFrame`
            The parameter data for the collection. If provided as a :py:class:`str` or a :py:class:`~pathlib.Path` object, then
            the resulting path should point to a ``.csv`` file containing the relevant parameters. If a :py:class:`pandas.DataFrame` instance
            is provided, then the instance should be a table containing each of the clusters in a column ``"name"`` and a float value for
            each subsequent parameter (column), which correspond (IN ORDER) with the arguments of the profile functions.
        schema: str or :py:class:`pathlib.Path` or dict
            The collection schema.
        """
        if isinstance(schema, (str, Path)):
            mylog.info(f"Loading collection schema from path: {schema}")

            try:
                with open(schema, "r") as f:
                    self._schema = yaml.load(f, _get_loader())
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Failed to locate the schema file at {schema}."
                )
            except yaml.YAMLError as exp:
                raise SystemError(
                    f"The format the schema file does not comply with standards: {exp.__repr__()}"
                )

        elif isinstance(schema, dict):
            self._schema = schema
        else:
            raise TypeError("Input 'schema' was not str, Path, or dict.")

        _enforce_format(self._schema)

        mylog.info(
            f"Loaded schema for collection {self._schema['main']['collection_name']}."
        )

        mylog.info(
            f"Loading the dataset for {self._schema['main']['collection_name']}."
        )
        if isinstance(data, (str, Path)):
            try:
                self.db = pd.read_csv(data)
            except FileNotFoundError:
                raise FileNotFoundError(f"The database file {data} was not found.")
        elif isinstance(data, pd.DataFrame):
            self.db = data
        else:
            raise TypeError(
                f"The 'data' argument had type {type(data)} not str, Path, or pd.DataFrame."
            )

        self.clusters = {}

        self._initialize_proto_clusters()

        mylog.info(f"Initialized {self}.")

    def __len__(self):
        return self._schema["main"]["n_clusters"]

    def __repr__(self):
        return f"< Collection - {self.name} - {len(self)} >"

    def __str__(self):
        return f"{self.name} collection"

    def __getitem__(self, item):
        return self.clusters[item]

    def __contains__(self, item):
        return item in self.clusters

    def __class_getitem__(cls, item):
        inst = cls()
        return inst.__getitem__(item)

    def __iter__(self):
        return iter(self.clusters)

    def keys(self):
        """Returns the keys of the collection. Equivalent to ``self.cluster.keys()``"""
        return self.clusters.keys()

    def values(self):
        """Returns the  values of the collection. Equivalent to ``self.cluster.values()``"""
        return self.clusters.values()

    def items(self):
        """Returns the items of the collection. Equivalent to ``self.cluster.items()``"""
        return self.clusters.items()

    def _initialize_proto_clusters(self):
        """This produces the relevant proto-clusters from the available datasets"""
        self.clusters = {k: None for k in list(self.db["name"])}

        for cluster in self.clusters.keys():
            parameters = {
                k: self.db.loc[self.db["name"] == cluster, k].item()
                for k in self.db.columns[1:]
            }
            p = {}  # holds parameters after sorting.
            f = {}  # holds the functions
            n = {}  # holds the function names
            for profile, data in self.profiles.items():
                _params = data["parameters"]
                _f = data["function"]

                if isinstance(_f, str):
                    # This is a built-in and will get its name directly from the built-in name.
                    _n = ""
                else:
                    # This is the name specified in the schema.
                    _n = data["function_name"]

                f[profile] = _f
                p[profile] = [parameters[k] for k in _params]
                n[profile] = _n

            self.clusters[cluster] = ProtoCluster(
                f, p, n, self._schema["schema"]["build"]["method"]
            )

    @property
    def name(self):
        """The name of the collection."""
        return self._schema["main"]["collection_name"]

    @property
    def citation(self):
        """The citation (bibtex) for the data source."""
        try:
            return self._schema["main"]["citation"]
        except KeyError:
            mylog.warning(f"Failed to locate a citation for collection {self.name}")
            return None

    @property
    def authors(self):
        """The collection authors (not the original source author)"""
        return self._schema["main"]["collection_authors"]

    @property
    def source_authors(self):
        """The original (source) authors."""
        return self._schema["main"]["source_authors"]

    @property
    def profiles(self):
        """The profiles from the schema."""
        return self._schema["schema"]["profiles"]


class Vikhlinin06(Collection):
    _data = os.path.join(_dir, "Vikhlinin06.csv")
    _schema_loc = os.path.join(_dir, "Vikhlinin06.yaml")

    def __init__(self):
        super().__init__(self._data, self._schema_loc)


if __name__ == "__main__":
    vik = Vikhlinin06()

    h = vik.clusters["A133"].load(1, 10000)
    import matplotlib.pyplot as plt

    f, a = h.panel_plot()
    from correction import NonPhysicalRegion

    h = NonPhysicalRegion.correct(h, recursive=True)
    h.panel_plot(fig=f, axes=a)
    plt.show()
