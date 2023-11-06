"""
Testing structure for the collections available in cluster_collections.py
"""
import inspect
import os
import sys

import pytest

import cluster_generator.cluster_collections as cc
from cluster_generator.cluster_collections import Ascasibar07, Sanderson10, Vikhlinin06
from cluster_generator.correction import NonPhysicalRegion
from cluster_generator.model import ClusterModel
from cluster_generator.tests.utils import model_answer_testing
from cluster_generator.utils import mylog

_ignored_classes = ["ProtoCluster", "Collection"]
print([k for k in sys.modules if "cluster_generator" in k])


def _locate_collections():
    """Locates the currently implemented collections"""
    class_members = inspect.getmembers(
        sys.modules["cluster_generator.cluster_collections"], inspect.isclass
    )

    collection_names = [
        k[0]
        for k in class_members
        if (k[1].__module__ == "cluster_generator.cluster_collections")
        and (k[0] not in _ignored_classes)
    ]

    return collection_names


@pytest.mark.parametrize("collection_name", _locate_collections())
def test_exists(collection_name):
    """Checks that the necessary files actually exist for each of the collections"""
    assert hasattr(cc, collection_name)
    assert os.path.exists(getattr(cc, collection_name)._data)
    assert os.path.exists(getattr(cc, collection_name)._schema_loc)
    _ = getattr(cc, collection_name)()


@pytest.mark.usefixtures("answer_store", "answer_dir")
class TestCollection:
    """Base class for testing the collection's system."""

    collection = None

    def test_schema(self):
        """Check the schema is complete"""
        if self.collection is None:
            return True

        c = self.collection()
        assert hasattr(c, "_schema"), "failed to locate a schema"

        assert "main" in c._schema, "Failed to find main key in schema"
        assert "schema" in c._schema, "Failed to find schema key in schema."

        assert all(
            i in c._schema["main"]
            for i in [
                "collection_name",
                "collection_authors",
                "source_authors",
                "citation",
                "n_clusters",
            ]
        ), "Missing keys found in schema[main]."
        assert all(
            i in c._schema["schema"] for i in ["profiles", "build"]
        ), "Missing required schema heading."

        for key in c._schema["schema"]["profiles"]:
            assert key in [
                "density",
                "total_density",
                "temperature",
                "entropy",
            ], "Invalid field name."

            assert all(
                [
                    i in c._schema["schema"]["profiles"][key]
                    for i in ["parameters", "function"]
                ]
            ), f"Missing header in {key}"

        assert hasattr(ClusterModel, c._schema["schema"]["build"]["method"])

    def test_construct(self, answer_dir, answer_store):
        """test random construction"""
        if self.collection is None:
            return True
        c = self.collection()

        if answer_store:
            for k, v in c.clusters.items():
                mylog.warning(f"Checking {k}")
                _v = v.load(1, 10000)
                _v = NonPhysicalRegion.correct(_v)
                model_answer_testing(
                    _v, f"{c.name}-{k}-mdl.h5", answer_store, answer_dir
                )
        else:
            import numpy as np

            keys = list(c.clusters.keys())
            rand = np.random.randint(0, len(keys), 3)
            keys = [k for i, k in enumerate(keys) if i in rand]

            for k, v in c.clusters.items():
                if k in keys:
                    mylog.warning(f"Checking {k}")
                    _v = v.load(1, 10000)
                    _v = NonPhysicalRegion.correct(_v)
                    model_answer_testing(
                        _v, f"{c.name}-{k}-mdl.h5", answer_store, answer_dir
                    )

    def test_metadata(self, answer_dir, answer_store):
        """test random construction"""
        if self.collection is None:
            return True


@pytest.mark.usefixtures("answer_store", "answer_dir")
class TestVikhlinin06(TestCollection):
    """Base class for testing the collection's system."""

    collection = Vikhlinin06


@pytest.mark.usefixtures("answer_store", "answer_dir")
class TestAscasibar07(TestCollection):
    """Base class for testing the collection's system."""

    collection = Ascasibar07


@pytest.mark.usefixtures("answer_store", "answer_dir")
class TestSanderson10(TestCollection):
    """Base class for testing the collection's system."""

    collection = Sanderson10
