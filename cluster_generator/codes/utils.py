"""
Utilities for working with external codes.
"""
from dataclasses import field


def ufield(*args, **kwargs):
    """User specified field / setting field"""
    metadata = kwargs.pop("metadata", {})
    metadata["type"] = "U"
    metadata["setter"] = kwargs.pop("setter", None)
    metadata["flag"] = kwargs.pop("flag", None)

    return field(*args, **kwargs, metadata=metadata)


def cfield(*args, **kwargs):
    """compile-time field."""
    metadata = kwargs.pop("metadata", {})
    metadata["type"] = "C"
    return field(*args, **kwargs, metadata=metadata)


def _const_factory(value):
    return lambda: value
