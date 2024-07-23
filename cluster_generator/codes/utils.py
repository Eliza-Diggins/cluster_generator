"""Utilities for working with external codes."""
from dataclasses import field


def ufield(*args, **kwargs):
    """User specified field / setting field."""
    metadata = kwargs.pop("metadata", {})
    metadata["type"] = "U"
    metadata["setter"] = kwargs.pop("setter", None)
    metadata["flag"] = kwargs.pop("flag", None)
    metadata["allowed_values"] = kwargs.pop("av", None)

    return field(*args, **kwargs, metadata=metadata)


def cfield(*args, **kwargs):
    """Compile-time field."""
    metadata = kwargs.pop("metadata", {})
    metadata["type"] = "C"
    metadata["allowed_values"] = kwargs.pop("av", None)
    return field(*args, **kwargs, metadata=metadata)


def _const_factory(value):
    return lambda: value
