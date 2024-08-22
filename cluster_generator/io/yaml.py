"""I/O Utilities for cluster generator."""
from typing import TYPE_CHECKING

import unyt
from ruamel.yaml import Loader, MappingNode, Node, ScalarNode, SequenceNode

if TYPE_CHECKING:
    pass


def unyt_array_constructor(loader: "Loader", node: "Node") -> unyt.unyt_array:
    """Custom constructor function for :py:class:`unyt.unyt_array`.

    Parameters
    ----------
    loader: :py:class:`Loader`
        The YAML loader.
    node: :py:class:`Node`
        The node to construct.

    Returns
    -------
    unyt.unyt_array
        The unyt array constructed.
    """
    if isinstance(node, ScalarNode):
        raise ValueError(
            f"Cannot reconstruct an unyt_array object from a ScalarNode {ScalarNode}."
        )
    elif isinstance(node, SequenceNode):
        # Assume format is [x_0,x_1,..., unit]
        sequence = loader.construct_sequence(node)

        try:
            return unyt.unyt_array(sequence[:-1], sequence[-1])
        except Exception:
            raise ValueError(
                f"Failed to construct unyt_array from YAML sequence {sequence}. (NODE={node})."
            )
    elif isinstance(node, MappingNode):
        mapping = loader.construct_mapping(node)

        try:
            array = mapping["array"]
            unit = mapping["unit"]
        except KeyError:
            raise KeyError(
                f"YAML node {node} should have keys array and unit to be parsed as unyt_array."
            )

        return unyt.unyt_array(array, unit)

    else:
        raise TypeError(f"Type {type(node)} not supported.")


def unyt_quantity_constructor(loader: "Loader", node: "Node") -> unyt.unyt_array:
    """Custom constructor function for :py:class:`unyt.unyt_quantity`.

    Parameters
    ----------
    loader: :py:class:`Loader`
        The YAML loader.
    node: :py:class:`Node`
        The node to construct.

    Returns
    -------
    unyt.unyt_array
        The unyt array constructed.
    """
    if isinstance(node, ScalarNode):
        scalar = loader.construct_scalar(node)
        if not isinstance(scalar, str):
            raise ValueError(
                f"Cannot parse unyt_quantity from scalar of type {type(scalar)}."
            )

        return unyt.unyt_quantity.from_string(scalar)

    elif isinstance(node, SequenceNode):
        # Assume format is [x_0,x_1,..., unit]
        sequence = loader.construct_sequence(node)

        if not len(sequence) == 2:
            raise ValueError(
                f"{node} has length {len(sequence)}, which cannot be parsed as unyt_quantity."
            )

        return unyt.unyt_quantity(sequence[0], sequence[1])

    elif isinstance(node, MappingNode):
        mapping = loader.construct_mapping(node)
        try:
            value = mapping["value"]
            unit = mapping["unit"]
        except KeyError:
            raise KeyError(
                f"YAML node {node} should have keys value and unit to be parsed as unyt_value."
            )

        return unyt.unyt_quantity(value, unit)

    else:
        raise TypeError(f"Type {type(node)} not supported.")
