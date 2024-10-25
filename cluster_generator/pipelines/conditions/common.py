"""Common / generic conditions which see frequent use.
"""
from functools import wraps

from cluster_generator.pipelines.conditions._types import ConditionLike
from cluster_generator.pipelines.conditions.abc import Condition

def condition(enabled: bool = True):
    """
    A decorator that marks a function as a condition for use in a pipeline system.

    This decorator is intended to be applied to methods or functions that will act as
    conditions in a pipeline. A condition is a callable that determines whether a
    particular solver or task should be executed, based on the state of the pipeline
    or model.

    When applied, the decorator attaches an `_is_condition` attribute to the function,
    marking it as available for use within the pipeline system.

    Parameters
    ----------
    enabled : bool, optional
        If `True` (default), the function will be marked as a condition. If `False`,
        the decorator does nothing, and the function will not be registered as a condition.

    Returns
    -------
    Callable
        The original function with an added `_is_condition` attribute set to `True`,
        or the original function unmodified if `enabled` is `False`.

    Raises
    ------
    ValueError
        If `func` is not callable, a `ValueError` is raised to ensure that only
        valid functions or methods are decorated.

    Examples
    --------
    Decorating a function as a condition:

    >>> @condition()
    ... def my_condition_function(pipeline, grid, result):
    ...     # Check some condition here
    ...     return True  # or False

    You can disable the condition marking by passing `enabled=False`:

    >>> @condition(enabled=False)
    ... def not_a_condition_function():
    ...     # This will not be registered as a condition
    ...     pass
    """

    def _condition(func: ConditionLike):
        if not callable(func):
            raise ValueError(f"Expected a callable, but got {type(func).__name__}.")

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            return func(*args, **kwargs)

        if enabled:
            wrapped_func._is_condition = True
        return wrapped_func

    return _condition

ALWAYS,NEVER = Condition(lambda _,__,___: True), Condition(lambda _,__,___: False)
