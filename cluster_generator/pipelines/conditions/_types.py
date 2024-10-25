from typing import Callable, Any, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from cluster_generator.pipelines.abc import Pipeline
    from cluster_generator.grids.grids import Grid

ConditionLike = Union['Condition', Callable[['Pipeline', 'Grid', Any], bool]]