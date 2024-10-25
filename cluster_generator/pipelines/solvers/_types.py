from typing import Callable, Union, TypeVar, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from cluster_generator.grids.grids import Grid
    from cluster_generator.pipelines.abc import Pipeline
    from cluster_generator.pipelines.solvers.abc import Solver

# Type aliases for arguments
PipelineType = 'Pipeline'
GridType = 'Grid'

# A generic type for return values from Solver-like callables
ReturnType = TypeVar('ReturnType', bound=Any)

# Solver-like type, accepting either a Solver instance or a callable with specific arguments
SolverLike = Union['Solver', Callable[[PipelineType, GridType], ReturnType]]

# Type alias for validator functions used to validate solvers
Validator = Callable[[PipelineType], bool]

# Type alias for setup methods that prepare solvers in the pipeline context
SetupMethod = Callable[[PipelineType], Any]
