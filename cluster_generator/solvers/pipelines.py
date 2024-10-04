from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

from cluster_generator.solvers._abc import Pipeline, Solver
from cluster_generator.solvers._types import DEFAULT_SOLVER_REGISTRY

if TYPE_CHECKING:
    from cluster_generator.models._abc import ClusterModel
    from cluster_generator.solvers._types import ModelSolverRegistry


class _GenericPipeline(Pipeline):
    FIELD_ORDER = []

    def __init__(self, model: "ClusterModel", registry: "ModelSolverRegistry" = None):
        self.model = model
        self.registry: ModelSolverRegistry = registry or DEFAULT_SOLVER_REGISTRY

        # Construct an ordered dictionary of fields and solvers.
        self.procedure: OrderedDict[str, Optional[Solver]] = OrderedDict(
            {field: None for field in self.FIELD_ORDER}
        )

    def __setitem__(self, field: str, solver: Solver) -> None:
        """
        Assign a solver to a specific field in the pipeline.

        Parameters
        ----------
        field : str
            The field for which the solver will be assigned.
        solver : Solver
            The solver to assign to the field.

        Raises
        ------
        ValueError
            If the solver's field does not match the expected field.
        KeyError
            If the field is not part of the pipeline.
        """
        if solver.field != field:
            raise ValueError(
                f"Solver field '{solver.field}' does not match expected field '{field}'."
            )
        self.procedure[field] = solver


class Density_TotalDensityPipeline(Pipeline):
    FIELD_ORDER = [
        "density",
        "total_density",
        "dark_matter_density",
        "gravitational_field",
        "pressure",
        "temperature",
    ]


class Density_TemperaturePipeline(Pipeline):
    pass
