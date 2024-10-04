from typing import TYPE_CHECKING

from cluster_generator.solvers._abc import Solver
from cluster_generator.utils import mylog

if TYPE_CHECKING:
    from cluster_generator.grids.grids import Grid
    from cluster_generator.models._abc import ClusterModel


class InjectionSolver(Solver):
    """
    A solver class that handles the injection of values for a specific field in the model based on
    predefined profiles.

    Attributes
    ----------
    FIELDS : None
        This solver can be applied to any field.
    PRIORITY : int
        The priority of the solver (0 by default).
    """

    FIELDS = None
    PRIORITY = 0

    def __init__(self, model: "ClusterModel", field: str = None):
        """
        Initialize the InjectionSolver.

        Parameters
        ----------
        model : ClusterModel
            The model associated with the solver.
        field : str, optional
            The field this solver will work on. If not provided, defaults to None.
        """
        super().__init__(model, field=field)

    def __call__(self, grid: "Grid", *args, **kwargs) -> None:
        """
        Perform the injection operation for the given field on the specified grid.

        The solver uses the model's profile to inject values into the field for each grid cell.

        Parameters
        ----------
        grid : Grid
            The grid on which the solver will operate.
        *args, **kwargs
            Additional arguments that can be passed to the solver for customization.
        """
        # Fetch the profile from the model
        profile = self.model.get_profile(self.field)
        if hasattr(self.model, self.field):
            units, dtype = (
                getattr(self.model, self.field).units,
                getattr(self.model, self.field).dtype,
            )
        else:
            units, dtype = "", "f8"

        # Inject values into the grid for this field
        grid.add_field_from_profile(profile, units=units, dtype=dtype, **kwargs)

        mylog.info(f"Injected values for field '{self.field}' using {self}.")

    @classmethod
    def check_availability(cls, model: "ClusterModel", field: str = None) -> bool:
        """
        Check if the InjectionSolver is available for the given field in the model.

        This solver is available if the model has a profile for the specified field.

        Parameters
        ----------
        model : ClusterModel
            The model to check for the profile.
        field : str, optional
            The field that needs solving.

        Returns
        -------
        bool
            True if the solver is available, False otherwise.
        """
        return model.has_profile(field)
