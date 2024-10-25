class SolverError(Exception):
    """Base class for errors related to solvers."""
    pass


class SolverSetupError(SolverError):
    """Raised when there is an error during solver setup."""
    pass


class SolverValidationError(SolverError):
    """Raised when validation of a solver fails."""
    pass