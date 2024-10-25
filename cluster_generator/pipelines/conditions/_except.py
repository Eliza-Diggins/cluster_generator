"""Exception classes for Conditions
"""

class ConditionError(Exception):
    """
    Generic exception raised whenever issues arise from working with conditions.
    """
    pass

class ConditionSerializationError(ConditionError):
    """
    Error raised when an issue occurs while serializing conditions.
    """
    pass