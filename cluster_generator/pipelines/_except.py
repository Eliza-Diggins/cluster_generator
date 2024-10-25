class SolverError(Exception):
    pass


class PipelineError(Exception):
    pass


class PipelineInitError(PipelineError):
    pass

class PipelineClassError(PipelineError):
    pass

class PipelineHandleNotFoundError(PipelineError):
    pass

class MissingScratchHandlerError(PipelineError):
    pass

class MissingTaskError(PipelineError):
    def __init__(self,pipeline,task):
        self.message = f'Failed to find task {task} in {pipeline}.'
