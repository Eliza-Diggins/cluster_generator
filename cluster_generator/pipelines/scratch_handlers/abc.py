import h5py
from typing import TYPE_CHECKING
from cluster_generator.utilities.io import HDF5FileHandler
from ._except import ScratchHandlerException

if TYPE_CHECKING:
    from cluster_generator.pipelines.abc import Pipeline

class ScratchSpaceHandler(HDF5FileHandler):
    """
    A handler to manage scratch space within an HDF5 pipeline, inheriting
    general functionality from `HDF5FileHandler` while adding methods specific
    to managing temporary scratch datasets and groups.
    """
    def __init__(self, pipeline: 'Pipeline'):
        """
        Initialize the scratch space handler using the pipeline's HDF5 handle.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline instance that provides the HDF5 file or group handle.
        """
        # Validation. Ensure that the pipeline has a handle that we can save in and
        # create the relevant scratch space.
        if not pipeline.handle:
            raise ScratchHandlerException(f'Cannot initialize {self.__class__.__name__} in pipline {pipeline} because it '
                                          f'does not have an io handle. This generally indicates that the pipeline doesn\'t '
                                          f'have a file it is saved in. Ensure that the pipeline has a handle and try again.')

        # Ensure that the handle has been created if it doesn't exist.
        _handle = pipeline.handle.require_group(pipeline.__class__.SCRATCH_GROUP_NAME)

        # Instantiate the HDF5 handler.
        super().__init__(pipeline.handle)

        # Set basic attributes.
        self.pipeline = pipeline
        self.pipeline._scratch = self

    def __str__(self):
        return f"<{self.__class__.__name__} | {self.pipeline}>"

    def __repr__(self):
        return f"<ScratchSpaceHandler(handle='{self.handle.name}',file='{self.handle.file}')>"
