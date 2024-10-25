import logging
from abc import ABC, abstractmethod
import logging
import sys
from cluster_generator.utilities.config import cgparams

# Setting up the logging system
streams = dict(
    mylog=getattr(sys, cgparams['logging','mylog','stream']),
    devlog=getattr(sys, cgparams['logging','devlog','stream']),
)
_loggers = dict(
    mylog=logging.Logger("cluster_generator"), devlog=logging.Logger("cg-development")
)

_handlers = {}

for k, v in _loggers.items():
    # Construct the formatter string.
    _handlers[k] = logging.StreamHandler(streams[k])
    _handlers[k].setFormatter(logging.Formatter(cgparams['logging',k,'format']))
    v.addHandler(_handlers[k])
    v.setLevel(cgparams['logging',k,'level'])
    v.propagate = False

    if k != "mylog":
        v.disabled = not cgparams['logging',k,'enabled']

mylog: logging.Logger = _loggers["mylog"]
""":py:class:`logging.Logger`: The main logger for ``pyXMIP``."""
devlog: logging.Logger = _loggers["devlog"]
""":py:class:`logging.Logger`: The development logger for ``pyXMIP``."""


class LogDescriptor(ABC):
    LOG_CLASS = logging.Logger  # Default to the standard Logger; can be overridden in subclasses

    def __get__(self, instance, owner) -> LOG_CLASS:
        if not hasattr(owner, "_logger") or owner._logger is None:
            # Fetch the logger default and then set the logger class to
            # the one specified by the descriptor class.
            original_logger_class = logging.getLoggerClass()
            logging.setLoggerClass(self.LOG_CLASS)

            try:
                # Get the logger as an instance of LOGCLASS
                owner._logger = logging.getLogger(owner.__name__)
                self.configure_logger(owner._logger)
            finally:
                # Restore the original logging class
                logging.setLoggerClass(original_logger_class)

        return owner._logger

    @abstractmethod
    def configure_logger(self, logger):
        pass

if __name__ == '__main__':
    print(cgparams.logging)