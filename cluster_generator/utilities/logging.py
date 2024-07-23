"""Logging management."""
import logging
import sys

from cluster_generator.utilities.config import cgparams

# Setting up the logging system
streams = dict(
    mylog=getattr(sys, cgparams.config.logging.mylog.stream),
    devlog=getattr(sys, cgparams.config.logging.devlog.stream),
)
_loggers = dict(
    mylog=logging.Logger("cluster_generator"), devlog=logging.Logger("CG-DEV")
)

_handlers = {}

for k, v in _loggers.items():
    # Construct the formatter string.
    _handlers[k] = logging.StreamHandler(streams[k])
    _handlers[k].setFormatter(
        logging.Formatter(getattr(cgparams.config.logging, k).format)
    )
    v.addHandler(_handlers[k])
    v.setLevel(getattr(cgparams.config.logging, k).level)
    v.propagate = False

    if k != "mylog":
        v.disabled = not getattr(cgparams.config.logging, k).enabled

mylog: logging.Logger = _loggers["mylog"]
""":py:class:`logging.Logger`: The main logger for ``pyXMIP``."""
devlog: logging.Logger = _loggers["devlog"]
""":py:class:`logging.Logger`: The development logger for ``pyXMIP``."""


# Defining special exceptions and other error raising entities
class ErrorGroup(Exception):
    """Special error class containing a group of exceptions."""

    def __init__(self, message: str, error_list: list[Exception]):
        self.errors: list[Exception] = error_list

        # Determining the message
        self.message = f"{len(self.errors)} ERRORS: {message}\n\n"

        for err_id, err in enumerate(self.errors):
            self.message += "%(err_id)-5s[%(err_type)10s]: %(msg)s\n" % dict(
                err_id=err_id + 1, err_type=err.__class__.__name__, msg=str(err)
            )

    def __str__(self):
        return self.message
