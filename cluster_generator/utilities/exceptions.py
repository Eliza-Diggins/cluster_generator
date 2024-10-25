import warnings
from tqdm.asyncio import tqdm

class CGExceptionGroup(Exception):
    def __init__(self, message, exceptions):
        super().__init__(message)
        self.exceptions = exceptions

    def __str__(self):
        exception_messages = "\n".join(f"- {type(e).__name__}: {e}" for e in self.exceptions)
        return f"{self.args[0]}:\n{exception_messages}"

    def handle_exceptions(self, handler=None):
        for exc in self.exceptions:
            if handler:
                handler(exc)
            else:
                print(f"Handling exception: {exc}")


class tqdmWarningRedirector:
    """
    A context manager to redirect all warnings and log them through tqdm.write()
    so that they don't interfere with the progress bar.
    """

    def __enter__(self):
        # Backup the original warnings.showwarning
        self._original_showwarning = warnings.showwarning

        # Override the warning display to use tqdm.write
        warnings.showwarning = self._tqdm_warning_handler
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the original warning display function
        warnings.showwarning = self._original_showwarning

    @staticmethod
    def _tqdm_warning_handler(
            message, category, filename, lineno, file=None, line=None
    ):
        """
        Custom handler to redirect warnings through tqdm.write().
        """
        tqdm.write(f"WARNING: {message}, in {filename}, line {lineno}")
