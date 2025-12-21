import logging
import sys
import os


class Logger:
    def __init__(self, name, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        formatter = logging.Formatter(
            "[%(asctime)s %(name)s %(filename)s:%(lineno)d %(levelname)s] %(message)s"
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(stream_handler)

        # Track messages that have been printed once
        self._printed_once = set()

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    def debug(self, message):
        self.logger.debug(message)

    def info_once(self, message):
        """Print info message only once."""
        if message not in self._printed_once:
            self._printed_once.add(message)
            self.logger.info(message)

    def warning_once(self, message):
        """Print warning message only once."""
        if message not in self._printed_once:
            self._printed_once.add(message)
            self.logger.warning(message)

    def debug_once(self, message):
        """Print debug message only once."""
        if message not in self._printed_once:
            self._printed_once.add(message)
            self.logger.debug(message)


GLOBAL_LOGGER = None

# Global set to track printed-once messages across all loggers
_GLOBAL_PRINTED_ONCE = set()


def get_logger():
    global GLOBAL_LOGGER
    if GLOBAL_LOGGER is None:
        level = os.getenv("TEFL_LOG_LEVEL", "INFO").upper()
        GLOBAL_LOGGER = Logger("TE-FL", level)
    return GLOBAL_LOGGER


def print_once(message):
    """
    Simple function to print a message only once globally.

    Args:
        message: The message to print
    """
    if message not in _GLOBAL_PRINTED_ONCE:
        _GLOBAL_PRINTED_ONCE.add(message)
        print(message)


def debug_print_once(func_name: str, backend_name: str = "Backend", *args, **kwargs):
    """
    Print debug information for function calls, but only once per function.

    Args:
        func_name: Name of the function being called
        backend_name: Name of the backend
        *args: Function arguments
        **kwargs: Function keyword arguments
    """
    # Create a unique key based on backend and function name
    key = f"{backend_name}.{func_name}"

    if key not in _GLOBAL_PRINTED_ONCE:
        _GLOBAL_PRINTED_ONCE.add(key)
        print(f"[{backend_name}] Calling {func_name}")
        if args:
            print(f"  args: {[type(a).__name__ for a in args[:5]]}...")
        if kwargs:
            print(f"  kwargs: {list(kwargs.keys())[:5]}...")
        print(f"[{backend_name}] {func_name} completed successfully")
