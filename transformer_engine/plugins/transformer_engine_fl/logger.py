# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from .logger_manager import Logger, LoggerManager

def get_logger():
 return LoggerManager.get_instance().get_logger()


def print_once(message):
 LoggerManager.get_instance().print_once(message)


def debug_print_once(func_name: str, backend_name: str = "Backend", *args, **kwargs):
 LoggerManager.get_instance().debug_print_once(func_name, backend_name, *args, **kwargs)


GLOBAL_LOGGER = None
_GLOBAL_PRINTED_ONCE = None

__all__ = [
 "Logger",
 "LoggerManager",
 "get_logger",
 "print_once",
 "debug_print_once",
]
